# Copyright 2023 solo-learn development team.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import Any, Dict, List, Sequence

import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F

from solo.losses.split_teacher_sigjepa import split_teacher_sigjepa_loss
from solo.methods.base import BaseMethod
from solo.utils.misc import omegaconf_select
from solo.utils.teacher import build_teacher, get_teacher_backend_defaults
from solo.utils.teacher_prefetch import (
    TeacherPrefetchReader,
    build_teacher_prefetch_fingerprint,
    resolve_teacher_prefetch_epoch,
)


def _off_diagonal_covariance_energy(z: torch.Tensor) -> torch.Tensor:
    if z.ndim != 2 or z.size(0) <= 1 or z.size(1) <= 1:
        return z.new_zeros(())
    z = z - z.mean(dim=0)
    cov = (z.T @ z) / max(z.size(0) - 1, 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return off_diag.pow(2).mean()


class SplitTeacherSIGJEPA(BaseMethod):
    """Minimal split-teacher SIGReg JEPA implementation from IDEA.md."""

    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)

        self.lambda_pred: float = float(cfg.method_kwargs.lambda_pred)
        self.lambda_teacher: float = float(cfg.method_kwargs.lambda_teacher)
        self.lambda_sigreg: float = float(cfg.method_kwargs.lambda_sigreg)

        self.projector_type: str = cfg.method_kwargs.projector_type
        self.compatible_dim: int = int(cfg.method_kwargs.compatible_dim)
        self.free_dim: int = int(cfg.method_kwargs.free_dim)
        self.proj_output_dim: int = int(cfg.method_kwargs.proj_output_dim)

        self.sigreg_num_slices: int = int(cfg.method_kwargs.sigreg_num_slices)
        self.sigreg_num_points: int = int(cfg.method_kwargs.sigreg_num_points)
        self.sigreg_t_min: float = float(cfg.method_kwargs.sigreg_t_min)
        self.sigreg_t_max: float = float(cfg.method_kwargs.sigreg_t_max)
        self.sigreg_use_real: bool = bool(cfg.method_kwargs.sigreg_use_real)
        self.teacher_use_same_views: bool = bool(cfg.method_kwargs.teacher_use_same_views)

        assert self.compatible_dim >= 0, "compatible_dim must be >= 0."
        assert self.free_dim >= 0, "free_dim must be >= 0."
        assert (
            self.compatible_dim + self.free_dim == self.proj_output_dim
        ), "compatible_dim + free_dim must equal proj_output_dim."
        if self.lambda_teacher > 0:
            assert self.compatible_dim > 0, "compatible_dim must be > 0 when lambda_teacher > 0."
        if self.lambda_sigreg > 0:
            assert self.free_dim > 0, "free_dim must be > 0 when lambda_sigreg > 0."
        if self.num_large_crops != 2:
            raise ValueError(
                f"SplitTeacherSIGJEPA requires exactly 2 large crops, got {self.num_large_crops}."
            )
        if not self.teacher_use_same_views:
            raise ValueError(
                "Only teacher_use_same_views=True is supported in the v0 implementation."
            )

        proj_hidden_dim: int = int(cfg.method_kwargs.proj_hidden_dim)
        if self.projector_type != "mlp":
            raise ValueError(
                f"Unsupported projector_type '{self.projector_type}'. Only 'mlp' is supported in v0."
            )

        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, self.proj_output_dim),
        )

        self.teacher_prefetch_enabled: bool = bool(
            omegaconf_select(cfg, "method_kwargs.teacher_prefetch.enabled", False)
        )
        self.teacher_prefetch_epoch_mode: str = str(
            omegaconf_select(cfg, "method_kwargs.teacher_prefetch.epoch_mode", "wrap")
        )
        self.teacher_prefetch_cache = None
        self._teacher_prefetch_epoch_ref = None

        teacher_branch_requested = self.lambda_teacher > 0 and self.compatible_dim > 0
        configured_teacher_dim = int(cfg.method_kwargs.teacher_output_dim)
        if teacher_branch_requested and self.teacher_prefetch_enabled:
            cache_dir = omegaconf_select(cfg, "method_kwargs.teacher_prefetch.cache_dir", None)
            if cache_dir is None:
                raise ValueError(
                    "teacher_prefetch.enabled=True requires method_kwargs.teacher_prefetch.cache_dir."
                )
            self.teacher = None
            self.teacher_prefetch_cache = TeacherPrefetchReader(
                cache_dir,
                expected_fingerprint=build_teacher_prefetch_fingerprint(cfg),
                expected_num_views=self.num_large_crops,
                expected_teacher_dim=configured_teacher_dim,
            )
            self.teacher_dim = int(self.teacher_prefetch_cache.teacher_dim)
        elif teacher_branch_requested:
            self.teacher = build_teacher(cfg)
            self.teacher.requires_grad_(False)
            self.teacher.eval()
            self.teacher_dim = int(getattr(self.teacher, "output_dim", configured_teacher_dim))
        else:
            self.teacher = None
            self.teacher_dim = configured_teacher_dim

        self.align_head = (
            nn.Linear(self.compatible_dim, self.teacher_dim, bias=False)
            if self.compatible_dim > 0
            else None
        )

        if self.teacher_prefetch_cache is not None:
            print(
                "SplitTeacherSIGJEPA teacher prefetch configured:",
                f"cache_dir={omegaconf_select(cfg, 'method_kwargs.teacher_prefetch.cache_dir', None)},",
                f"cache_epochs={self.teacher_prefetch_cache.num_epochs},",
                f"epoch_mode={self.teacher_prefetch_epoch_mode},",
                f"teacher_dim={self.teacher_dim}",
            )
        elif teacher_branch_requested and self.teacher is not None:
            print(
                "SplitTeacherSIGJEPA teacher configured:",
                f"backend={cfg.method_kwargs.teacher_backend},",
                f"model_id={cfg.method_kwargs.teacher_model_id},",
                f"teacher_dim={self.teacher_dim},",
                f"pooling={cfg.method_kwargs.teacher_pooling},",
                f"chunk_size={cfg.method_kwargs.teacher_chunk_size}",
            )

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        cfg = super(SplitTeacherSIGJEPA, SplitTeacherSIGJEPA).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")

        cfg.method_kwargs.compatible_dim = omegaconf_select(cfg, "method_kwargs.compatible_dim", 512)
        cfg.method_kwargs.free_dim = omegaconf_select(
            cfg,
            "method_kwargs.free_dim",
            int(cfg.method_kwargs.proj_output_dim) - int(cfg.method_kwargs.compatible_dim),
        )

        cfg.method_kwargs.lambda_pred = omegaconf_select(cfg, "method_kwargs.lambda_pred", 1.0)
        cfg.method_kwargs.lambda_teacher = omegaconf_select(cfg, "method_kwargs.lambda_teacher", 1.0)
        cfg.method_kwargs.lambda_sigreg = omegaconf_select(cfg, "method_kwargs.lambda_sigreg", 0.05)

        cfg.method_kwargs.sigreg_num_slices = omegaconf_select(
            cfg, "method_kwargs.sigreg_num_slices", 256
        )
        cfg.method_kwargs.sigreg_num_points = omegaconf_select(
            cfg, "method_kwargs.sigreg_num_points", 17
        )
        cfg.method_kwargs.sigreg_t_min = omegaconf_select(cfg, "method_kwargs.sigreg_t_min", -5.0)
        cfg.method_kwargs.sigreg_t_max = omegaconf_select(cfg, "method_kwargs.sigreg_t_max", 5.0)
        cfg.method_kwargs.sigreg_use_real = omegaconf_select(
            cfg, "method_kwargs.sigreg_use_real", False
        )

        cfg.method_kwargs.teacher_backend = omegaconf_select(
            cfg, "method_kwargs.teacher_backend", "hf_dinov2"
        )
        teacher_defaults = get_teacher_backend_defaults(
            cfg.method_kwargs.teacher_backend,
            omegaconf_select(cfg, "method_kwargs.teacher_model_id", None),
        )
        cfg.method_kwargs.teacher_model_id = omegaconf_select(
            cfg, "method_kwargs.teacher_model_id", teacher_defaults["model_id"]
        )
        cfg.method_kwargs.teacher_local_dir = omegaconf_select(
            cfg, "method_kwargs.teacher_local_dir", None
        )
        cfg.method_kwargs.teacher_pooling = omegaconf_select(
            cfg, "method_kwargs.teacher_pooling", teacher_defaults["pooling"]
        )
        cfg.method_kwargs.teacher_output_dim = int(omegaconf_select(
            cfg, "method_kwargs.teacher_output_dim", teacher_defaults["output_dim"]
        ))
        cfg.method_kwargs.teacher_chunk_size = omegaconf_select(
            cfg, "method_kwargs.teacher_chunk_size", 16
        )
        cfg.method_kwargs.teacher_use_same_views = omegaconf_select(
            cfg, "method_kwargs.teacher_use_same_views", True
        )
        cfg.method_kwargs.teacher_eager_load = omegaconf_select(
            cfg, "method_kwargs.teacher_eager_load", False
        )

        cfg.method_kwargs.teacher_prefetch = omegaconf_select(
            cfg, "method_kwargs.teacher_prefetch", {}
        )
        cfg.method_kwargs.teacher_prefetch.enabled = omegaconf_select(
            cfg, "method_kwargs.teacher_prefetch.enabled", False
        )
        cfg.method_kwargs.teacher_prefetch.cache_dir = omegaconf_select(
            cfg,
            "method_kwargs.teacher_prefetch.cache_dir",
            f"teacher_prefetch_cache/{omegaconf_select(cfg, 'name', 'split_teacher_sigjepa')}",
        )
        cfg.method_kwargs.teacher_prefetch.num_epochs = int(
            omegaconf_select(cfg, "method_kwargs.teacher_prefetch.num_epochs", cfg.max_epochs)
        )
        cfg.method_kwargs.teacher_prefetch.epoch_mode = omegaconf_select(
            cfg, "method_kwargs.teacher_prefetch.epoch_mode", "wrap"
        )
        cfg.method_kwargs.teacher_prefetch.dtype = omegaconf_select(
            cfg, "method_kwargs.teacher_prefetch.dtype", "float16"
        )
        cfg.method_kwargs.teacher_prefetch.base_seed = int(
            omegaconf_select(cfg, "method_kwargs.teacher_prefetch.base_seed", cfg.seed)
        )
        cfg.method_kwargs.teacher_prefetch.batch_size = int(
            omegaconf_select(
                cfg,
                "method_kwargs.teacher_prefetch.batch_size",
                cfg.optimizer.batch_size,
            )
        )
        cfg.method_kwargs.teacher_prefetch.num_workers = int(
            omegaconf_select(
                cfg,
                "method_kwargs.teacher_prefetch.num_workers",
                cfg.data.num_workers,
            )
        )
        cfg.method_kwargs.teacher_prefetch.overwrite = bool(
            omegaconf_select(cfg, "method_kwargs.teacher_prefetch.overwrite", False)
        )

        cfg.method_kwargs.projector_type = omegaconf_select(
            cfg, "method_kwargs.projector_type", "mlp"
        )
        cfg.method_kwargs.add_projector_classifier = omegaconf_select(
            cfg, "method_kwargs.add_projector_classifier", True
        )

        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        extra = [{"name": "projector", "params": self.projector.parameters()}]
        if self.align_head is not None:
            extra.append({"name": "align_head", "params": self.align_head.parameters()})
        return super().learnable_params + extra

    @property
    def use_teacher_branch(self) -> bool:
        return self.lambda_teacher > 0 and self.compatible_dim > 0 and self.align_head is not None

    @property
    def use_free_branch(self) -> bool:
        return self.lambda_sigreg > 0 and self.free_dim > 0

    def attach_teacher_prefetch_epoch_ref(self, epoch_ref) -> None:
        self._teacher_prefetch_epoch_ref = epoch_ref

    def _resolve_teacher_prefetch_epoch(self, train_epoch: int) -> int:
        if self.teacher_prefetch_cache is None:
            return int(train_epoch)
        return resolve_teacher_prefetch_epoch(
            train_epoch=int(train_epoch),
            num_prefetch_epochs=int(self.teacher_prefetch_cache.num_epochs),
            epoch_mode=self.teacher_prefetch_epoch_mode,
        )

    def on_train_epoch_start(self) -> None:
        if self.teacher_prefetch_cache is None:
            return

        if self._teacher_prefetch_epoch_ref is None:
            raise RuntimeError(
                "teacher_prefetch is enabled, but no deterministic dataset epoch reference was "
                "attached to SplitTeacherSIGJEPA. main_pretrain.py should attach the epoch "
                "reference returned by the training dataset."
            )

        cache_epoch = self._resolve_teacher_prefetch_epoch(int(self.current_epoch))
        with self._teacher_prefetch_epoch_ref.get_lock():
            self._teacher_prefetch_epoch_ref.value = cache_epoch
        self.teacher_prefetch_cache.set_epoch(cache_epoch)

    def _get_prefetched_teacher_targets(
        self,
        img_indexes: torch.Tensor,
        device: torch.device,
    ) -> Sequence[torch.Tensor]:
        if self.teacher_prefetch_cache is None:
            raise RuntimeError("Teacher prefetch cache is not initialized.")

        cache_epoch = self._resolve_teacher_prefetch_epoch(int(self.current_epoch))
        self.teacher_prefetch_cache.set_epoch(cache_epoch)
        cached = self.teacher_prefetch_cache.fetch(
            img_indexes,
            device=device,
            dtype=torch.float32,
        )
        if cached.ndim != 3 or cached.size(1) < 2:
            raise ValueError(
                "Teacher prefetch cache must return a tensor with shape [B, num_views, teacher_dim] "
                f"and at least 2 views, got {tuple(cached.shape)}."
            )
        return F.normalize(cached[:, 0], dim=1), F.normalize(cached[:, 1], dim=1)

    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        out = super().forward(X)
        z = self.projector(out["feats"])
        z_c = z[:, : self.compatible_dim]
        z_f = z[:, self.compatible_dim :]

        out.update({
            "z": z,
            "z_c": z_c,
            "z_f": z_f,
        })

        if self.projector_classifier is not None:
            out["projector_logits"] = self.projector_classifier(z.detach())
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]

        z1, z2 = out["z"]
        z_c1, z_c2 = out["z_c"]
        z_f1, z_f2 = out["z_f"]

        img_indexes, X, _targets = batch
        X = [X] if isinstance(X, torch.Tensor) else X
        x1, x2 = X[:2]

        if self.use_teacher_branch:
            assert self.align_head is not None
            a1 = F.normalize(self.align_head(z_c1), dim=1)
            a2 = F.normalize(self.align_head(z_c2), dim=1)
            with torch.no_grad():
                if self.teacher_prefetch_cache is not None:
                    t1, t2 = self._get_prefetched_teacher_targets(img_indexes, device=x1.device)
                else:
                    if self.teacher is None:
                        raise RuntimeError(
                            "Teacher branch is enabled, but neither an online teacher nor a "
                            "teacher_prefetch cache is available."
                        )
                    t1 = F.normalize(self.teacher(x1), dim=1)
                    t2 = F.normalize(self.teacher(x2), dim=1)
        else:
            a1 = a2 = t1 = t2 = None

        ssl_loss, pred_loss, teacher_loss, free_loss, free_loss_zf1, free_loss_zf2 = (
            split_teacher_sigjepa_loss(
                z1=z1,
                z2=z2,
                a1=a1,
                a2=a2,
                t1=t1,
                t2=t2,
                z_f1=z_f1 if self.use_free_branch else None,
                z_f2=z_f2 if self.use_free_branch else None,
                global_step=self.global_step,
                lambda_pred=self.lambda_pred,
                lambda_teacher=self.lambda_teacher,
                lambda_sigreg=self.lambda_sigreg,
                num_slices=self.sigreg_num_slices,
                num_points=self.sigreg_num_points,
                t_min=self.sigreg_t_min,
                t_max=self.sigreg_t_max,
                sigreg_use_real=self.sigreg_use_real,
                ddp_sync=True,
            )
        )

        projector_class_loss = z1.new_zeros(())
        if self.projector_classifier is not None and "proj_loss" in out:
            projector_class_loss = sum(out["proj_loss"]) / len(out["proj_loss"])
            self.log("train_proj_loss", projector_class_loss, on_epoch=True, sync_dist=True)
            self.log(
                "train_proj_acc1",
                sum(out["proj_acc1"]) / len(out["proj_acc1"]),
                on_epoch=True,
                sync_dist=True,
            )
            self.log(
                "train_proj_acc5",
                sum(out["proj_acc5"]) / len(out["proj_acc5"]),
                on_epoch=True,
                sync_dist=True,
            )

        teacher_cosine = z1.new_zeros(())
        if self.use_teacher_branch and a1 is not None and a2 is not None and t1 is not None and t2 is not None:
            teacher_cosine = 0.5 * (
                (a1 * t1).sum(dim=1).mean() + (a2 * t2).sum(dim=1).mean()
            )

        zc_norm = z_c1.new_zeros(()) if self.compatible_dim == 0 else 0.5 * (
            z_c1.norm(dim=1).mean() + z_c2.norm(dim=1).mean()
        )
        zf_norm = z_f1.new_zeros(()) if self.free_dim == 0 else 0.5 * (
            z_f1.norm(dim=1).mean() + z_f2.norm(dim=1).mean()
        )

        self.log("train_split_teacher_sigjepa_loss", ssl_loss, on_epoch=True, sync_dist=True)
        self.log("train_pred_loss", pred_loss, on_epoch=True, sync_dist=True)
        self.log("train_teacher_loss", teacher_loss, on_epoch=True, sync_dist=True)
        self.log("train_sigreg_loss", free_loss, on_epoch=True, sync_dist=True)
        self.log("train_teacher_cosine", teacher_cosine, on_epoch=True, sync_dist=True)
        self.log("train_zc_norm", zc_norm, on_epoch=True, sync_dist=True)
        self.log("train_zf_norm", zf_norm, on_epoch=True, sync_dist=True)

        if self.global_step % self.logging_interval == 0:
            var_zc = z_c1.new_zeros(()) if self.compatible_dim == 0 else 0.5 * (
                z_c1.var(dim=0).mean() + z_c2.var(dim=0).mean()
            )
            var_zf = z_f1.new_zeros(()) if self.free_dim == 0 else 0.5 * (
                z_f1.var(dim=0).mean() + z_f2.var(dim=0).mean()
            )
            cov_full_z = 0.5 * (
                _off_diagonal_covariance_energy(z1) + _off_diagonal_covariance_energy(z2)
            )
            self.log("train_var_zc", var_zc, on_epoch=True, sync_dist=True)
            self.log("train_var_zf", var_zf, on_epoch=True, sync_dist=True)
            self.log("train_cov_full_z", cov_full_z, on_epoch=True, sync_dist=True)
            self.log("train_sigreg_zf1", free_loss_zf1, on_epoch=True, sync_dist=True)
            self.log("train_sigreg_zf2", free_loss_zf2, on_epoch=True, sync_dist=True)

        total = ssl_loss + class_loss + projector_class_loss
        return total
