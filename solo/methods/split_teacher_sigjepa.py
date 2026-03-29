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
from solo.utils.teacher import build_teacher


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

        self.teacher = build_teacher(cfg)
        self.teacher.requires_grad_(False)
        self.teacher.eval()
        self.teacher_dim: int = int(getattr(self.teacher, "output_dim", cfg.method_kwargs.teacher_output_dim))

        self.align_head = (
            nn.Linear(self.compatible_dim, self.teacher_dim, bias=False)
            if self.compatible_dim > 0
            else None
        )

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
            cfg, "method_kwargs.teacher_backend", "hf_ijepa"
        )
        cfg.method_kwargs.teacher_model_id = omegaconf_select(
            cfg, "method_kwargs.teacher_model_id", "facebook/ijepa_vith14_1k"
        )
        cfg.method_kwargs.teacher_local_dir = omegaconf_select(
            cfg, "method_kwargs.teacher_local_dir", None
        )
        cfg.method_kwargs.teacher_pooling = omegaconf_select(
            cfg, "method_kwargs.teacher_pooling", "mean"
        )
        cfg.method_kwargs.teacher_output_dim = omegaconf_select(
            cfg, "method_kwargs.teacher_output_dim", 1280
        )
        cfg.method_kwargs.teacher_chunk_size = omegaconf_select(
            cfg, "method_kwargs.teacher_chunk_size", 16
        )
        cfg.method_kwargs.teacher_use_same_views = omegaconf_select(
            cfg, "method_kwargs.teacher_use_same_views", True
        )
        cfg.method_kwargs.teacher_eager_load = omegaconf_select(
            cfg, "method_kwargs.teacher_eager_load", False
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

        _, X, _targets = batch
        X = [X] if isinstance(X, torch.Tensor) else X
        x1, x2 = X[:2]

        if self.use_teacher_branch:
            assert self.align_head is not None
            a1 = F.normalize(self.align_head(z_c1), dim=1)
            a2 = F.normalize(self.align_head(z_c2), dim=1)
            with torch.no_grad():
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
