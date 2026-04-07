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


def _off_diagonal_covariance_energy(z: torch.Tensor) -> torch.Tensor:
    if z.ndim != 2 or z.size(0) <= 1 or z.size(1) <= 1:
        return z.new_zeros(())
    z = z - z.mean(dim=0)
    cov = (z.T @ z) / max(z.size(0) - 1, 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return off_diag.pow(2).mean()


class TeacherSIGJEPAFull(BaseMethod):
    """Barebones full-latent teacher + SIGReg baseline."""

    def __init__(self, cfg: omegaconf.DictConfig):
        super().__init__(cfg)

        self.lambda_pred: float = float(cfg.method_kwargs.lambda_pred)
        self.lambda_teacher: float = float(cfg.method_kwargs.lambda_teacher)
        self.lambda_sigreg: float = float(cfg.method_kwargs.lambda_sigreg)

        self.projector_type: str = cfg.method_kwargs.projector_type
        self.proj_output_dim: int = int(cfg.method_kwargs.proj_output_dim)
        self.sigreg_num_slices: int = int(cfg.method_kwargs.sigreg_num_slices)
        self.sigreg_num_points: int = int(cfg.method_kwargs.sigreg_num_points)
        self.sigreg_t_min: float = float(cfg.method_kwargs.sigreg_t_min)
        self.sigreg_t_max: float = float(cfg.method_kwargs.sigreg_t_max)
        self.sigreg_use_real: bool = bool(cfg.method_kwargs.sigreg_use_real)
        self.teacher_use_same_views: bool = bool(cfg.method_kwargs.teacher_use_same_views)

        if self.num_large_crops != 2:
            raise ValueError(
                f"TeacherSIGJEPAFull requires exactly 2 large crops, got {self.num_large_crops}."
            )
        if not self.teacher_use_same_views:
            raise ValueError(
                "Only teacher_use_same_views=True is supported in the barebones full-latent baseline."
            )
        if self.projector_type != "mlp":
            raise ValueError(
                f"Unsupported projector_type '{self.projector_type}'. Only 'mlp' is supported."
            )

        proj_hidden_dim: int = int(cfg.method_kwargs.proj_hidden_dim)
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, self.proj_output_dim),
        )

        teacher_branch_requested = self.lambda_teacher > 0
        configured_teacher_dim = int(cfg.method_kwargs.teacher_output_dim)
        if teacher_branch_requested:
            self.teacher = build_teacher(cfg)
            self.teacher.requires_grad_(False)
            self.teacher.eval()
            self.teacher_dim = int(getattr(self.teacher, "output_dim", configured_teacher_dim))
        else:
            self.teacher = None
            self.teacher_dim = configured_teacher_dim

        self.align_head = (
            nn.Linear(self.proj_output_dim, self.teacher_dim, bias=False)
            if teacher_branch_requested
            else None
        )

        if teacher_branch_requested and self.teacher is not None:
            print(
                "TeacherSIGJEPAFull teacher configured:",
                f"backend={cfg.method_kwargs.teacher_backend},",
                f"model_id={cfg.method_kwargs.teacher_model_id},",
                f"teacher_dim={self.teacher_dim},",
                f"pooling={cfg.method_kwargs.teacher_pooling},",
                f"chunk_size={cfg.method_kwargs.teacher_chunk_size}",
            )

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        cfg = super(TeacherSIGJEPAFull, TeacherSIGJEPAFull).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")

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
        cfg.method_kwargs.teacher_output_dim = int(
            omegaconf_select(cfg, "method_kwargs.teacher_output_dim", teacher_defaults["output_dim"])
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
        return self.lambda_teacher > 0 and self.align_head is not None

    @property
    def use_sigreg_branch(self) -> bool:
        return self.lambda_sigreg > 0

    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        out = super().forward(X)
        z = self.projector(out["feats"])
        out.update({"z": z})
        if self.projector_classifier is not None:
            out["projector_logits"] = self.projector_classifier(z.detach())
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]

        z1, z2 = out["z"]
        _img_indexes, X, _targets = batch
        X = [X] if isinstance(X, torch.Tensor) else X
        x1, x2 = X[:2]

        if self.use_teacher_branch:
            assert self.align_head is not None
            a1 = F.normalize(self.align_head(z1), dim=1)
            a2 = F.normalize(self.align_head(z2), dim=1)
            with torch.no_grad():
                if self.teacher is None:
                    raise RuntimeError("Teacher branch is enabled but no frozen teacher is available.")
                t1 = F.normalize(self.teacher(x1), dim=1)
                t2 = F.normalize(self.teacher(x2), dim=1)
        else:
            a1 = a2 = t1 = t2 = None

        ssl_loss, pred_loss, teacher_loss, sigreg_loss, sigreg_z1, sigreg_z2 = (
            split_teacher_sigjepa_loss(
                z1=z1,
                z2=z2,
                a1=a1,
                a2=a2,
                t1=t1,
                t2=t2,
                z_f1=z1 if self.use_sigreg_branch else None,
                z_f2=z2 if self.use_sigreg_branch else None,
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

        z_norm = 0.5 * (z1.norm(dim=1).mean() + z2.norm(dim=1).mean())

        self.log("train_teacher_sigjepa_full_loss", ssl_loss, on_epoch=True, sync_dist=True)
        self.log("train_pred_loss", pred_loss, on_epoch=True, sync_dist=True)
        self.log("train_teacher_loss", teacher_loss, on_epoch=True, sync_dist=True)
        self.log("train_sigreg_loss", sigreg_loss, on_epoch=True, sync_dist=True)
        self.log("train_teacher_cosine", teacher_cosine, on_epoch=True, sync_dist=True)
        self.log("train_z_norm", z_norm, on_epoch=True, sync_dist=True)

        if self.global_step % self.logging_interval == 0:
            var_z = 0.5 * (z1.var(dim=0).mean() + z2.var(dim=0).mean())
            cov_full_z = 0.5 * (
                _off_diagonal_covariance_energy(z1) + _off_diagonal_covariance_energy(z2)
            )
            self.log("train_var_z", var_z, on_epoch=True, sync_dist=True)
            self.log("train_cov_full_z", cov_full_z, on_epoch=True, sync_dist=True)
            self.log("train_sigreg_z1", sigreg_z1, on_epoch=True, sync_dist=True)
            self.log("train_sigreg_z2", sigreg_z2, on_epoch=True, sync_dist=True)

        total = ssl_loss + class_loss + projector_class_loss
        return total
