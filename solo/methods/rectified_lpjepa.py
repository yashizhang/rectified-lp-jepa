# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

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
from solo.losses.rectified_lpjepa import (
    rectified_lp_jepa_multicrop_loss,
    determine_sigma_for_lp_dist,
    choose_sigma_for_unit_var,
)
from solo.utils.metrics import l1_sparsity_metric, l0_sparsity_metric
from solo.methods.base import BaseMethod
from solo.utils.misc import omegaconf_select

import math

# =========================
# Projection Vector Generation
# =========================
class Projections:
    @staticmethod
    def generate_random_projections(num_projections, D, device=None, dtype=None):
        """Generates a set of random, normalized projection vectors on the unit sphere."""
        P_directions = torch.randn(num_projections, D, device=device, dtype=dtype)
        P_directions = P_directions / torch.norm(P_directions, dim=1, keepdim=True)
        return P_directions

    @staticmethod
    def generate_svd_projections(z):
        """Computes right-singular vectors (V^T) of a centered feature matrix."""
        with torch.amp.autocast('cuda', enabled=False):
            z = z.detach().float()
            z_centered = z - z.mean(dim=0)

            try:
                _, _, Vt = torch.linalg.svd(z_centered, full_matrices=False)
                return Vt
            except Exception:
                B, D = z_centered.shape
                k = min(B, D)
                A = torch.matmul(z_centered.T, z_centered)
                X = torch.randn(D, k, device=z.device, dtype=torch.float32)

                try:
                    _, V = torch.lobpcg(A, X=X, largest=True)
                    return V.T
                except Exception:
                    _, V = torch.linalg.eigh(A)
                    return V.T.flip(0)[:k]

    @staticmethod
    def get_projection_vectors(
        z_views,
        num_projections: int,
        projection_vectors_type: str,
        proj_output_dim: int,
    ):
        """Returns projection vectors for an arbitrary number of views.

        For random projections, a single projection matrix is shared across all views.
        For SVD-based variants, one projection matrix is returned per view.
        """

        if isinstance(z_views, torch.Tensor):
            z_views = [z_views]
        z_views = list(z_views)
        if len(z_views) == 0:
            raise ValueError('At least one view is required to build projection vectors.')

        if projection_vectors_type == 'random':
            ref = z_views[0]
            return Projections.generate_random_projections(
                num_projections, proj_output_dim, device=ref.device, dtype=ref.dtype
            )

        if projection_vectors_type not in {
            'torch_svd_and_random',
            'torch_svd_bottom_half_eigen_and_random',
        }:
            raise ValueError(f'Unsupported projection_vectors_type: {projection_vectors_type}')

        projections = []
        for z in z_views:
            Vt = Projections.generate_svd_projections(z)
            if projection_vectors_type == 'torch_svd_bottom_half_eigen_and_random':
                Vt = Vt[Vt.size(0) // 2 :]

            if num_projections < Vt.size(0):
                per_view_proj = Vt[:num_projections]
            elif num_projections == Vt.size(0):
                per_view_proj = Vt
            else:
                random_projs = Projections.generate_random_projections(
                    num_projections - Vt.size(0),
                    proj_output_dim,
                    device=z.device,
                    dtype=z.dtype,
                )
                per_view_proj = torch.vstack([Vt.to(device=z.device, dtype=z.dtype), random_projs])

            projections.append(per_view_proj)

        return projections


# =========================
# Rectified LpJEPA Method
# =========================
class RectifiedLpJEPA(BaseMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """
        Implements Rectified LpJEPA: Joint-Embedding Predictive Architectures
        with Sparse and Maximum-Entropy Representations.
        """
        super().__init__(cfg)

        # Loss weights
        self.invariance_loss_weight: float = cfg.method_kwargs.invariance_loss_weight
        self.rdm_reg_loss_weight: float = cfg.method_kwargs.rdm_reg_loss_weight

        # Distribution and Projection parameters
        self.target_distribution: str = cfg.method_kwargs.target_distribution
        self.num_projections = cfg.method_kwargs.num_projections
        self.projection_vectors_type: str = cfg.method_kwargs.projection_vectors_type
        self.mean_shift_value: float = cfg.method_kwargs.mean_shift_value
        self.lp_norm_parameter: float = cfg.method_kwargs.lp_norm_parameter
        
        # Determine target scale (chosen_sigma)
        self.mode_of_sigma: str = cfg.method_kwargs.mode_of_sigma
        if self.mode_of_sigma == "sigma_GN":
            # Scale GN_p to unit variance before rectification
            self.chosen_sigma = determine_sigma_for_lp_dist(self.lp_norm_parameter)
        elif self.mode_of_sigma == "sigma_RGN":
            # Scale GN_p such that ReLU(GN_p) has unit variance
            self.chosen_sigma = choose_sigma_for_unit_var(self.lp_norm_parameter, self.mean_shift_value)
        else:
            raise ValueError(f"Invalid mode of sigma: {self.mode_of_sigma}")
        
        print(f"Chosen sigma for {self.target_distribution} with mean shift {self.mean_shift_value} and p_norm {self.lp_norm_parameter} is {self.chosen_sigma}")

        # Projector configuration
        proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim: int = cfg.method_kwargs.proj_output_dim
        self.proj_output_dim = proj_output_dim
        self.projector_type: str = cfg.method_kwargs.projector_type

        # Define Projector Architecture
        if self.projector_type == "mlp":
            # Standard 3-layer MLP projector
            self.projector = nn.Sequential(
                nn.Linear(self.features_dim, proj_hidden_dim),
                nn.BatchNorm1d(proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(proj_hidden_dim, proj_hidden_dim),
                nn.BatchNorm1d(proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(proj_hidden_dim, proj_output_dim),
            )
        elif self.projector_type == "rectified_mlp":
            # 3-layer MLP projector with final ReLU for non-negativity
            self.projector = nn.Sequential(
                nn.Linear(self.features_dim, proj_hidden_dim),
                nn.BatchNorm1d(proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(proj_hidden_dim, proj_hidden_dim),
                nn.BatchNorm1d(proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(proj_hidden_dim, proj_output_dim),
                nn.ReLU(),
            )
        else:
            raise ValueError(f"Invalid projector type: {self.projector_type}")

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """
        Adds method-specific default values and checks to the config.
        """
        cfg = super(RectifiedLpJEPA, RectifiedLpJEPA).add_and_assert_specific_cfg(cfg)

        # Default loss weights
        cfg.method_kwargs.invariance_loss_weight = omegaconf_select(cfg, "method_kwargs.invariance_loss_weight", 25.0)
        cfg.method_kwargs.rdm_reg_loss_weight = omegaconf_select(cfg, "method_kwargs.rdm_reg_loss_weight", 125.0)
        
        # Default distribution/projection settings
        cfg.method_kwargs.num_projections = omegaconf_select(cfg, "method_kwargs.num_projections", 8192)
        cfg.method_kwargs.projection_vectors_type = omegaconf_select(cfg, "method_kwargs.projection_vectors_type", "random")
        cfg.method_kwargs.mean_shift_value = omegaconf_select(cfg, "method_kwargs.mean_shift_value", 0.0)
        cfg.method_kwargs.lp_norm_parameter = omegaconf_select(cfg, "method_kwargs.lp_norm_parameter", 1.0)
        cfg.method_kwargs.mode_of_sigma = omegaconf_select(cfg, "method_kwargs.mode_of_sigma", "sigma_GN")
        cfg.method_kwargs.projector_type = omegaconf_select(cfg, "method_kwargs.projector_type", "rectified_mlp")

        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        """
        Returns the list of learnable parameters for the optimizer.
        """
        extra_learnable_params = [{"name": "projector", "params": self.projector.parameters()}]
        return super().learnable_params + extra_learnable_params

    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """
        Performs the forward pass: Backbone -> Projector.
        """
        out = super().forward(X)
        z = self.projector(out["feats"])
        out.update({"z": z})
        
        # Optional projector classifier for online evaluation
        if self.projector_classifier is not None:
            projector_logits = self.projector_classifier(z.detach())
            out.update({"projector_logits": projector_logits})
        return out

    def multicrop_forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Forward pass for local crops so multi-crop training exposes projector outputs."""

        if not self.no_channel_last:
            X = X.to(memory_format=torch.channels_last)
        feats = self.backbone(X)
        z = self.projector(feats)
        return {"feats": feats, "z": z}

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Main training step with optional LeJEPA-style multi-crop support."""

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]

        z_views = out["z"]
        if isinstance(z_views, torch.Tensor):
            z_views = [z_views]
        else:
            z_views = list(z_views)

        if len(z_views) < self.num_large_crops:
            raise ValueError(
                f"Expected at least {self.num_large_crops} global views, got {len(z_views)} total views."
            )

        global_z_views = z_views[: self.num_large_crops]
        do_log = self.global_step % self.logging_interval == 0

        projection_vectors = Projections.get_projection_vectors(
            z_views,
            self.num_projections,
            self.projection_vectors_type,
            self.proj_output_dim,
        )

        from solo.utils.misc import gather

        if len(z_views) == 1:
            gathered_z_views = [gather(z_views[0])]
        else:
            stacked_views = torch.stack(z_views, dim=0)
            gathered_stacked_views = gather(stacked_views, dim=1)
            gathered_z_views = [gathered_stacked_views[i] for i in range(gathered_stacked_views.size(0))]

        loss_val, sim_l, reg_l = rectified_lp_jepa_multicrop_loss(
            global_views=global_z_views,
            all_views=z_views,
            gathered_views=gathered_z_views,
            projection_vectors=projection_vectors,
            target_distribution=self.target_distribution,
            invariance_loss_weight=self.invariance_loss_weight,
            rdm_reg_loss_weight=self.rdm_reg_loss_weight,
            mean_shift_value=self.mean_shift_value,
            lp_norm_parameter=self.lp_norm_parameter,
            chosen_sigma=self.chosen_sigma,
        )

        self.log("train_rectified_lp_jepa_loss", loss_val, on_epoch=True, sync_dist=True)
        self.log("train_invariance_loss", sim_l, on_epoch=True, sync_dist=True)
        self.log("train_rdm_reg_loss", reg_l, on_epoch=True, sync_dist=True)

        if do_log:
            from solo.utils.metrics import variance_loss, covariance_loss

            z1_gathered = gathered_z_views[0]
            z2_gathered = gathered_z_views[1] if len(gathered_z_views) > 1 else gathered_z_views[0]
            self.log("train_variance_loss", variance_loss(z1_gathered, z2_gathered), on_epoch=True, sync_dist=True)
            self.log("train_covariance_loss", covariance_loss(z1_gathered, z2_gathered), on_epoch=True, sync_dist=True)

            mean_l1 = sum(l1_sparsity_metric(z) for z in z_views) / len(z_views)
            mean_l0 = sum(l0_sparsity_metric(z) for z in z_views) / len(z_views)
            self.log("train_l1_sparsity_metric", mean_l1, on_epoch=True, sync_dist=True)
            self.log("train_l0_sparsity_metric", mean_l0, on_epoch=True, sync_dist=True)

        projector_class_loss = torch.tensor(0.0, device=self.device)
        if self.projector_classifier is not None:
            _, _, targets = batch
            proj_metrics = [self._projector_classifier_step(z, targets) for z in global_z_views]
            proj_metrics = [m for m in proj_metrics if m]
            if proj_metrics:
                projector_class_loss = sum(m["proj_loss"] for m in proj_metrics) / len(proj_metrics)
                if do_log:
                    self.log("train_proj_loss", projector_class_loss, on_epoch=True, sync_dist=True)
                    self.log(
                        "train_proj_acc1",
                        sum(m["proj_acc1"] for m in proj_metrics) / len(proj_metrics),
                        on_epoch=True,
                        sync_dist=True,
                    )

        return loss_val + class_loss + projector_class_loss
