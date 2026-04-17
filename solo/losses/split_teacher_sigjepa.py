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

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from solo.losses.mmd import mmd_regularization_loss
from solo.losses.sigreg import sigreg, sigreg_real


def predictive_loss(
    global_latents: Sequence[torch.Tensor],
    all_latents: Sequence[torch.Tensor],
) -> torch.Tensor:
    """LeJEPA-style predictive loss.

    Given ``V_g`` global views and ``V`` total views, this matches every view against
    the center induced by the global views:

        mu = mean(global_views)
        L_pred = mean_v ||mu - z_v||^2

    This reduces to the two-view MSE up to a constant scaling factor when no local
    views are present, while correctly extending to the ``2 global + 6 local``
    multi-crop recipe used in LeJEPA.
    """

    if len(global_latents) == 0:
        raise ValueError("predictive_loss requires at least one global latent tensor.")
    if len(all_latents) == 0:
        raise ValueError("predictive_loss requires at least one latent tensor.")

    global_stack = torch.stack(tuple(global_latents), dim=0)
    all_stack = torch.stack(tuple(all_latents), dim=0)
    centers = global_stack.mean(dim=0)
    return (all_stack - centers.unsqueeze(0)).square().mean()


def teacher_alignment_loss(
    aligned_globals: Sequence[torch.Tensor],
    teacher_globals: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Normalized teacher alignment loss on the compatible branch only."""

    if len(aligned_globals) == 0:
        raise ValueError("teacher_alignment_loss requires at least one aligned tensor.")
    if len(aligned_globals) != len(teacher_globals):
        raise ValueError(
            "teacher_alignment_loss expects aligned student and teacher sequences of the same length, "
            f"got {len(aligned_globals)} and {len(teacher_globals)}."
        )

    losses = [F.mse_loss(student, teacher) for student, teacher in zip(aligned_globals, teacher_globals)]
    return torch.stack(losses).mean()


def free_regularization_loss(
    free_views: Optional[Sequence[torch.Tensor]],
    global_step: int,
    num_slices: int = 256,
    num_points: int = 17,
    t_min: float = -5.0,
    t_max: float = 5.0,
    use_real: bool = False,
    ddp_sync: bool = True,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """SIGReg regularization applied independently to each free-branch view."""

    if free_views is None or len(free_views) == 0:
        raise ValueError("free_regularization_loss requires at least one free-branch tensor.")

    losses: List[torch.Tensor] = []
    fn = sigreg_real if use_real else sigreg
    for view in free_views:
        if view is None or view.numel() == 0:
            losses.append(view.new_zeros(()) if view is not None else torch.tensor(0.0))
            continue
        losses.append(
            fn(
                view,
                global_step=global_step,
                num_slices=num_slices,
                num_points=num_points,
                t_min=t_min,
                t_max=t_max,
                ddp_sync=ddp_sync,
            )
        )

    return torch.stack(losses).mean(), losses


def split_teacher_sigjepa_loss(
    global_latents: Sequence[torch.Tensor],
    all_latents: Sequence[torch.Tensor],
    aligned_globals: Optional[Sequence[torch.Tensor]],
    teacher_globals: Optional[Sequence[torch.Tensor]],
    free_views: Optional[Sequence[torch.Tensor]],
    global_step: int,
    lambda_pred: float = 1.0,
    lambda_teacher: float = 1.0,
    lambda_sigreg: float = 0.05,
    lambda_mmd: float = 0.0,
    mmd_kernel: str = "energy",
    mmd_views: Optional[Sequence[torch.Tensor]] = None,
    num_slices: int = 256,
    num_points: int = 17,
    t_min: float = -5.0,
    t_max: float = 5.0,
    sigreg_use_real: bool = False,
    ddp_sync: bool = True,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    List[torch.Tensor],
    List[torch.Tensor],
]:
    """Returns total SSL loss and its main components.

    Output order:
        total_ssl_loss, pred_loss, teacher_loss, free_sigreg_loss,
        mmd_loss, free_sigreg_loss_per_view, mmd_loss_per_view
    """

    if len(global_latents) == 0:
        raise ValueError("split_teacher_sigjepa_loss requires at least one global latent tensor.")
    if len(all_latents) == 0:
        raise ValueError("split_teacher_sigjepa_loss requires at least one latent tensor.")

    pred = predictive_loss(global_latents=global_latents, all_latents=all_latents)
    ref = all_latents[0]

    if aligned_globals is not None and teacher_globals is not None:
        teacher = teacher_alignment_loss(aligned_globals=aligned_globals, teacher_globals=teacher_globals)
    else:
        teacher = ref.new_zeros(())

    if free_views is not None:
        free, free_per_view = free_regularization_loss(
            free_views,
            global_step=global_step,
            num_slices=num_slices,
            num_points=num_points,
            t_min=t_min,
            t_max=t_max,
            use_real=sigreg_use_real,
            ddp_sync=ddp_sync,
        )
    else:
        free = ref.new_zeros(())
        free_per_view = []

    if lambda_mmd > 0:
        if mmd_views is None:
            mmd_views = all_latents
        mmd, mmd_per_view = mmd_regularization_loss(
            mmd_views,
            global_step=global_step,
            kernel=mmd_kernel,
            ddp_sync=ddp_sync,
        )
    else:
        mmd = ref.new_zeros(())
        mmd_per_view = []

    total = (
        lambda_pred * pred
        + lambda_teacher * teacher
        + lambda_sigreg * free
        + lambda_mmd * mmd
    )
    return total, pred, teacher, free, mmd, free_per_view, mmd_per_view
