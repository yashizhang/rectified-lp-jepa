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

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from solo.losses.sigreg import sigreg, sigreg_real


def predictive_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Minimal JEPA predictive loss on the full latent."""

    return F.mse_loss(z1, z2)


def teacher_alignment_loss(
    a1: torch.Tensor,
    a2: torch.Tensor,
    t1: torch.Tensor,
    t2: torch.Tensor,
) -> torch.Tensor:
    """Normalized teacher alignment loss on the compatible branch only."""

    if a1.numel() == 0 or a2.numel() == 0:
        return a1.new_zeros(())
    return 0.5 * (F.mse_loss(a1, t1) + F.mse_loss(a2, t2))


def free_regularization_loss(
    z_f1: Optional[torch.Tensor],
    z_f2: Optional[torch.Tensor],
    global_step: int,
    num_slices: int = 256,
    num_points: int = 17,
    t_min: float = -5.0,
    t_max: float = 5.0,
    use_real: bool = False,
    ddp_sync: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SIGReg regularization applied only to the free branch."""

    ref = z_f1 if z_f1 is not None else z_f2
    if ref is None:
        raise ValueError("At least one free-branch tensor must be provided.")

    zero = ref.new_zeros(())
    if z_f1 is None or z_f1.numel() == 0:
        free1 = zero
    else:
        fn = sigreg_real if use_real else sigreg
        free1 = fn(
            z_f1,
            global_step=global_step,
            num_slices=num_slices,
            num_points=num_points,
            t_min=t_min,
            t_max=t_max,
            ddp_sync=ddp_sync,
        )

    if z_f2 is None or z_f2.numel() == 0:
        free2 = zero
    else:
        fn = sigreg_real if use_real else sigreg
        free2 = fn(
            z_f2,
            global_step=global_step,
            num_slices=num_slices,
            num_points=num_points,
            t_min=t_min,
            t_max=t_max,
            ddp_sync=ddp_sync,
        )

    return 0.5 * (free1 + free2), free1, free2


def split_teacher_sigjepa_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    a1: Optional[torch.Tensor],
    a2: Optional[torch.Tensor],
    t1: Optional[torch.Tensor],
    t2: Optional[torch.Tensor],
    z_f1: Optional[torch.Tensor],
    z_f2: Optional[torch.Tensor],
    global_step: int,
    lambda_pred: float = 1.0,
    lambda_teacher: float = 1.0,
    lambda_sigreg: float = 0.05,
    num_slices: int = 256,
    num_points: int = 17,
    t_min: float = -5.0,
    t_max: float = 5.0,
    sigreg_use_real: bool = False,
    ddp_sync: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns total SSL loss and its main components.

    Output order:
        total_ssl_loss, pred_loss, teacher_loss, free_loss, free_loss_view1, free_loss_view2
    """

    pred = predictive_loss(z1, z2)

    if all(t is not None for t in (a1, a2, t1, t2)):
        teacher = teacher_alignment_loss(a1, a2, t1, t2)
    else:
        teacher = z1.new_zeros(())

    if z_f1 is not None or z_f2 is not None:
        free, free1, free2 = free_regularization_loss(
            z_f1,
            z_f2,
            global_step=global_step,
            num_slices=num_slices,
            num_points=num_points,
            t_min=t_min,
            t_max=t_max,
            use_real=sigreg_use_real,
            ddp_sync=ddp_sync,
        )
    else:
        free = free1 = free2 = z1.new_zeros(())

    total = lambda_pred * pred + lambda_teacher * teacher + lambda_sigreg * free
    return total, pred, teacher, free, free1, free2
