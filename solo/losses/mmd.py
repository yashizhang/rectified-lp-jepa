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

"""MMD regularization losses for SIGJEPA-style methods.

The loss samples a standard Gaussian target with the same shape as the model
embeddings and compares the two empirical distributions with GeomLoss' MMD-like
sample losses: energy, gaussian, or laplacian.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List, Sequence, Tuple

import torch
import torch.distributed as dist

from solo.utils.misc import gather

VALID_MMD_KERNELS = ("energy", "gaussian", "laplacian")


def _work_dtype(x: torch.Tensor) -> torch.dtype:
    if x.dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return x.dtype


def _device_type(device: torch.device) -> str:
    return device.type if isinstance(device, torch.device) else str(device)


def _sync_samples(samples: torch.Tensor, ddp_sync: bool) -> torch.Tensor:
    if (
        ddp_sync
        and dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    ):
        return gather(samples, dim=0)
    return samples


@lru_cache(maxsize=None)
def _geomloss_samples_loss(kernel: str):
    if kernel not in VALID_MMD_KERNELS:
        raise ValueError(
            f"Unsupported MMD kernel '{kernel}'. Expected one of {VALID_MMD_KERNELS}."
        )

    try:
        from geomloss import SamplesLoss
    except ImportError as exc:
        raise ImportError(
            "lambda_mmd > 0 requires the 'geomloss' package. Install it with "
            "`pip install geomloss` or install this project with its updated requirements."
        ) from exc

    return SamplesLoss(loss=kernel)


def _standard_normal_like(
    x: torch.Tensor,
    global_step: int,
    view_idx: int,
) -> torch.Tensor:
    """Samples N(0, 1) with deterministic per-step/per-view randomness.

    After optional DDP gathering, all ranks see the same gathered embedding tensor.
    Seeding the local generator from global_step and view_idx gives identical
    Gaussian targets on all ranks, which keeps the MMD loss/logs deterministic
    across DDP workers while preserving gradients through the gathered embeddings.
    """

    generator = torch.Generator(device=_device_type(x.device))
    # Large odd constants decorrelate views without relying on Python's salted hash.
    seed = 13_371 + int(global_step) * 1_000_003 + int(view_idx) * 97_409
    generator.manual_seed(seed)
    return torch.randn(
        x.shape,
        generator=generator,
        device=x.device,
        dtype=x.dtype,
    )


def mmd_loss(
    x: torch.Tensor,
    global_step: int,
    kernel: str = "energy",
    view_idx: int = 0,
    ddp_sync: bool = True,
) -> torch.Tensor:
    """Compares embeddings against a same-shaped N(0, 1) target via GeomLoss.

    Args:
        x: Embedding tensor with shape ``[batch, dim]``.
        global_step: Current trainer global step. Used only to seed the Gaussian
            target deterministically.
        kernel: One of ``energy``, ``gaussian``, or ``laplacian``.
        view_idx: View index used to decorrelate the sampled target across crops.
        ddp_sync: If true, gather embeddings across DDP ranks before computing MMD.
    """

    if x.ndim != 2:
        raise ValueError(f"mmd_loss expects a 2D tensor [N, K], got shape {tuple(x.shape)}.")
    if x.size(1) == 0 or x.numel() == 0:
        return x.new_zeros(())

    kernel = str(kernel).lower()
    work_dtype = _work_dtype(x)
    x = x.to(dtype=work_dtype)
    x = _sync_samples(x, ddp_sync=ddp_sync)
    target = _standard_normal_like(x, global_step=global_step, view_idx=view_idx).detach()

    loss_fn = _geomloss_samples_loss(kernel)
    return loss_fn(x, target)


def mmd_regularization_loss(
    views: Sequence[torch.Tensor],
    global_step: int,
    kernel: str = "energy",
    ddp_sync: bool = True,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Applies MMD regularization independently to each embedding view."""

    if views is None or len(views) == 0:
        raise ValueError("mmd_regularization_loss requires at least one embedding tensor.")

    losses: List[torch.Tensor] = []
    for view_idx, view in enumerate(views):
        if view is None or view.numel() == 0:
            losses.append(view.new_zeros(()) if view is not None else torch.tensor(0.0))
            continue
        losses.append(
            mmd_loss(
                view,
                global_step=global_step,
                kernel=kernel,
                view_idx=view_idx,
                ddp_sync=ddp_sync,
            )
        )

    return torch.stack(losses).mean(), losses
