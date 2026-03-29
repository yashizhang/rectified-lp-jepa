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

import torch
import torch.distributed as dist
import torch.nn.functional as F

from solo.utils.misc import gather


def _work_dtype(x: torch.Tensor) -> torch.dtype:
    if x.dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return x.dtype


def _device_type(device: torch.device) -> str:
    return device.type if isinstance(device, torch.device) else str(device)


def sample_unit_sphere(
    dim: int,
    num_slices: int,
    global_step: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Samples deterministic random unit directions on the sphere.

    The generator is seeded from ``global_step`` so all ranks draw the same
    directions during DDP training.
    """

    if dim <= 0:
        return torch.empty(dim, num_slices, device=device, dtype=dtype)
    if num_slices <= 0:
        raise ValueError(f"num_slices must be > 0, got {num_slices}.")

    generator = torch.Generator(device=_device_type(device))
    generator.manual_seed(int(global_step))
    directions = torch.randn(
        dim,
        num_slices,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    return F.normalize(directions, dim=0)


def _sync_projected(projected: torch.Tensor, ddp_sync: bool) -> torch.Tensor:
    if (
        ddp_sync
        and dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    ):
        return gather(projected, dim=0)
    return projected


def sigreg(
    x: torch.Tensor,
    global_step: int,
    num_slices: int = 256,
    num_points: int = 17,
    t_min: float = -5.0,
    t_max: float = 5.0,
    ddp_sync: bool = True,
) -> torch.Tensor:
    """Computes the complex-valued SIGReg loss.

    This follows the LeJEPA-style characteristic-function regularizer from the
    IDEA.md spec while using differentiable all-gather for DDP synchronization.
    """

    if x.ndim != 2:
        raise ValueError(f"sigreg expects a 2D tensor [N, K], got shape {tuple(x.shape)}.")
    if x.size(1) == 0 or x.numel() == 0:
        return x.new_zeros(())
    if num_points < 2:
        raise ValueError(f"num_points must be >= 2, got {num_points}.")

    work_dtype = _work_dtype(x)
    x = x.to(dtype=work_dtype)
    device = x.device

    A = sample_unit_sphere(
        dim=x.size(1),
        num_slices=num_slices,
        global_step=global_step,
        device=device,
        dtype=work_dtype,
    )
    t = torch.linspace(t_min, t_max, num_points, device=device, dtype=work_dtype)
    target_cf = torch.exp(-0.5 * t.square())

    projected = x @ A
    projected = _sync_projected(projected, ddp_sync=ddp_sync)
    num_samples = projected.size(0)

    projected_t = projected.unsqueeze(-1) * t.view(1, 1, -1)
    ecf = torch.exp(1j * projected_t).mean(dim=0)

    err = (ecf - target_cf.view(1, -1)).abs().square() * target_cf.view(1, -1)
    per_slice = torch.trapz(err, t, dim=-1) * num_samples
    return per_slice.mean().real


def sigreg_real(
    x: torch.Tensor,
    global_step: int,
    num_slices: int = 256,
    num_points: int = 17,
    t_min: float = -5.0,
    t_max: float = 5.0,
    ddp_sync: bool = True,
) -> torch.Tensor:
    """Safer real-valued SIGReg variant for AMP-sensitive setups."""

    if x.ndim != 2:
        raise ValueError(f"sigreg_real expects a 2D tensor [N, K], got shape {tuple(x.shape)}.")
    if x.size(1) == 0 or x.numel() == 0:
        return x.new_zeros(())
    if num_points < 2:
        raise ValueError(f"num_points must be >= 2, got {num_points}.")

    work_dtype = _work_dtype(x)
    x = x.to(dtype=work_dtype)
    device = x.device

    A = sample_unit_sphere(
        dim=x.size(1),
        num_slices=num_slices,
        global_step=global_step,
        device=device,
        dtype=work_dtype,
    )
    t = torch.linspace(t_min, t_max, num_points, device=device, dtype=work_dtype)
    target_cf = torch.exp(-0.5 * t.square())

    projected = x @ A
    projected = _sync_projected(projected, ddp_sync=ddp_sync)
    num_samples = projected.size(0)

    projected_t = projected.unsqueeze(-1) * t.view(1, 1, -1)
    ecf_real = torch.cos(projected_t).mean(dim=0)
    ecf_imag = torch.sin(projected_t).mean(dim=0)

    err = ((ecf_real - target_cf.view(1, -1)) ** 2 + ecf_imag.square()) * target_cf.view(1, -1)
    per_slice = torch.trapz(err, t, dim=-1) * num_samples
    return per_slice.mean()
