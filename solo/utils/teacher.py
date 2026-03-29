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

from __future__ import annotations

from typing import Dict, Iterator, Optional, Tuple

import torch
import torch.nn as nn

try:
    from transformers import AutoModel
except Exception as exc:  # pragma: no cover - dependency/import error path
    AutoModel = None
    _TRANSFORMERS_IMPORT_ERROR = exc
else:
    _TRANSFORMERS_IMPORT_ERROR = None

from solo.utils.misc import omegaconf_select


class FrozenIJepaTeacher(nn.Module):
    """Frozen Hugging Face I-JEPA teacher wrapper.

    The wrapped HF model is intentionally kept out of the module state_dict so
    training checkpoints do not serialize the entire frozen teacher.
    """

    _VALID_POOLING = {"mean", "cls", "patch_mean"}

    def __init__(
        self,
        model_id: str = "facebook/ijepa_vith14_1k",
        pooling: str = "mean",
        chunk_size: int = 16,
        cast_output_to_float32: bool = True,
        local_dir: Optional[str] = None,
        output_dim: Optional[int] = 1280,
        eager_load: bool = False,
    ):
        super().__init__()

        if pooling not in self._VALID_POOLING:
            raise ValueError(
                f"Unknown pooling '{pooling}'. Expected one of {sorted(self._VALID_POOLING)}."
            )

        self.model_id = model_id
        self.local_dir = local_dir
        self.pooling = pooling
        self.chunk_size = int(chunk_size) if chunk_size is not None else 0
        self.cast_output_to_float32 = cast_output_to_float32
        self.output_dim = int(output_dim) if output_dim is not None else None
        self.register_buffer("_device_tracker", torch.empty(0), persistent=False)

        object.__setattr__(self, "_model", None)
        if eager_load:
            self._ensure_model_loaded()

    @property
    def model(self) -> nn.Module:
        return self._ensure_model_loaded()

    def _load_source(self) -> Tuple[str, Dict[str, object]]:
        source = self.local_dir or self.model_id
        kwargs: Dict[str, object] = {"local_files_only": self.local_dir is not None}
        return source, kwargs

    def _ensure_model_loaded(self) -> nn.Module:
        model = getattr(self, "_model", None)
        if model is not None:
            return model

        if AutoModel is None:
            raise ImportError(
                "transformers is required to use FrozenIJepaTeacher."
            ) from _TRANSFORMERS_IMPORT_ERROR

        source, kwargs = self._load_source()
        model = AutoModel.from_pretrained(source, **kwargs)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

        hidden_size = getattr(model.config, "hidden_size", None)
        if hidden_size is None:
            if self.output_dim is None:
                raise ValueError(
                    "Could not infer teacher hidden size from model.config.hidden_size and no "
                    "fallback teacher_output_dim was provided."
                )
        else:
            hidden_size = int(hidden_size)
            if self.output_dim is not None and hidden_size != self.output_dim:
                raise ValueError(
                    f"Configured teacher_output_dim={self.output_dim} does not match "
                    f"loaded hidden_size={hidden_size}."
                )
            self.output_dim = hidden_size

        model.to(device=self._device_tracker.device)
        object.__setattr__(self, "_model", model)
        return model

    def _pool_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return hidden_states.mean(dim=1)
        if self.pooling == "cls":
            return hidden_states[:, 0]
        if self.pooling == "patch_mean":
            if hidden_states.size(1) <= 1:
                raise ValueError(
                    "patch_mean pooling requires a sequence with at least one patch token."
                )
            return hidden_states[:, 1:].mean(dim=1)
        raise ValueError(f"Unknown pooling: {self.pooling}")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        model = self.model
        model.eval()

        if x.ndim != 4:
            raise ValueError(
                f"FrozenIJepaTeacher expects image tensors [B, C, H, W], got {tuple(x.shape)}."
            )

        chunk_size = self.chunk_size if self.chunk_size and self.chunk_size > 0 else x.size(0)
        feats = []
        for x_chunk in x.split(chunk_size, dim=0):
            outputs = model(pixel_values=x_chunk, return_dict=True)
            if not hasattr(outputs, "last_hidden_state"):
                raise ValueError(
                    "Teacher model output does not expose last_hidden_state, which is required "
                    "for v0 mean/cls/patch_mean pooling."
                )
            pooled = self._pool_hidden_states(outputs.last_hidden_state)
            if self.cast_output_to_float32:
                pooled = pooled.float()
            feats.append(pooled)
        return torch.cat(feats, dim=0)

    def train(self, mode: bool = True):
        super().train(False)
        model = getattr(self, "_model", None)
        if model is not None:
            model.train(False)
        return self

    def requires_grad_(self, requires_grad: bool = False):
        model = getattr(self, "_model", None)
        if model is not None:
            for param in model.parameters():
                param.requires_grad = requires_grad
        return self

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        model = getattr(self, "_model", None)
        if model is None:
            return iter(())
        return model.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):
        model = getattr(self, "_model", None)
        if model is None:
            return iter(())
        return model.named_parameters(
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )

    def _apply(self, fn):
        super()._apply(fn)
        model = getattr(self, "_model", None)
        if model is not None:
            model._apply(fn)
        return self


class IdentityTeacher(nn.Module):
    """Fallback no-op teacher placeholder used when the teacher branch is disabled."""

    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = int(output_dim)
        self.register_buffer("_device_tracker", torch.empty(0), persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.size(0), self.output_dim, device=x.device, dtype=torch.float32)

    def train(self, mode: bool = True):
        return super().train(False)


def build_teacher(cfg) -> nn.Module:
    """Builds the v0 teacher defined in IDEA.md."""

    method_kwargs = cfg.method_kwargs
    teacher_backend = omegaconf_select(method_kwargs, "teacher_backend", "hf_ijepa")
    teacher_output_dim = int(omegaconf_select(method_kwargs, "teacher_output_dim", 1280))

    if teacher_backend in {None, "none"}:
        return IdentityTeacher(output_dim=teacher_output_dim)

    if teacher_backend != "hf_ijepa":
        raise ValueError(
            f"Unsupported teacher_backend '{teacher_backend}'. Only 'hf_ijepa' is supported in v0."
        )

    return FrozenIJepaTeacher(
        model_id=omegaconf_select(method_kwargs, "teacher_model_id", "facebook/ijepa_vith14_1k"),
        local_dir=omegaconf_select(method_kwargs, "teacher_local_dir", None),
        pooling=omegaconf_select(method_kwargs, "teacher_pooling", "mean"),
        chunk_size=int(omegaconf_select(method_kwargs, "teacher_chunk_size", 16)),
        cast_output_to_float32=True,
        output_dim=teacher_output_dim,
        eager_load=bool(omegaconf_select(method_kwargs, "teacher_eager_load", False)),
    )
