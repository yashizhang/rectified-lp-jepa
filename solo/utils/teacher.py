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

from dataclasses import dataclass
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


@dataclass(frozen=True)
class TeacherBackendDefaults:
    model_id: Optional[str]
    output_dim: Optional[int]
    pooling: str


_TEACHER_BACKEND_DEFAULTS: Dict[str, TeacherBackendDefaults] = {
    "hf_ijepa": TeacherBackendDefaults(
        model_id="facebook/ijepa_vith14_1k",
        output_dim=1280,
        pooling="mean",
    ),
    "hf_dinov2": TeacherBackendDefaults(
        model_id="facebook/dinov2-large",
        output_dim=1024,
        pooling="cls",
    ),
    "hf_dinov2_with_registers": TeacherBackendDefaults(
        model_id="facebook/dinov2-with-registers-large",
        output_dim=1024,
        pooling="cls",
    ),
    "hf_auto": TeacherBackendDefaults(
        model_id="facebook/dinov2-large",
        output_dim=1024,
        pooling="cls",
    ),
}


def infer_teacher_output_dim_from_model_id(model_id: Optional[str]) -> Optional[int]:
    """Best-effort hidden-size inference for common teacher checkpoints.

    This keeps the config ergonomic when users swap between common I-JEPA / DINOv2
    checkpoints without manually updating `teacher_output_dim` each time.
    """

    if not model_id:
        return None

    model_id = str(model_id).lower()
    size_patterns = (
        (("ijepa_vith14", "vit-h", "vit_h", "huge", "vith14"), 1280),
        (("dinov2-with-registers-small", "dinov2-small", "vits14", "vit-s", "vit_s", "small"), 384),
        (("dinov2-with-registers-base", "dinov2-base", "vitb14", "vit-b", "vit_b", "base"), 768),
        (("dinov2-with-registers-large", "dinov2-large", "vitl14", "vit-l", "vit_l", "large"), 1024),
        (("dinov2-with-registers-giant", "dinov2-giant", "vitg14", "vit-g", "vit_g", "giant"), 1536),
    )
    for patterns, dim in size_patterns:
        if any(pattern in model_id for pattern in patterns):
            return dim
    return None


def get_teacher_backend_defaults(
    teacher_backend: Optional[str],
    teacher_model_id: Optional[str] = None,
) -> Dict[str, Optional[object]]:
    """Resolves backend-specific defaults for the frozen teacher path."""

    if teacher_backend in {None, "none"}:
        return {
            "model_id": teacher_model_id,
            "output_dim": infer_teacher_output_dim_from_model_id(teacher_model_id) or 768,
            "pooling": "cls",
        }

    teacher_backend = str(teacher_backend)
    if teacher_backend not in _TEACHER_BACKEND_DEFAULTS:
        raise ValueError(
            f"Unsupported teacher_backend '{teacher_backend}'. Expected one of "
            f"{sorted(_TEACHER_BACKEND_DEFAULTS)} or 'none'."
        )

    preset = _TEACHER_BACKEND_DEFAULTS[teacher_backend]
    resolved_model_id = teacher_model_id or preset.model_id
    inferred_output_dim = infer_teacher_output_dim_from_model_id(resolved_model_id)

    return {
        "model_id": resolved_model_id,
        "output_dim": inferred_output_dim or preset.output_dim,
        "pooling": preset.pooling,
    }


class FrozenHFAutoTeacher(nn.Module):
    """Frozen Hugging Face AutoModel teacher wrapper for I-JEPA / DINOv2 models.

    The wrapped HF model is intentionally kept out of the module state_dict so
    training checkpoints do not serialize the entire frozen teacher.
    """

    _VALID_POOLING = {"mean", "cls", "patch_mean", "pooler"}

    def __init__(
        self,
        model_id: str,
        pooling: str = "cls",
        chunk_size: int = 16,
        cast_output_to_float32: bool = True,
        local_dir: Optional[str] = None,
        output_dim: Optional[int] = None,
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
                "transformers is required to use FrozenHFAutoTeacher."
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

    def _num_prefix_tokens(self, model: Optional[nn.Module] = None) -> int:
        model = self.model if model is None else model
        num_register_tokens = int(getattr(model.config, "num_register_tokens", 0) or 0)
        return 1 + num_register_tokens

    def _pool_hidden_states(
        self,
        hidden_states: torch.Tensor,
        *,
        outputs=None,
        model: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        model = self.model if model is None else model

        if self.pooling == "mean":
            return hidden_states.mean(dim=1)
        if self.pooling == "cls":
            return hidden_states[:, 0]
        if self.pooling == "patch_mean":
            prefix_tokens = self._num_prefix_tokens(model)
            if hidden_states.size(1) <= prefix_tokens:
                raise ValueError(
                    "patch_mean pooling requires a sequence with at least one patch token after "
                    f"excluding {prefix_tokens} prefix / register tokens."
                )
            return hidden_states[:, prefix_tokens:].mean(dim=1)
        if self.pooling == "pooler":
            pooled_output = None if outputs is None else getattr(outputs, "pooler_output", None)
            if pooled_output is None:
                raise ValueError(
                    "pooler pooling requires the teacher model output to expose pooler_output."
                )
            return pooled_output
        raise ValueError(f"Unknown pooling: {self.pooling}")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        model = self.model
        model.eval()

        if x.ndim != 4:
            raise ValueError(
                f"FrozenHFAutoTeacher expects image tensors [B, C, H, W], got {tuple(x.shape)}."
            )

        chunk_size = self.chunk_size if self.chunk_size and self.chunk_size > 0 else x.size(0)
        feats = []
        for x_chunk in x.split(chunk_size, dim=0):
            outputs = model(pixel_values=x_chunk, return_dict=True)
            if not hasattr(outputs, "last_hidden_state"):
                raise ValueError(
                    "Teacher model output does not expose last_hidden_state, which is required "
                    "for cls/mean/patch_mean pooling."
                )
            pooled = self._pool_hidden_states(
                outputs.last_hidden_state,
                outputs=outputs,
                model=model,
            )
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


# Backwards-compatible aliases.
FrozenIJepaTeacher = FrozenHFAutoTeacher
FrozenDinoV2Teacher = FrozenHFAutoTeacher


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
    """Builds the configured frozen teacher for split-teacher SIGReg JEPA."""

    method_kwargs = cfg.method_kwargs
    teacher_backend = omegaconf_select(method_kwargs, "teacher_backend", "hf_dinov2")
    teacher_model_id = omegaconf_select(method_kwargs, "teacher_model_id", None)
    defaults = get_teacher_backend_defaults(teacher_backend, teacher_model_id)

    teacher_output_dim = int(
        omegaconf_select(method_kwargs, "teacher_output_dim", defaults["output_dim"])
    )
    if teacher_backend in {None, "none"}:
        return IdentityTeacher(output_dim=teacher_output_dim)

    return FrozenHFAutoTeacher(
        model_id=omegaconf_select(method_kwargs, "teacher_model_id", defaults["model_id"]),
        local_dir=omegaconf_select(method_kwargs, "teacher_local_dir", None),
        pooling=omegaconf_select(method_kwargs, "teacher_pooling", defaults["pooling"]),
        chunk_size=int(omegaconf_select(method_kwargs, "teacher_chunk_size", 16)),
        cast_output_to_float32=True,
        output_dim=teacher_output_dim,
        eager_load=bool(omegaconf_select(method_kwargs, "teacher_eager_load", False)),
    )
