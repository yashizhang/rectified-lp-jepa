from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import torch
from numpy.lib.format import open_memmap
from omegaconf import OmegaConf

from solo.utils.misc import omegaconf_select


_VALID_CACHE_DTYPES = {
    "float16": np.float16,
    "float32": np.float32,
}


def resolve_teacher_prefetch_epoch(
    train_epoch: int,
    num_prefetch_epochs: int,
    epoch_mode: str = "wrap",
) -> int:
    if num_prefetch_epochs <= 0:
        raise ValueError(f"num_prefetch_epochs must be > 0, got {num_prefetch_epochs}.")

    train_epoch = int(train_epoch)
    epoch_mode = str(epoch_mode)
    if epoch_mode == "wrap":
        return train_epoch % num_prefetch_epochs
    if epoch_mode == "strict":
        if train_epoch >= num_prefetch_epochs:
            raise ValueError(
                "Teacher prefetch cache does not contain enough epochs for strict mode: "
                f"requested train_epoch={train_epoch}, available_prefetch_epochs={num_prefetch_epochs}."
            )
        return train_epoch
    raise ValueError(
        f"Unsupported teacher_prefetch.epoch_mode '{epoch_mode}'. Expected 'wrap' or 'strict'."
    )


def build_teacher_prefetch_fingerprint(cfg) -> str:
    payload = {
        "dataset": str(cfg.data.dataset),
        "train_path": str(cfg.data.train_path),
        "format": str(cfg.data.format),
        "no_labels": bool(omegaconf_select(cfg, "data.no_labels", False)),
        "fraction": float(omegaconf_select(cfg, "data.fraction", -1.0)),
        "preload": bool(omegaconf_select(cfg, "data.preload", False)),
        "num_large_crops": int(cfg.data.num_large_crops),
        "num_small_crops": int(cfg.data.num_small_crops),
        "augmentations": OmegaConf.to_container(cfg.augmentations, resolve=True),
        "teacher_backend": omegaconf_select(cfg, "method_kwargs.teacher_backend", None),
        "teacher_model_id": omegaconf_select(cfg, "method_kwargs.teacher_model_id", None),
        "teacher_local_dir": omegaconf_select(cfg, "method_kwargs.teacher_local_dir", None),
        "teacher_pooling": omegaconf_select(cfg, "method_kwargs.teacher_pooling", None),
        "teacher_output_dim": int(omegaconf_select(cfg, "method_kwargs.teacher_output_dim", 0)),
        "teacher_prefetch_seed": int(
            omegaconf_select(cfg, "method_kwargs.teacher_prefetch.base_seed", cfg.seed)
        ),
    }
    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload_json.encode("utf-8")).hexdigest()


def build_teacher_prefetch_metadata(
    cfg,
    *,
    num_samples: int,
    num_views: int,
    teacher_dim: int,
    cache_dtype: str,
    normalized: bool = True,
) -> Dict[str, Any]:
    cache_dtype = str(cache_dtype)
    if cache_dtype not in _VALID_CACHE_DTYPES:
        raise ValueError(
            f"Unsupported teacher prefetch dtype '{cache_dtype}'. "
            f"Expected one of {sorted(_VALID_CACHE_DTYPES)}."
        )

    num_prefetch_epochs = int(
        omegaconf_select(cfg, "method_kwargs.teacher_prefetch.num_epochs", cfg.max_epochs)
    )
    epoch_mode = str(omegaconf_select(cfg, "method_kwargs.teacher_prefetch.epoch_mode", "wrap"))
    base_seed = int(omegaconf_select(cfg, "method_kwargs.teacher_prefetch.base_seed", cfg.seed))

    return {
        "version": 1,
        "fingerprint": build_teacher_prefetch_fingerprint(cfg),
        "num_epochs": num_prefetch_epochs,
        "num_samples": int(num_samples),
        "num_views": int(num_views),
        "teacher_dim": int(teacher_dim),
        "dtype": cache_dtype,
        "normalized": bool(normalized),
        "teacher": {
            "backend": omegaconf_select(cfg, "method_kwargs.teacher_backend", None),
            "model_id": omegaconf_select(cfg, "method_kwargs.teacher_model_id", None),
            "local_dir": omegaconf_select(cfg, "method_kwargs.teacher_local_dir", None),
            "pooling": omegaconf_select(cfg, "method_kwargs.teacher_pooling", None),
            "output_dim": int(omegaconf_select(cfg, "method_kwargs.teacher_output_dim", teacher_dim)),
        },
        "data": {
            "dataset": str(cfg.data.dataset),
            "train_path": str(cfg.data.train_path),
            "format": str(cfg.data.format),
            "no_labels": bool(omegaconf_select(cfg, "data.no_labels", False)),
            "fraction": float(omegaconf_select(cfg, "data.fraction", -1.0)),
            "preload": bool(omegaconf_select(cfg, "data.preload", False)),
            "num_large_crops": int(cfg.data.num_large_crops),
            "num_small_crops": int(cfg.data.num_small_crops),
        },
        "augmentations": OmegaConf.to_container(cfg.augmentations, resolve=True),
        "prefetch": {
            "base_seed": base_seed,
            "epoch_mode": epoch_mode,
        },
    }


class TeacherPrefetchWriter:
    def __init__(
        self,
        cache_dir: Union[str, Path],
        metadata: Dict[str, Any],
        *,
        overwrite: bool = False,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.cache_dir / "meta.json"
        self.metadata = metadata
        self.overwrite = bool(overwrite)

        self.num_epochs = int(metadata["num_epochs"])
        self.num_samples = int(metadata["num_samples"])
        self.num_views = int(metadata["num_views"])
        self.teacher_dim = int(metadata["teacher_dim"])
        self.cache_dtype = str(metadata["dtype"])
        if self.cache_dtype not in _VALID_CACHE_DTYPES:
            raise ValueError(
                f"Unsupported teacher prefetch dtype '{self.cache_dtype}'. "
                f"Expected one of {sorted(_VALID_CACHE_DTYPES)}."
            )
        self.np_dtype = _VALID_CACHE_DTYPES[self.cache_dtype]
        self.shape = (self.num_samples, self.num_views, self.teacher_dim)

        self._current_epoch: Optional[int] = None
        self._current_memmap = None
        self._current_final_path: Optional[Path] = None
        self._current_tmp_path: Optional[Path] = None

        self._validate_or_write_metadata()

    def _validate_or_write_metadata(self) -> None:
        if self.meta_path.exists():
            with open(self.meta_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if existing != self.metadata:
                if not self.overwrite:
                    raise ValueError(
                        "Teacher prefetch metadata mismatch for cache directory "
                        f"'{self.cache_dir}'. Either delete the directory, choose a new cache_dir, "
                        "or use overwrite=True with the new configuration."
                    )
                for stale_file in self.cache_dir.glob("epoch_*.npy"):
                    stale_file.unlink()
                for stale_file in self.cache_dir.glob("epoch_*.tmp.npy"):
                    stale_file.unlink()
            else:
                return

        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, sort_keys=True)

    def epoch_path(self, epoch: int) -> Path:
        return self.cache_dir / f"epoch_{int(epoch):06d}.npy"

    def tmp_epoch_path(self, epoch: int) -> Path:
        return self.cache_dir / f"epoch_{int(epoch):06d}.tmp.npy"

    def epoch_exists(self, epoch: int) -> bool:
        return self.epoch_path(epoch).is_file()

    def start_epoch(self, epoch: int) -> bool:
        epoch = int(epoch)
        if not (0 <= epoch < self.num_epochs):
            raise ValueError(
                f"Epoch {epoch} is out of range for num_prefetch_epochs={self.num_epochs}."
            )
        if self._current_memmap is not None:
            raise RuntimeError("A teacher prefetch epoch is already open. Finish it before reopening.")

        final_path = self.epoch_path(epoch)
        tmp_path = self.tmp_epoch_path(epoch)

        if final_path.exists():
            if not self.overwrite:
                return False
            final_path.unlink()
        if tmp_path.exists():
            tmp_path.unlink()

        self._current_memmap = open_memmap(
            tmp_path,
            mode="w+",
            dtype=self.np_dtype,
            shape=self.shape,
        )
        self._current_epoch = epoch
        self._current_final_path = final_path
        self._current_tmp_path = tmp_path
        return True

    def write_batch(self, indices: Union[np.ndarray, Sequence[int], torch.Tensor], features: torch.Tensor) -> None:
        if self._current_memmap is None:
            raise RuntimeError("Call start_epoch(...) before write_batch(...).")

        if not torch.is_tensor(features):
            raise TypeError("features must be a torch.Tensor.")
        if features.ndim != 3:
            raise ValueError(
                f"Expected features with shape [B, V, D], got {tuple(features.shape)}."
            )
        if features.size(1) != self.num_views:
            raise ValueError(
                f"Expected num_views={self.num_views}, got features.size(1)={features.size(1)}."
            )
        if features.size(2) != self.teacher_dim:
            raise ValueError(
                f"Expected teacher_dim={self.teacher_dim}, got features.size(2)={features.size(2)}."
            )

        if torch.is_tensor(indices):
            index_array = indices.detach().cpu().numpy().astype(np.int64, copy=False)
        else:
            index_array = np.asarray(indices, dtype=np.int64)
        feature_array = features.detach().cpu().numpy().astype(self.np_dtype, copy=False)
        self._current_memmap[index_array] = feature_array

    def finish_epoch(self) -> None:
        if self._current_memmap is None:
            return
        self._current_memmap.flush()
        self._current_memmap = None
        assert self._current_tmp_path is not None and self._current_final_path is not None
        os.replace(self._current_tmp_path, self._current_final_path)
        self._current_epoch = None
        self._current_tmp_path = None
        self._current_final_path = None

    def abort_epoch(self) -> None:
        if self._current_memmap is not None:
            self._current_memmap = None
        if self._current_tmp_path is not None and self._current_tmp_path.exists():
            self._current_tmp_path.unlink()
        self._current_epoch = None
        self._current_tmp_path = None
        self._current_final_path = None


class TeacherPrefetchReader:
    def __init__(
        self,
        cache_dir: Union[str, Path],
        *,
        expected_fingerprint: Optional[str] = None,
        expected_num_views: Optional[int] = None,
        expected_teacher_dim: Optional[int] = None,
        mmap_mode: str = "r",
    ):
        self.cache_dir = Path(cache_dir)
        self.meta_path = self.cache_dir / "meta.json"
        if not self.meta_path.is_file():
            raise FileNotFoundError(
                f"Teacher prefetch metadata file not found at '{self.meta_path}'."
            )

        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.num_epochs = int(self.metadata["num_epochs"])
        self.num_samples = int(self.metadata["num_samples"])
        self.num_views = int(self.metadata["num_views"])
        self.teacher_dim = int(self.metadata["teacher_dim"])
        self.cache_dtype = str(self.metadata["dtype"])
        self.normalized = bool(self.metadata.get("normalized", True))
        self.mmap_mode = mmap_mode

        if expected_fingerprint is not None and self.metadata.get("fingerprint") != expected_fingerprint:
            raise ValueError(
                "Teacher prefetch cache fingerprint mismatch. The cache was produced with a different "
                "dataset/augmentation/teacher configuration than the current training run."
            )
        if expected_num_views is not None and self.num_views != int(expected_num_views):
            raise ValueError(
                f"Teacher prefetch cache expects num_views={self.num_views}, "
                f"but training requested num_views={expected_num_views}."
            )
        if expected_teacher_dim is not None and self.teacher_dim != int(expected_teacher_dim):
            raise ValueError(
                f"Teacher prefetch cache expects teacher_dim={self.teacher_dim}, "
                f"but training requested teacher_dim={expected_teacher_dim}."
            )

        self._current_epoch: Optional[int] = None
        self._current_data = None

    def epoch_path(self, epoch: int) -> Path:
        return self.cache_dir / f"epoch_{int(epoch):06d}.npy"

    def set_epoch(self, epoch: int) -> None:
        epoch = int(epoch)
        if self._current_epoch == epoch and self._current_data is not None:
            return

        path = self.epoch_path(epoch)
        if not path.is_file():
            raise FileNotFoundError(
                f"Teacher prefetch epoch file not found at '{path}'."
            )

        self._current_data = np.load(path, mmap_mode=self.mmap_mode, allow_pickle=False)
        self._current_epoch = epoch

    def fetch(
        self,
        indices: Union[np.ndarray, Sequence[int], torch.Tensor],
        *,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if self._current_data is None:
            raise RuntimeError("Call set_epoch(...) before fetch(...).")

        if torch.is_tensor(indices):
            index_array = indices.detach().cpu().numpy().astype(np.int64, copy=False)
        else:
            index_array = np.asarray(indices, dtype=np.int64)

        batch = np.asarray(self._current_data[index_array], dtype=np.float32)
        tensor = torch.from_numpy(batch)
        if device is not None:
            if str(device).startswith("cuda"):
                tensor = tensor.pin_memory()
            tensor = tensor.to(device=device, dtype=dtype, non_blocking=True)
        else:
            tensor = tensor.to(dtype=dtype)
        return tensor
