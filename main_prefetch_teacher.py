# Copyright 2023 solo-learn development team.

import os
from pathlib import Path
from typing import Iterator

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from lightning.pytorch import seed_everything
from numpy.lib.format import open_memmap
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from solo.args.pretrain import parse_cfg
from solo.data.pretrain_dataloader import (
    FullTransformPipeline,
    NCropAugmentation,
    build_transform_pipeline,
    prepare_datasets,
)
from solo.methods import METHODS
from solo.utils.misc import omegaconf_select
from solo.utils.teacher import build_teacher
from solo.utils.teacher_prefetch import (
    TeacherPrefetchWriter,
    build_teacher_prefetch_metadata,
)

_CACHE_DTYPES = {
    "float16": np.float16,
    "float32": np.float32,
}


def _format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PB"


def _num_configured_devices(devices) -> int:
    if devices is None:
        return 0
    if isinstance(devices, int):
        return int(devices)
    try:
        return len(devices)
    except TypeError:
        return 1


class ShardedSequentialSampler(Sampler[int]):
    """Evenly shards dataset indices across ranks without padding or duplication."""

    def __init__(self, dataset_size: int, rank: int, num_replicas: int):
        if dataset_size < 0:
            raise ValueError(f"dataset_size must be >= 0, got {dataset_size}.")
        if num_replicas <= 0:
            raise ValueError(f"num_replicas must be > 0, got {num_replicas}.")
        if not (0 <= rank < num_replicas):
            raise ValueError(
                f"rank must be in [0, {num_replicas}), got rank={rank}."
            )

        self.dataset_size = int(dataset_size)
        self.rank = int(rank)
        self.num_replicas = int(num_replicas)

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, self.dataset_size, self.num_replicas))

    def __len__(self) -> int:
        remaining = self.dataset_size - self.rank
        if remaining <= 0:
            return 0
        return (remaining + self.num_replicas - 1) // self.num_replicas


class TeacherPrefetchShardWriter:
    """Writes one rank's local prefetch shard so rank 0 can assemble the final cache."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        epoch: int,
        rank: int,
        num_samples: int,
        num_views: int,
        teacher_dim: int,
        cache_dtype: str,
    ):
        if cache_dtype not in _CACHE_DTYPES:
            raise ValueError(
                f"Unsupported teacher prefetch dtype '{cache_dtype}'. "
                f"Expected one of {sorted(_CACHE_DTYPES)}."
            )

        self.cache_dir = Path(cache_dir)
        self.shard_dir = self.cache_dir / ".dist_prefetch_shards"
        self.shard_dir.mkdir(parents=True, exist_ok=True)

        self.epoch = int(epoch)
        self.rank = int(rank)
        self.num_samples = int(num_samples)
        self.num_views = int(num_views)
        self.teacher_dim = int(teacher_dim)
        self.cache_dtype = str(cache_dtype)
        self.np_dtype = _CACHE_DTYPES[self.cache_dtype]
        self._cursor = 0
        self._features_memmap = None
        self._indices_memmap = None

        stem = f"epoch_{self.epoch:06d}.rank_{self.rank:06d}"
        self.features_path = self.shard_dir / f"{stem}.features.npy"
        self.indices_path = self.shard_dir / f"{stem}.indices.npy"
        self.tmp_features_path = self.shard_dir / f"{stem}.features.tmp.npy"
        self.tmp_indices_path = self.shard_dir / f"{stem}.indices.tmp.npy"

    def start_epoch(self) -> None:
        self._cursor = 0
        self._features_memmap = None
        self._indices_memmap = None

        for path in (
            self.features_path,
            self.indices_path,
            self.tmp_features_path,
            self.tmp_indices_path,
        ):
            if path.exists():
                path.unlink()

        if self.num_samples == 0:
            return

        self._features_memmap = open_memmap(
            self.tmp_features_path,
            mode="w+",
            dtype=self.np_dtype,
            shape=(self.num_samples, self.num_views, self.teacher_dim),
        )
        self._indices_memmap = open_memmap(
            self.tmp_indices_path,
            mode="w+",
            dtype=np.int64,
            shape=(self.num_samples,),
        )

    def write_batch(self, indices: torch.Tensor, features: torch.Tensor) -> None:
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

        if not torch.is_tensor(indices):
            indices = torch.as_tensor(indices, dtype=torch.int64)
        else:
            indices = indices.detach().cpu().to(dtype=torch.int64)

        batch_size = int(indices.numel())
        if self._cursor + batch_size > self.num_samples:
            raise ValueError(
                f"Shard overflow for rank {self.rank}: cursor={self._cursor}, "
                f"batch_size={batch_size}, num_samples={self.num_samples}."
            )

        if self.num_samples == 0:
            if batch_size != 0:
                raise ValueError(
                    f"Rank {self.rank} was assigned zero samples but received a non-empty batch."
                )
            return

        assert self._features_memmap is not None and self._indices_memmap is not None
        end = self._cursor + batch_size
        self._indices_memmap[self._cursor:end] = indices.numpy().astype(np.int64, copy=False)
        self._features_memmap[self._cursor:end] = (
            features.detach().cpu().numpy().astype(self.np_dtype, copy=False)
        )
        self._cursor = end

    def finish_epoch(self) -> None:
        if self._cursor != self.num_samples:
            raise ValueError(
                f"Rank {self.rank} wrote {self._cursor} samples, expected {self.num_samples}."
            )

        if self.num_samples == 0:
            np.save(
                self.tmp_indices_path,
                np.empty((0,), dtype=np.int64),
                allow_pickle=False,
            )
            np.save(
                self.tmp_features_path,
                np.empty((0, self.num_views, self.teacher_dim), dtype=self.np_dtype),
                allow_pickle=False,
            )
        else:
            assert self._features_memmap is not None and self._indices_memmap is not None
            self._indices_memmap.flush()
            self._features_memmap.flush()
            self._indices_memmap = None
            self._features_memmap = None

        os.replace(self.tmp_indices_path, self.indices_path)
        os.replace(self.tmp_features_path, self.features_path)

    def abort_epoch(self) -> None:
        self._indices_memmap = None
        self._features_memmap = None
        for path in (self.tmp_indices_path, self.tmp_features_path):
            if path.exists():
                path.unlink()


def _get_dist_context():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        return True, rank, local_rank, world_size, False

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size <= 1:
        return False, 0, 0, 1, False

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    return True, rank, local_rank, world_size, True


def _broadcast_bool(value: bool, *, device: torch.device) -> bool:
    payload = torch.tensor([1 if value else 0], device=device, dtype=torch.int32)
    dist.broadcast(payload, src=0)
    return bool(payload.item())


def _merge_distributed_epoch(
    *,
    cache_dir: Path,
    epoch: int,
    world_size: int,
    final_writer: TeacherPrefetchWriter,
    merge_chunk_size: int,
) -> None:
    started = final_writer.start_epoch(epoch)
    if not started:
        return

    merged = False
    try:
        shard_dir = cache_dir / ".dist_prefetch_shards"
        for shard_rank in range(world_size):
            stem = f"epoch_{int(epoch):06d}.rank_{int(shard_rank):06d}"
            indices_path = shard_dir / f"{stem}.indices.npy"
            features_path = shard_dir / f"{stem}.features.npy"

            if not indices_path.is_file() or not features_path.is_file():
                raise FileNotFoundError(
                    f"Missing distributed teacher-prefetch shard(s) for epoch={epoch}, "
                    f"rank={shard_rank}: indices='{indices_path}', features='{features_path}'."
                )

            shard_indices = np.load(indices_path, mmap_mode="r", allow_pickle=False)
            shard_features = np.load(features_path, mmap_mode="r", allow_pickle=False)

            if shard_indices.shape[0] != shard_features.shape[0]:
                raise ValueError(
                    f"Shard size mismatch for epoch={epoch}, rank={shard_rank}: "
                    f"indices={shard_indices.shape[0]}, features={shard_features.shape[0]}."
                )

            for start in range(0, shard_indices.shape[0], merge_chunk_size):
                end = min(start + merge_chunk_size, shard_indices.shape[0])
                batch_indices = np.asarray(shard_indices[start:end], dtype=np.int64)
                batch_features = torch.from_numpy(np.asarray(shard_features[start:end]))
                final_writer.write_batch(batch_indices, batch_features)

        final_writer.finish_epoch()
        merged = True
    except Exception:
        final_writer.abort_epoch()
        raise
    finally:
        if merged:
            shard_dir = cache_dir / ".dist_prefetch_shards"
            for shard_rank in range(world_size):
                stem = f"epoch_{int(epoch):06d}.rank_{int(shard_rank):06d}"
                for suffix in ("indices.npy", "features.npy"):
                    shard_path = shard_dir / f"{stem}.{suffix}"
                    if shard_path.exists():
                        shard_path.unlink()


@hydra.main(version_base="1.2")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    cfg = parse_cfg(cfg)

    assert cfg.method in METHODS, f"Choose from {METHODS.keys()}"
    if cfg.method != "split_teacher_sigjepa":
        raise ValueError(
            "main_prefetch_teacher.py currently supports method='split_teacher_sigjepa' only."
        )

    cfg = METHODS[cfg.method].add_and_assert_specific_cfg(cfg)
    seed_everything(cfg.seed)

    if cfg.data.format == "dali":
        raise ValueError(
            "Teacher prefetch is only implemented for the standard PyTorch dataloader path; "
            "set data.format=image_folder or h5 instead of dali."
        )

    teacher_backend = omegaconf_select(cfg, "method_kwargs.teacher_backend", "hf_dinov2")
    if teacher_backend in {None, "none"}:
        raise ValueError(
            "A real frozen teacher is required to precompute teacher representations; "
            "teacher_backend='none' cannot be prefetched."
        )

    prefetch_cache_dir = Path(cfg.method_kwargs.teacher_prefetch.cache_dir)
    num_prefetch_epochs = int(cfg.method_kwargs.teacher_prefetch.num_epochs)
    prefetch_batch_size = int(cfg.method_kwargs.teacher_prefetch.batch_size)
    prefetch_num_workers = int(cfg.method_kwargs.teacher_prefetch.num_workers)
    prefetch_seed = int(cfg.method_kwargs.teacher_prefetch.base_seed)
    prefetch_overwrite = bool(cfg.method_kwargs.teacher_prefetch.overwrite)
    cache_dtype = str(cfg.method_kwargs.teacher_prefetch.dtype)

    pipelines = []
    for aug_cfg in cfg.augmentations:
        pipelines.append(
            NCropAugmentation(build_transform_pipeline(cfg.data.dataset, aug_cfg), aug_cfg.num_crops)
        )
    transform = FullTransformPipeline(pipelines)

    if cfg.debug_augmentations:
        print("Transforms:")
        print(transform)

    train_dataset = prepare_datasets(
        cfg.data.dataset,
        transform,
        train_data_path=cfg.data.train_path,
        data_format=cfg.data.format,
        no_labels=cfg.data.no_labels,
        data_fraction=cfg.data.fraction,
        preload=cfg.data.preload,
        deterministic_augmentations=True,
        deterministic_augmentations_seed=prefetch_seed,
    )

    teacher = build_teacher(cfg)
    teacher.requires_grad_(False)
    teacher.eval()
    teacher_dim = int(getattr(teacher, "output_dim", cfg.method_kwargs.teacher_output_dim))

    metadata = build_teacher_prefetch_metadata(
        cfg,
        num_samples=len(train_dataset),
        num_views=int(cfg.data.num_large_crops),
        teacher_dim=teacher_dim,
        cache_dtype=cache_dtype,
        normalized=True,
    )

    is_distributed, rank, local_rank, world_size, owns_process_group = _get_dist_context()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        sync_device = device
    else:
        device = torch.device("cpu")
        sync_device = device

    final_writer = None
    if rank == 0:
        final_writer = TeacherPrefetchWriter(
            prefetch_cache_dir,
            metadata,
            overwrite=prefetch_overwrite,
        )
    if is_distributed:
        dist.barrier()

    dtype_bytes = {"float16": 2, "float32": 4}[cache_dtype]
    estimated_bytes = (
        len(train_dataset)
        * int(cfg.data.num_large_crops)
        * teacher_dim
        * dtype_bytes
        * num_prefetch_epochs
    )
    if rank == 0:
        print(
            "Teacher prefetch target:",
            f"backend={teacher_backend}",
            f"model_id={omegaconf_select(cfg, 'method_kwargs.teacher_model_id', None)}",
            f"cache_dir={prefetch_cache_dir}",
            f"num_epochs={num_prefetch_epochs}",
            f"num_samples={len(train_dataset)}",
            f"num_views={cfg.data.num_large_crops}",
            f"teacher_dim={teacher_dim}",
            f"dtype={cache_dtype}",
            f"estimated_disk={_format_bytes(int(estimated_bytes))}",
        )
    teacher = teacher.to(device)
    if rank == 0:
        if is_distributed:
            print(
                "Distributed teacher prefetch enabled:",
                f"world_size={world_size}",
                f"local_batch_size={prefetch_batch_size}",
            )
        elif _num_configured_devices(getattr(cfg, "devices", None)) > 1:
            print(
                "Multiple devices are configured, but teacher prefetch is running in a single "
                "process. Use torchrun to enable multi-GPU prefetch."
            )

        if device.type == "cuda":
            print(f"Using CUDA device: {torch.cuda.get_device_name(device)}")
        else:
            print("Using CPU for teacher prefetch.")

    autocast_enabled = device.type == "cuda"
    num_large_crops = int(cfg.data.num_large_crops)
    merge_chunk_size = max(1, prefetch_batch_size)

    sampler = None
    local_num_samples = len(train_dataset)
    if is_distributed:
        sampler = ShardedSequentialSampler(
            len(train_dataset),
            rank=rank,
            num_replicas=world_size,
        )
        local_num_samples = len(sampler)

    train_loader = DataLoader(
        train_dataset,
        batch_size=prefetch_batch_size,
        num_workers=prefetch_num_workers,
        shuffle=False,
        sampler=sampler,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )

    for epoch in range(num_prefetch_epochs):
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)

        should_run_epoch = True
        if rank == 0:
            assert final_writer is not None
            epoch_path = final_writer.epoch_path(epoch)
            should_run_epoch = prefetch_overwrite or not epoch_path.is_file()
            if not should_run_epoch:
                print(f"Skipping existing teacher prefetch epoch {epoch} at {epoch_path}")
        if is_distributed:
            should_run_epoch = _broadcast_bool(should_run_epoch, device=sync_device)

        if not should_run_epoch:
            continue

        if rank == 0:
            print(f"Prefetching teacher features for epoch {epoch + 1}/{num_prefetch_epochs}...")

        if not is_distributed:
            assert final_writer is not None
            started = final_writer.start_epoch(epoch)
            if not started:
                continue
            try:
                progress = tqdm(
                    train_loader,
                    desc=f"teacher-prefetch epoch {epoch}",
                    leave=False,
                )
                for img_indexes, X, _targets in progress:
                    X = [X] if isinstance(X, torch.Tensor) else X
                    large_views = X[:num_large_crops]

                    batch_features = []
                    with torch.no_grad():
                        for view in large_views:
                            view = view.to(device, non_blocking=True)
                            with torch.autocast(
                                device_type=device.type,
                                dtype=torch.float16,
                                enabled=autocast_enabled,
                            ):
                                teacher_feats = teacher(view)
                            teacher_feats = F.normalize(teacher_feats.float(), dim=1).cpu()
                            batch_features.append(teacher_feats)
                    batch_features = torch.stack(batch_features, dim=1)
                    final_writer.write_batch(img_indexes, batch_features)
                final_writer.finish_epoch()
            except Exception:
                final_writer.abort_epoch()
                raise
            continue

        shard_writer = TeacherPrefetchShardWriter(
            prefetch_cache_dir,
            epoch=epoch,
            rank=rank,
            num_samples=local_num_samples,
            num_views=num_large_crops,
            teacher_dim=teacher_dim,
            cache_dtype=cache_dtype,
        )
        shard_writer.start_epoch()

        try:
            progress = tqdm(
                train_loader,
                desc=f"teacher-prefetch epoch {epoch}",
                leave=False,
                disable=rank != 0,
            )
            for img_indexes, X, _targets in progress:
                X = [X] if isinstance(X, torch.Tensor) else X
                large_views = X[:num_large_crops]

                batch_features = []
                with torch.no_grad():
                    for view in large_views:
                        view = view.to(device, non_blocking=True)
                        with torch.autocast(
                            device_type=device.type,
                            dtype=torch.float16,
                            enabled=autocast_enabled,
                        ):
                            teacher_feats = teacher(view)
                        teacher_feats = F.normalize(teacher_feats.float(), dim=1).cpu()
                        batch_features.append(teacher_feats)
                batch_features = torch.stack(batch_features, dim=1)
                shard_writer.write_batch(img_indexes, batch_features)
            shard_writer.finish_epoch()
        except Exception:
            shard_writer.abort_epoch()
            raise

        dist.barrier()
        if rank == 0:
            assert final_writer is not None
            _merge_distributed_epoch(
                cache_dir=prefetch_cache_dir,
                epoch=epoch,
                world_size=world_size,
                final_writer=final_writer,
                merge_chunk_size=merge_chunk_size,
            )
        dist.barrier()

    if rank == 0:
        print(f"Teacher prefetch cache is ready at: {prefetch_cache_dir}")

    if is_distributed and owns_process_group:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
