# Copyright 2023 solo-learn development team.

import os
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from lightning.pytorch import seed_everything
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from solo.args.pretrain import parse_cfg
from solo.data.pretrain_dataloader import (
    FullTransformPipeline,
    NCropAugmentation,
    build_transform_pipeline,
    prepare_dataloader,
    prepare_datasets,
)
from solo.methods import METHODS
from solo.utils.misc import omegaconf_select
from solo.utils.teacher import build_teacher
from solo.utils.teacher_prefetch import (
    TeacherPrefetchWriter,
    build_teacher_prefetch_metadata,
)


def _format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PB"


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

    teacher_backend = omegaconf_select(cfg, "method_kwargs.teacher_backend", "hf_ijepa")
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
    train_loader = prepare_dataloader(
        train_dataset,
        batch_size=prefetch_batch_size,
        num_workers=prefetch_num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
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
    writer = TeacherPrefetchWriter(
        prefetch_cache_dir,
        metadata,
        overwrite=prefetch_overwrite,
    )

    dtype_bytes = {"float16": 2, "float32": 4}[cache_dtype]
    estimated_bytes = (
        len(train_dataset)
        * int(cfg.data.num_large_crops)
        * teacher_dim
        * dtype_bytes
        * num_prefetch_epochs
    )
    print(
        "Teacher prefetch target:",
        f"cache_dir={prefetch_cache_dir}",
        f"num_epochs={num_prefetch_epochs}",
        f"num_samples={len(train_dataset)}",
        f"num_views={cfg.data.num_large_crops}",
        f"teacher_dim={teacher_dim}",
        f"dtype={cache_dtype}",
        f"estimated_disk={_format_bytes(int(estimated_bytes))}",
    )

    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    else:
        device = torch.device("cpu")
    teacher = teacher.to(device)
    if device.type == "cuda":
        print(f"Using CUDA device: {torch.cuda.get_device_name(device)}")
    else:
        print("Using CPU for teacher prefetch.")

    autocast_enabled = device.type == "cuda"
    num_large_crops = int(cfg.data.num_large_crops)

    for epoch in range(num_prefetch_epochs):
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)

        started = writer.start_epoch(epoch)
        if not started:
            print(f"Skipping existing teacher prefetch epoch {epoch} at {writer.epoch_path(epoch)}")
            continue

        print(f"Prefetching teacher features for epoch {epoch + 1}/{num_prefetch_epochs}...")
        try:
            progress = tqdm(train_loader, desc=f"teacher-prefetch epoch {epoch}", leave=False)
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
                writer.write_batch(img_indexes, batch_features)
            writer.finish_epoch()
        except Exception:
            writer.abort_epoch()
            raise

    print(f"Teacher prefetch cache is ready at: {prefetch_cache_dir}")


if __name__ == "__main__":
    main()
