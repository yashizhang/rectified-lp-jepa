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

import os
import random
from contextlib import contextmanager
from multiprocessing import Value
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Type, Union

import numpy as np
import torch
import torchvision
from PIL import Image, ImageFilter, ImageOps
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.datasets import STL10, ImageFolder
from tqdm import tqdm

try:
    from solo.data.h5_dataset import H5Dataset
except ImportError:
    _h5_available = False
else:
    _h5_available = True


def dataset_with_index(DatasetClass: Type[Dataset]) -> Type[Dataset]:
    """Factory for datasets that also returns the data index.

    Args:
        DatasetClass (Type[Dataset]): Dataset class to be wrapped.

    Returns:
        Type[Dataset]: dataset with index.
    """

    class DatasetWithIndex(DatasetClass):
        def __getitem__(self, index):
            data = super().__getitem__(index)
            return (index, *data)

    return DatasetWithIndex

# See any dataloder needs basically 3 functions to funciton properly: __getitem__, __len__, and __init__
# The main thing we want to implement is that 
class InMemoryDataset(Dataset):
    def __init__(self, base_dataset: Dataset, transform: Callable = None, num_workers: int = 16):
        self.base_dataset = base_dataset
        # Use provided transform or fallback to the one on the base dataset
        self.transform = transform if transform is not None else getattr(base_dataset, "transform", None)
        self.images = [None] * len(base_dataset)
        self.labels = [None] * len(base_dataset)

        print(f"Parallel Pre-loading dataset into RAM ({len(base_dataset)} samples using {num_workers} threads)...")
        
        # Temporarily remove transform from base_dataset to get raw images
        orig_transform = None
        if hasattr(base_dataset, "transform"):
            orig_transform = base_dataset.transform
            base_dataset.transform = None

        def load_and_convert(i):
            item = base_dataset[i]
            # Handle different return formats from __getitem__
            if isinstance(item, (list, tuple)):
                img, label = item[0], item[1]
            else:
                img, label = item, -1

            if isinstance(img, Image.Image):
                img = img.convert("RGB")
                # Resize to a reasonable size to save RAM if it's too large
                # Use thumbnail to strictly bound the longest edge to 320.
                # This avoids upscaling 'panoramic' images which T.resize(320) would do (constraining shortest edge).
                if max(img.size) > 320:
                    img.thumbnail((320, 320))
                return img, label
            return img, label

        # Use ThreadPoolExecutor for parallel I/O and decoding
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Process results as they complete to avoid a temporary list of the whole dataset
            for i, (img, label) in enumerate(tqdm(executor.map(load_and_convert, range(len(base_dataset))), total=len(base_dataset))):
                self.images[i] = img
                self.labels[i] = label

        # Restore original transform
        if hasattr(base_dataset, "transform"):
            base_dataset.transform = orig_transform

    def __getitem__(self, index):
        img = self.images[index]
        label = self.labels[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.images)


class CustomDatasetWithoutLabels(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.images = os.listdir(root)

    def __getitem__(self, index):
        path = self.root / self.images[index]
        x = Image.open(path).convert("RGB")
        if self.transform is not None:
            x = self.transform(x)
        return x, -1

    def __len__(self):
        return len(self.images)


class GaussianBlur:
    def __init__(self, sigma: Sequence[float] = None):
        """Gaussian blur as a callable object.

        Args:
            sigma (Sequence[float]): range to sample the radius of the gaussian blur filter.
                Defaults to [0.1, 2.0].
        """

        if sigma is None:
            sigma = [0.1, 2.0]

        self.sigma = sigma

    def __call__(self, img: Image) -> Image:
        """Applies gaussian blur to an input image.

        Args:
            img (Image): an image in the PIL.Image format.

        Returns:
            Image: blurred image.
        """

        sigma = random.uniform(self.sigma[0], self.sigma[1])
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img


class Solarization:
    """Solarization as a callable object."""

    def __call__(self, img: Image) -> Image:
        """Applies solarization to an input image.

        Args:
            img (Image): an image in the PIL.Image format.

        Returns:
            Image: solarized image.
        """

        return ImageOps.solarize(img)


class Equalization:
    def __call__(self, img: Image) -> Image:
        return ImageOps.equalize(img)


class NCropAugmentation:
    def __init__(self, transform: Callable, num_crops: int):
        """Creates a pipeline that apply a transformation pipeline multiple times.

        Args:
            transform (Callable): transformation pipeline.
            num_crops (int): number of crops to create from the transformation pipeline.
        """

        self.transform = transform
        self.num_crops = num_crops

    def __call__(self, x: Image) -> List[torch.Tensor]:
        """Applies transforms n times to generate n crops.

        Args:
            x (Image): an image in the PIL.Image format.

        Returns:
            List[torch.Tensor]: an image in the tensor format.
        """

        return [self.transform(x) for _ in range(self.num_crops)]

    def __repr__(self) -> str:
        return f"{self.num_crops} x [{self.transform}]"


class FullTransformPipeline:
    def __init__(self, transforms: Callable) -> None:
        self.transforms = transforms

    def __call__(self, x: Image) -> List[torch.Tensor]:
        """Applies transforms n times to generate n crops.

        Args:
            x (Image): an image in the PIL.Image format.

        Returns:
            List[torch.Tensor]: an image in the tensor format.
        """

        out = []
        for transform in self.transforms:
            out.extend(transform(x))
        return out

    def __repr__(self) -> str:
        return "\n".join(str(transform) for transform in self.transforms)


_SEED_MASK_64 = (1 << 64) - 1


def _mix_seed(base_seed: int, epoch: int, index: int, crop_idx: int) -> int:
    seed = int(base_seed) & _SEED_MASK_64
    for value in (epoch, index, crop_idx):
        value = int(value) & _SEED_MASK_64
        seed ^= (value + 0x9E3779B97F4A7C15 + ((seed << 6) & _SEED_MASK_64) + (seed >> 2))
        seed &= _SEED_MASK_64
    return seed


@contextmanager
def _temporary_random_seed(seed: int):
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)


class IndexedPretrainWrapper(Dataset):
    """Adds sample indexes and optional deterministic per-(epoch, sample, crop) augmentations."""

    def __init__(
        self,
        dataset: Dataset,
        transform: Optional[Callable] = None,
        deterministic: bool = False,
        deterministic_seed: int = 0,
    ):
        self.dataset = dataset
        self.transform = transform
        self.deterministic = bool(deterministic)
        self.deterministic_seed = int(deterministic_seed)
        self._epoch_ref = Value("i", 0)

        if self.deterministic and self.transform is not None and not isinstance(
            self.transform, FullTransformPipeline
        ):
            raise TypeError(
                "Deterministic pretrain augmentations require a FullTransformPipeline so crops can "
                "be seeded independently per (epoch, sample, crop)."
            )

    def set_epoch(self, epoch: int) -> None:
        with self._epoch_ref.get_lock():
            self._epoch_ref.value = int(epoch)

    def get_epoch(self) -> int:
        return int(self._epoch_ref.value)

    def get_epoch_ref(self):
        return self._epoch_ref

    def _apply_transform(self, img: Image, index: int):
        if self.transform is None:
            return img

        if not self.deterministic:
            return self.transform(img)

        out = []
        crop_idx = 0
        current_epoch = self.get_epoch()
        for transform in self.transform.transforms:
            base_transform = getattr(transform, "transform", transform)
            num_crops = int(getattr(transform, "num_crops", 1))
            for _ in range(num_crops):
                seed = _mix_seed(self.deterministic_seed, current_epoch, int(index), crop_idx)
                with _temporary_random_seed(seed):
                    out.append(base_transform(img))
                crop_idx += 1
        return out

    def __getitem__(self, index):
        item = self.dataset[index]
        if isinstance(item, (list, tuple)):
            img, label = item[0], item[1]
        else:
            img, label = item, -1

        img = self._apply_transform(img, index)
        return (index, img, label)

    def __len__(self):
        return len(self.dataset)


def _extract_labels_for_subset(dataset: Dataset) -> Optional[List[int]]:
    if isinstance(dataset, CustomDatasetWithoutLabels):
        return None
    if hasattr(dataset, "targets"):
        targets = getattr(dataset, "targets")
        if targets is None:
            return None
        return list(np.asarray(targets).tolist())
    if hasattr(dataset, "labels"):
        labels = getattr(dataset, "labels")
        if labels is None:
            return None
        labels = np.asarray(labels)
        if labels.ndim == 0:
            return None
        return list(labels.tolist())
    if hasattr(dataset, "samples"):
        return [int(label) for _, label in getattr(dataset, "samples")]
    if hasattr(dataset, "_data"):
        return [int(entry[2]) for entry in getattr(dataset, "_data")]
    return None


def _build_subset(dataset: Dataset, data_fraction: float) -> Dataset:
    if data_fraction <= 0:
        return dataset

    assert data_fraction < 1, "Only use data_fraction for values smaller than 1."
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(dataset))
    labels = _extract_labels_for_subset(dataset)
    stratify = None
    if labels is not None:
        unique_labels = set(labels)
        if len(unique_labels) > 1 and -1 not in unique_labels:
            stratify = labels

    subset_indices, _ = train_test_split(
        indices,
        train_size=data_fraction,
        stratify=stratify,
        random_state=42,
    )
    subset_indices = sorted(int(i) for i in subset_indices)
    return Subset(dataset, subset_indices)


def _build_raw_pretrain_dataset(
    dataset: str,
    train_data_path: Union[str, Path],
    data_format: str,
    no_labels: bool,
    download: bool,
) -> Dataset:
    if dataset in ["cifar10", "cifar100"]:
        DatasetClass = vars(torchvision.datasets)[dataset.upper()]
        return DatasetClass(train_data_path, train=True, download=download, transform=None)

    if dataset == "stl10":
        return STL10(train_data_path, split="train+unlabeled", download=download, transform=None)

    if dataset == "celeba":
        base_dataset = torchvision.datasets.CelebA(
            train_data_path,
            split="train",
            download=download,
            transform=None,
        )

        class CelebARawWithoutLabels(Dataset):
            def __init__(self, dataset):
                self.dataset = dataset

            def __getitem__(self, index):
                img, _ = self.dataset[index]
                return img, -1

            def __len__(self):
                return len(self.dataset)

        return CelebARawWithoutLabels(base_dataset)

    if dataset in ["imagenet", "imagenet100", "imagenet10"]:
        if data_format == "h5":
            assert _h5_available
            return H5Dataset(dataset, train_data_path, transform=None)
        return ImageFolder(train_data_path, transform=None)

    if dataset == "custom":
        dataset_class = CustomDatasetWithoutLabels if no_labels else ImageFolder
        return dataset_class(train_data_path, transform=None)

    raise ValueError(f"Unsupported dataset '{dataset}' for deterministic pretrain augmentation mode.")


def _prepare_deterministic_dataset(
    dataset: str,
    transform: Callable,
    train_data_path: Union[str, Path],
    data_format: str,
    no_labels: bool,
    download: bool,
    data_fraction: float,
    preload: bool,
    deterministic_seed: int,
) -> Dataset:
    base_dataset = _build_raw_pretrain_dataset(
        dataset=dataset,
        train_data_path=train_data_path,
        data_format=data_format,
        no_labels=no_labels,
        download=download,
    )
    base_dataset = _build_subset(base_dataset, data_fraction)
    if preload:
        base_dataset = InMemoryDataset(base_dataset, transform=None, num_workers=16)

    return IndexedPretrainWrapper(
        base_dataset,
        transform=transform,
        deterministic=True,
        deterministic_seed=deterministic_seed,
    )


def build_transform_pipeline(dataset, cfg):
    """Creates a pipeline of transformations given a dataset and an augmentation Cfg node.
    The node needs to be in the following format:
        crop_size: int
        [OPTIONAL] mean: float
        [OPTIONAL] std: float
        rrc:
            enabled: bool
            crop_min_scale: float
            crop_max_scale: float
        color_jitter:
            prob: float
            brightness: float
            contrast: float
            saturation: float
            hue: float
        grayscale:
            prob: float
        gaussian_blur:
            prob: float
        solarization:
            prob: float
        equalization:
            prob: float
        horizontal_flip:
            prob: float
    """

    MEANS_N_STD = {
        "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        "stl10": ((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
        "imagenet100": (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        "imagenet": (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        "imagenet10": (IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    }

    mean, std = MEANS_N_STD.get(
        dataset, (cfg.get("mean", IMAGENET_DEFAULT_MEAN), cfg.get("std", IMAGENET_DEFAULT_STD))
    )

    augmentations = []
    if cfg.rrc.enabled:
        augmentations.append(
            transforms.RandomResizedCrop(
                cfg.crop_size,
                scale=(cfg.rrc.crop_min_scale, cfg.rrc.crop_max_scale),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
        )
    else:
        augmentations.append(
            transforms.Resize(
                cfg.crop_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
        )

    if cfg.color_jitter.prob:
        augmentations.append(
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        cfg.color_jitter.brightness,
                        cfg.color_jitter.contrast,
                        cfg.color_jitter.saturation,
                        cfg.color_jitter.hue,
                    )
                ],
                p=cfg.color_jitter.prob,
            ),
        )

    if cfg.grayscale.prob:
        augmentations.append(transforms.RandomGrayscale(p=cfg.grayscale.prob))

    if cfg.gaussian_blur.prob:
        augmentations.append(transforms.RandomApply([GaussianBlur()], p=cfg.gaussian_blur.prob))

    if cfg.solarization.prob:
        augmentations.append(transforms.RandomApply([Solarization()], p=cfg.solarization.prob))

    if cfg.equalization.prob:
        augmentations.append(transforms.RandomApply([Equalization()], p=cfg.equalization.prob))

    if cfg.horizontal_flip.prob:
        augmentations.append(transforms.RandomHorizontalFlip(p=cfg.horizontal_flip.prob))

    augmentations.append(transforms.ToTensor())
    augmentations.append(transforms.Normalize(mean=mean, std=std))

    augmentations = transforms.Compose(augmentations)
    return augmentations


def prepare_n_crop_transform(
    transforms: List[Callable], num_crops_per_aug: List[int]
) -> NCropAugmentation:
    """Turns a single crop transformation to an N crops transformation.

    Args:
        transforms (List[Callable]): list of transformations.
        num_crops_per_aug (List[int]): number of crops per pipeline.

    Returns:
        NCropAugmentation: an N crop transformation.
    """

    assert len(transforms) == len(num_crops_per_aug)

    T = []
    for transform, num_crops in zip(transforms, num_crops_per_aug):
        T.append(NCropAugmentation(transform, num_crops))
    return FullTransformPipeline(T)


def prepare_datasets(
    dataset: str,
    transform: Callable,
    train_data_path: Optional[Union[str, Path]] = None,
    data_format: Optional[str] = "image_folder",
    no_labels: Optional[Union[str, Path]] = False,
    download: bool = True,
    data_fraction: float = -1.0,
    preload: bool = False,
    deterministic_augmentations: bool = False,
    deterministic_augmentations_seed: int = 0,
) -> Dataset:
    """Prepares the desired dataset.

    Args:
        dataset (str): the name of the dataset.
        transform (Callable): a transformation.
        train_dir (Optional[Union[str, Path]]): training data path. Defaults to None.
        data_format (Optional[str]): format of the data. Defaults to "image_folder".
            Possible values are "image_folder" and "h5".
        no_labels (Optional[bool]): if the custom dataset has no labels.
        data_fraction (Optional[float]): percentage of data to use. Use all data when set to -1.0.
            Defaults to -1.0.
        preload (bool): if the dataset should be pre-loaded into RAM.
    Returns:
        Dataset: the desired dataset with transformations.
    """

    if train_data_path is None:
        sandbox_folder = Path(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
        train_data_path = sandbox_folder / "datasets"

    if deterministic_augmentations:
        return _prepare_deterministic_dataset(
            dataset=dataset,
            transform=transform,
            train_data_path=train_data_path,
            data_format=data_format,
            no_labels=no_labels,
            download=download,
            data_fraction=data_fraction,
            preload=preload,
            deterministic_seed=deterministic_augmentations_seed,
        )

    if dataset in ["cifar10", "cifar100"]:
        DatasetClass = vars(torchvision.datasets)[dataset.upper()]
        train_dataset = dataset_with_index(DatasetClass)(
            train_data_path,
            train=True,
            download=download,
            transform=transform,
        )

    elif dataset == "stl10":
        train_dataset = dataset_with_index(STL10)(
            train_data_path,
            split="train+unlabeled",
            download=download,
            transform=transform,
        )

    elif dataset == "celeba":
        train_dataset = torchvision.datasets.CelebA(
            train_data_path,
            split="train",
            download=download,
            transform=transform,
        )
        if preload:
            train_dataset = InMemoryDataset(train_dataset, num_workers=16)

        # wrap with index and ensure label is -1
        class CelebAWithIndex(Dataset):
            def __init__(self, dataset):
                self.dataset = dataset

            def __getitem__(self, index):
                img, _ = self.dataset[index]
                return (index, img, -1) # hardcoded label -1 because CelebA has no labels

            def __len__(self):
                return len(self.dataset)

        train_dataset = CelebAWithIndex(train_dataset)

    elif dataset in ["imagenet", "imagenet100", "imagenet10"]:
        if data_format == "h5":
            assert _h5_available
            train_dataset = dataset_with_index(H5Dataset)(dataset, train_data_path, transform)
        else:
            train_dataset = ImageFolder(train_data_path, transform)
            if preload:
                train_dataset = InMemoryDataset(train_dataset, num_workers=16)

            # wrap with index
            class DatasetWithIndex(Dataset):
                def __init__(self, dataset):
                    self.dataset = dataset

                def __getitem__(self, index):
                    return (index, *self.dataset[index])

                def __len__(self):
                    return len(self.dataset)

            train_dataset = DatasetWithIndex(train_dataset)

    elif dataset == "custom":
        if no_labels:
            dataset_class = CustomDatasetWithoutLabels
        else:
            dataset_class = ImageFolder

        train_dataset = dataset_class(train_data_path, transform)
        if preload:
            train_dataset = InMemoryDataset(train_dataset)

        # wrap with index
        class DatasetWithIndex(Dataset):
            def __init__(self, dataset):
                self.dataset = dataset

            def __getitem__(self, index):
                return (index, *self.dataset[index])

            def __len__(self):
                return len(self.dataset)

        train_dataset = DatasetWithIndex(train_dataset)

    if data_fraction > 0:
        assert data_fraction < 1, "Only use data_fraction for values smaller than 1."
        from sklearn.model_selection import train_test_split

        if isinstance(train_dataset, CustomDatasetWithoutLabels):
            files = train_dataset.images
            (
                files,
                _,
            ) = train_test_split(files, train_size=data_fraction, random_state=42)
            train_dataset.images = files
        else:
            data = train_dataset.samples
            files = [f for f, _ in data]
            labels = [l for _, l in data]
            files, _, labels, _ = train_test_split(
                files, labels, train_size=data_fraction, stratify=labels, random_state=42
            )
            train_dataset.samples = [tuple(p) for p in zip(files, labels)]

    return train_dataset


def prepare_dataloader(
    train_dataset: Dataset,
    batch_size: int = 64,
    num_workers: int = 4,
    shuffle: bool = True,
    drop_last: bool = True,
    pin_memory: bool = True,
) -> DataLoader:
    """Prepares the training dataloader for pretraining.
    Args:
        train_dataset (Dataset): the name of the dataset.
        batch_size (int, optional): batch size. Defaults to 64.
        num_workers (int, optional): number of workers. Defaults to 4.
    Returns:
        DataLoader: the training dataloader with the desired dataset.
    """

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return train_loader
