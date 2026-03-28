import argparse
import os
from pathlib import Path


DEFAULT_SOURCE_CANDIDATES = [
    os.environ.get("IMAGENET_ROOT"),
    "/network/datasets/imagenet.var/imagenet_torchvision",
    "/datasets/imagenet",
    "/data/imagenet",
]


def load_imagenet100_classes() -> list[str]:
    repo_root = Path(__file__).resolve().parents[2]
    classes_file = repo_root / "solo" / "data" / "dataset_subset" / "imagenet100_classes.txt"
    return classes_file.read_text().strip().split()


def resolve_source_root(explicit_source_root: str | None) -> Path:
    candidates = [explicit_source_root, *DEFAULT_SOURCE_CANDIDATES]
    for candidate in candidates:
        if not candidate:
            continue
        candidate_path = Path(candidate).expanduser().resolve()
        if (candidate_path / "train").is_dir() and (candidate_path / "val").is_dir():
            return candidate_path
    raise FileNotFoundError(
        "Could not find a valid ImageNet root. "
        "Pass it explicitly or set IMAGENET_ROOT to a directory containing train/ and val/."
    )


def ensure_symlink(source_path: Path, dest_path: Path) -> bool:
    if dest_path.is_symlink():
        if dest_path.resolve() == source_path.resolve():
            return False
        raise FileExistsError(f"{dest_path} already points to {dest_path.resolve()}, expected {source_path}")
    if dest_path.exists():
        raise FileExistsError(f"{dest_path} already exists and is not a symlink")
    dest_path.symlink_to(source_path, target_is_directory=True)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Create an ImageNet-100 image-folder dataset using symlinks into a full ImageNet tree."
    )
    parser.add_argument(
        "full_imagenet_path",
        nargs="?",
        default=None,
        help="Path to the full ImageNet root containing train/ and val/.",
    )
    parser.add_argument(
        "imagenet100_path",
        nargs="?",
        default="datasets/imagenet100",
        help="Destination directory for the generated ImageNet-100 tree.",
    )
    args = parser.parse_args()

    source_root = resolve_source_root(args.full_imagenet_path)
    destination_root = Path(args.imagenet100_path).expanduser().resolve()
    classes = load_imagenet100_classes()

    created = 0
    for split in ["train", "val"]:
        split_root = destination_root / split
        split_root.mkdir(parents=True, exist_ok=True)
        for class_name in classes:
            source_path = source_root / split / class_name
            if not source_path.is_dir():
                raise FileNotFoundError(f"Missing source class directory: {source_path}")
            dest_path = split_root / class_name
            created += int(ensure_symlink(source_path, dest_path))

    print(f"ImageNet root: {source_root}")
    print(f"ImageNet-100 destination: {destination_root}")
    print(f"Classes linked: {len(classes)}")
    print(f"New symlinks created: {created}")


if __name__ == "__main__":
    main()
