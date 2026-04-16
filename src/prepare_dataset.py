"""
Prepare a reproducible train/val/test split from data/raw/images.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


RAW_IMAGES_DIR = Path("data/raw/images")
PROCESSED_DIR = Path("data/processed")
SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a reproducible split under data/processed."
    )
    parser.add_argument("--raw-dir", type=Path, default=RAW_IMAGES_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--test-ratio", type=float, default=TEST_RATIO)
    return parser.parse_args()


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {train_ratio} + {val_ratio} + "
            f"{test_ratio} = {total}"
        )


def list_images(class_dir: Path) -> list[Path]:
    return sorted(
        path for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )


def compute_split_sizes(
    total: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[int, int, int]:
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count

    return train_count, val_count, test_count


def reset_split_dirs(output_dir: Path) -> None:
    for split_name in ("train", "val", "test"):
        split_dir = output_dir / split_name
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)


def copy_subset(files: list[Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for source in files:
        shutil.copy2(source, destination / source.name)


def main() -> None:
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    raw_dir = args.raw_dir
    output_dir = args.output_dir

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset directory not found: {raw_dir}")

    class_dirs = sorted(path for path in raw_dir.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(f"No class directories found in {raw_dir}")

    reset_split_dirs(output_dir)

    rng = random.Random(args.seed)
    for class_dir in class_dirs:
        files = list_images(class_dir)
        if not files:
            continue

        shuffled = files[:]
        rng.shuffle(shuffled)

        train_count, val_count, test_count = compute_split_sizes(
            len(shuffled),
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
        )

        train_files = shuffled[:train_count]
        val_files = shuffled[train_count:train_count + val_count]
        test_files = shuffled[train_count + val_count:train_count + val_count + test_count]

        copy_subset(train_files, output_dir / "train" / class_dir.name)
        copy_subset(val_files, output_dir / "val" / class_dir.name)
        copy_subset(test_files, output_dir / "test" / class_dir.name)

    print(f"Processed dataset created in {output_dir}")
    for class_dir in class_dirs:
        train_count = len(list_images(output_dir / "train" / class_dir.name))
        val_count = len(list_images(output_dir / "val" / class_dir.name))
        test_count = len(list_images(output_dir / "test" / class_dir.name))
        total_count = train_count + val_count + test_count
        print(
            f"{class_dir.name}: total={total_count}, train={train_count}, "
            f"val={val_count}, test={test_count}"
        )


if __name__ == "__main__":
    main()
