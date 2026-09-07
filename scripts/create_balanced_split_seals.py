from __future__ import annotations

import argparse
import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path


# Default split ratios
DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VAL_RATIO = 0.10
DEFAULT_TEST_PUBLIC_RATIO = 0.10
DEFAULT_TEST_PRIVATE_RATIO = 0.10

DEFAULT_SEED = 42


@dataclass
class ImageInfo:
    path: Path
    original_number: str


def parse_filename(image_path: Path) -> str | None:
    """
    Parse filename:

        10319_1580784_1.png

    Returns:
        "1580784"

    If the last number is 0, returns None and
    the image will be skipped.
    """

    parts = image_path.stem.split("_")

    if len(parts) < 3:
        raise ValueError(
            f"Unexpected filename format: {image_path.name}. "
            f"Expected something like 10319_1580784_1.png"
        )

    # Last number before .png
    last_number = parts[-1]

    if not last_number.isdigit():
        raise ValueError(
            f"Could not parse last number from filename: "
            f"{image_path.name}"
        )

    # Skip *_0.png
    if int(last_number) == 0:
        return None

    # Number we want to store in CSV
    middle_number = parts[-2]

    if not middle_number.isdigit():
        raise ValueError(
            f"Could not parse middle number from filename: "
            f"{image_path.name}"
        )

    return middle_number


def collect_images(dataset_root: Path) -> list[ImageInfo]:
    """
    Recursively find PNG files in all subdirectories.
    """

    images: list[ImageInfo] = []

    for image_path in sorted(dataset_root.rglob("*.png")):

        if not image_path.is_file():
            continue

        original_number = parse_filename(image_path)

        # *_0.png files are skipped
        if original_number is None:
            continue

        images.append(
            ImageInfo(
                path=image_path,
                original_number=original_number,
            )
        )

    return images


def split_images(
    images: list[ImageInfo],
    train_ratio: float,
    val_ratio: float,
    test_public_ratio: float,
    test_private_ratio: float,
    rng: random.Random,
) -> dict[str, list[ImageInfo]]:
    """
    Randomly shuffle all images and split them into:

        train
        val
        test_public
        test_private
    """

    shuffled = list(images)
    rng.shuffle(shuffled)

    total = len(shuffled)

    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_public_count = int(total * test_public_ratio)

    train_end = train_count
    val_end = train_end + val_count
    test_public_end = val_end + test_public_count

    train_images = shuffled[:train_end]

    val_images = shuffled[
        train_end:val_end
    ]

    test_public_images = shuffled[
        val_end:test_public_end
    ]

    # Everything remaining goes to the private test set.
    # This also makes sure rounding errors are handled automatically.
    test_private_images = shuffled[
        test_public_end:
    ]

    return {
        "train": train_images,
        "val": val_images,
        "test_public": test_public_images,
        "test_private": test_private_images,
    }


def create_split(
    split_name: str,
    images: list[ImageInfo],
    output_dir: Path,
) -> None:
    """
    Copy images to the split directory and rename them:

        00000.png
        00001.png
        00002.png
        ...

    Also create CSV:

        filename;number
        00000.png;1580784
        00001.png;1234567
        ...
    """

    split_dir = output_dir / split_name
    split_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = output_dir / f"{split_name}.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.writer(
            csv_file,
            delimiter=";",
            lineterminator="\n",
        )

        writer.writerow(
            [
                "filename",
                "number",
            ]
        )

        for index, image in enumerate(images):

            new_filename = f"{index:05d}.png"

            destination = split_dir / new_filename

            shutil.copy2(
                image.path,
                destination,
            )

            writer.writerow(
                [
                    new_filename,
                    image.original_number,
                ]
            )

    print(
        f"{split_name:13s}: "
        f"{len(images):6d} images"
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Create randomized train/validation/public-test/"
            "private-test datasets from PNG images."
        )
    )

    parser.add_argument(
        "dataset_root",
        type=Path,
        help=(
            "Input directory containing the dataset "
            "and its subdirectories."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Output directory where the generated "
            "dataset will be stored."
        ),
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_TRAIN_RATIO,
        help="Training ratio (default: 0.70).",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_VAL_RATIO,
        help="Validation ratio (default: 0.10).",
    )

    parser.add_argument(
        "--test-public-ratio",
        type=float,
        default=DEFAULT_TEST_PUBLIC_RATIO,
        help="Public test ratio (default: 0.10).",
    )

    parser.add_argument(
        "--test-private-ratio",
        type=float,
        default=DEFAULT_TEST_PRIVATE_RATIO,
        help="Private test ratio (default: 0.10).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed (default: 42).",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Allow using an existing non-empty output directory."
        ),
    )

    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Validate ratios
    # ---------------------------------------------------------------

    ratios = (
        args.train_ratio,
        args.val_ratio,
        args.test_public_ratio,
        args.test_private_ratio,
    )

    if any(ratio < 0 for ratio in ratios):
        raise ValueError(
            "Ratios cannot be negative."
        )

    ratio_sum = sum(ratios)

    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError(
            "Ratios must sum to 1.0. "
            f"Got {ratio_sum:.6f}."
        )

    # ---------------------------------------------------------------
    # Validate input
    # ---------------------------------------------------------------

    if not args.dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: "
            f"{args.dataset_root}"
        )

    if not args.dataset_root.is_dir():
        raise NotADirectoryError(
            f"Dataset root is not a directory: "
            f"{args.dataset_root}"
        )

    # ---------------------------------------------------------------
    # Validate output
    # ---------------------------------------------------------------

    if args.output_dir.exists():

        if not args.output_dir.is_dir():
            raise NotADirectoryError(
                f"Output path is not a directory: "
                f"{args.output_dir}"
            )

        if not args.force and any(args.output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory already exists and "
                f"is not empty:\n"
                f"  {args.output_dir}\n\n"
                f"Use --force to overwrite it."
            )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # Collect images
    # ---------------------------------------------------------------

    print()
    print("Input:")
    print(f"  {args.dataset_root}")

    print()
    print("Searching for images...")

    images = collect_images(
        args.dataset_root
    )

    if not images:
        raise ValueError(
            "No valid PNG images found."
        )

    print(
        f"Found {len(images)} valid images."
    )

    # ---------------------------------------------------------------
    # Random split
    # ---------------------------------------------------------------

    rng = random.Random(
        args.seed
    )

    splits = split_images(
        images=images,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_public_ratio=args.test_public_ratio,
        test_private_ratio=args.test_private_ratio,
        rng=rng,
    )

    # ---------------------------------------------------------------
    # Print split information
    # ---------------------------------------------------------------

    print()
    print("Dataset split:")

    total = len(images)

    for split_name, split_images_list in splits.items():

        percentage = (
            len(split_images_list)
            / total
            * 100.0
        )

        print(
            f"  {split_name:13s}: "
            f"{len(split_images_list):6d} "
            f"({percentage:5.1f} %)"
        )

    # ---------------------------------------------------------------
    # Create datasets
    # ---------------------------------------------------------------

    print()
    print("Creating output:")

    for split_name, split_images_list in splits.items():

        create_split(
            split_name=split_name,
            images=split_images_list,
            output_dir=args.output_dir,
        )

    # ---------------------------------------------------------------
    # Final information
    # ---------------------------------------------------------------

    print()
    print("Done.")
    print()
    print(
        f"Output directory:\n"
        f"  {args.output_dir}"
    )

    print()
    print(
        f"Random seed:\n"
        f"  {args.seed}"
    )

    print()
    print("Generated:")
    print(
        f"  {args.output_dir / 'train'}"
    )
    print(
        f"  {args.output_dir / 'val'}"
    )
    print(
        f"  {args.output_dir / 'test_public'}"
    )
    print(
        f"  {args.output_dir / 'test_private'}"
    )

    print()
    print("CSV files:")
    print(
        f"  {args.output_dir / 'train.csv'}"
    )
    print(
        f"  {args.output_dir / 'val.csv'}"
    )
    print(
        f"  {args.output_dir / 'test_public.csv'}"
    )
    print(
        f"  {args.output_dir / 'test_private.csv'}"
    )


if __name__ == "__main__":
    main()

