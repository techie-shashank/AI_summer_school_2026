from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = DEFAULT_PROJECT_ROOT / "seals-dataset/ground_truth_chars_balanced"
DEFAULT_OUTPUT_DIR = DEFAULT_PROJECT_ROOT / "splits/ground_truth_chars_balanced"


def parse_label(image_path: Path) -> int:
    label_text = image_path.stem.rsplit("_", maxsplit=1)[-1]
    label_char = label_text[-1]
    if not label_char.isdigit():
        raise ValueError(f"Could not parse numeric label from filename: {image_path.name}")
    return int(label_char)


def relative_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def collect_images(dataset_root: Path, extensions: tuple[str, ...]) -> dict[int, list[Path]]:
    by_label: dict[int, list[Path]] = defaultdict(list)
    for image_path in sorted(dataset_root.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in extensions:
            continue
        by_label[parse_label(image_path)].append(image_path)
    return dict(sorted(by_label.items()))


def split_class_paths(
    paths: list[Path],
    train_ratio: float,
    val_ratio: float,
    rng: random.Random,
) -> tuple[list[Path], list[Path], list[Path]]:
    shuffled = list(paths)
    rng.shuffle(shuffled)

    train_count = int(len(shuffled) * train_ratio)
    val_count = int(len(shuffled) * val_ratio)

    train_paths = shuffled[:train_count]
    val_paths = shuffled[train_count : train_count + val_count]
    test_paths = shuffled[train_count + val_count :]
    return train_paths, val_paths, test_paths


def write_manifest(
    manifest_path: Path,
    rows: list[tuple[Path, int]],
    project_root: Path,
    force: bool,
) -> None:
    if manifest_path.exists() and not force:
        raise FileExistsError(f"{manifest_path} already exists. Pass --force to overwrite it.")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{relative_path(image_path, project_root)} {label}\n"
        for image_path, label in sorted(rows, key=lambda row: row[0].as_posix())
    ]
    manifest_path.write_text("".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create stratified train/val/test manifests for seal numeral images."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError(f"Ratios must sum to 1.0, got {ratio_sum}")
    if not args.dataset_root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {args.dataset_root}")

    extensions = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
    by_label = collect_images(args.dataset_root, extensions)
    if not by_label:
        raise ValueError(f"No image files found in: {args.dataset_root}")

    rng = random.Random(args.seed)
    splits: dict[str, list[tuple[Path, int]]] = {"train": [], "val": [], "test": []}

    for label, paths in by_label.items():
        train_paths, val_paths, test_paths = split_class_paths(
            paths,
            args.train_ratio,
            args.val_ratio,
            rng,
        )
        splits["train"].extend((path, label) for path in train_paths)
        splits["val"].extend((path, label) for path in val_paths)
        splits["test"].extend((path, label) for path in test_paths)

        print(
            f"class {label}: "
            f"train={len(train_paths)}, val={len(val_paths)}, test={len(test_paths)}"
        )

    for split_name, rows in splits.items():
        write_manifest(
            args.output_dir / f"{split_name}.txt",
            rows,
            args.project_root,
            args.force,
        )

    print()
    print(f"Wrote manifests to: {args.output_dir}")
    for split_name, rows in splits.items():
        print(f"{split_name}: {len(rows)} images")


if __name__ == "__main__":
    main()
