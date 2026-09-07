from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = DEFAULT_PROJECT_ROOT / "seals-dataset/ground_truth_chars_balanced"


class SealDigitsDataset(Dataset):
    """PyTorch dataset for cropped seal numeral images.

    Labels are read from the filename. For a file named ``102423_7.tif``, the
    class label is ``7``. Pass ``manifest_path`` to load a predefined split.
    Manifest rows must contain ``image_path label``. Relative image paths are
    resolved from ``project_root``.
    """

    def __init__(
        self,
        root: str | Path = DEFAULT_DATASET_ROOT,
        manifest_path: str | Path | None = None,
        project_root: str | Path = DEFAULT_PROJECT_ROOT,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        image_size: tuple[int, int] | None = None,
        return_path: bool = False,
        extensions: tuple[str, ...] = (".tif", ".tiff", ".png", ".jpg", ".jpeg"),
    ) -> None:
        self.root = Path(root)
        self.manifest_path = Path(manifest_path) if manifest_path is not None else None
        self.project_root = Path(project_root)
        self.transform = transform
        self.target_transform = target_transform
        self.image_size = image_size
        self.return_path = return_path
        self.extensions = tuple(extension.lower() for extension in extensions)

        if self.manifest_path is not None:
            self.samples = self._load_manifest_samples()
        else:
            if not self.root.exists():
                raise FileNotFoundError(f"Dataset directory does not exist: {self.root}")
            self.samples = self._load_directory_samples()

        if not self.samples:
            raise ValueError("No image samples found")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("L")

        if self.image_size is not None:
            image = self._letterbox_resize(image, self.image_size)

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = self._pil_to_tensor(image)

        if self.target_transform is not None:
            label = self.target_transform(label)

        if self.return_path:
            return image, label, image_path
        return image, label

    def class_counts(self) -> dict[int, int]:
        return dict(sorted(Counter(label for _, label in self.samples).items()))

    def _load_directory_samples(self) -> list[tuple[Path, int]]:
        samples = []
        for image_path in sorted(self.root.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in self.extensions:
                continue
            samples.append((image_path, self._label_from_path(image_path)))
        return samples

    def _load_manifest_samples(self) -> list[tuple[Path, int]]:
        manifest_path = self.manifest_path
        if not manifest_path.is_absolute():
            manifest_path = self.project_root / manifest_path
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest file does not exist: {manifest_path}")

        samples = []
        for line_number, line in enumerate(manifest_path.read_text().splitlines(), start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(
                    f"Expected 'image_path label' in {manifest_path}:{line_number}, got: {line}"
                )

            image_path = Path(parts[0])
            if not image_path.is_absolute():
                image_path = self.project_root / image_path
            samples.append((image_path, int(parts[1])))
        return samples

    @staticmethod
    def _label_from_path(image_path: Path) -> int:
        label_text = image_path.stem.rsplit("_", maxsplit=1)[-1]
        label_char = label_text[-1]
        if not label_char.isdigit():
            raise ValueError(f"Could not parse numeric label from filename: {image_path.name}")
        return int(label_char)

    @staticmethod
    def _letterbox_resize(image: Image.Image, image_size: tuple[int, int]) -> Image.Image:
        target_width, target_height = image_size
        if target_width <= 0 or target_height <= 0:
            raise ValueError(f"image_size must contain positive dimensions, got: {image_size}")

        source_width, source_height = image.size
        source_ratio = source_width / source_height
        target_ratio = target_width / target_height

        if source_ratio > target_ratio:
            canvas_width = source_width
            canvas_height = round(source_width / target_ratio)
        else:
            canvas_height = source_height
            canvas_width = round(source_height * target_ratio)

        canvas = Image.new(image.mode, (canvas_width, canvas_height), color=0)
        offset = (
            (canvas_width - source_width) // 2,
            (canvas_height - source_height) // 2,
        )
        canvas.paste(image, offset)
        return canvas.resize(image_size, resample=Image.Resampling.LANCZOS)

    @staticmethod
    def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
        data = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(data).unsqueeze(0)


if __name__ == "__main__":
    dataset = SealDigitsDataset(
        manifest_path="splits/ground_truth_chars_balanced/train.txt",
        image_size=(64, 64),
        return_path=True,
    )
    image, label, path = dataset[0]
    print(f"Images: {len(dataset)}")
    print(f"Class counts: {dataset.class_counts()}")
    print(f"First sample: image_shape={tuple(image.shape)}, label={label}, path={path}")
