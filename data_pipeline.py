from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset

from preprocess import CODE_LENGTH, IMAGE_SIZE, encode_code, load_image


class SealDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        image_dir: str | Path,
        augment: bool = False,
        max_samples: int | None = None,
    ) -> None:
        self.manifest = Path(manifest)
        self.image_dir = Path(image_dir)
        self.augment = augment
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be positive or None")
        if not self.manifest.exists():
            raise FileNotFoundError(f"Manifest does not exist: {self.manifest}")
        self.samples = self._read_manifest()
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        if not self.samples:
            raise ValueError(f"Manifest contains no available image samples: {self.manifest}")

    def _read_manifest(self) -> list[tuple[Path, torch.Tensor]]:
        samples = []
        missing = 0
        with self.manifest.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter=";")
            if reader.fieldnames != ["filename", "number"]:
                raise ValueError(f"Expected columns filename;number in {self.manifest}")
            for row in reader:
                image_path = self.image_dir / row["filename"]
                if not image_path.is_file():
                    missing += 1
                    continue
                samples.append((image_path, encode_code(row["number"], CODE_LENGTH)))

        if missing:
            print(
                f"Warning: skipped {missing} missing images referenced by {self.manifest}"
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, target = self.samples[index]
        image = load_image(path, IMAGE_SIZE)
        if self.augment and torch.rand(()) < 0.5:
            image = torch.clamp(image + torch.randn_like(image) * 0.02, 0.0, 1.0)
        return image, target