from __future__ import annotations

import csv
import random
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from preprocess import CODE_LENGTH, IMAGE_SIZE, encode_code, load_image


class SealDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path | list[str | Path],
        image_dir: str | Path | list[str | Path],
        augment: bool = False,
        max_samples: int | None = None,
        crop_to_digits: bool = False,
    ) -> None:
        # Accepts either a single manifest/image_dir pair, or matching lists
        # of several (e.g. to train on train.csv + test.csv at once, each
        # with its own image folder).
        manifests = manifest if isinstance(manifest, (list, tuple)) else [manifest]
        image_dirs = image_dir if isinstance(image_dir, (list, tuple)) else [image_dir]
        if len(manifests) != len(image_dirs):
            raise ValueError("manifest and image_dir must have the same number of entries")
        self.manifest = [Path(m) for m in manifests]
        self.image_dir = [Path(d) for d in image_dirs]
        self.augment = augment
        self.crop_to_digits = crop_to_digits
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be positive or None")
        for m in self.manifest:
            if not m.exists():
                raise FileNotFoundError(f"Manifest does not exist: {m}")
        self.samples = self._read_manifest()
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        if not self.samples:
            raise ValueError(f"Manifest(s) contain no available image samples: {self.manifest}")

    def _read_manifest(self) -> list[tuple[Path, torch.Tensor]]:
        samples = []
        missing = 0
        for manifest, image_dir in zip(self.manifest, self.image_dir):
            with manifest.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle, delimiter=";")
                if reader.fieldnames != ["filename", "number"]:
                    raise ValueError(f"Expected columns filename;number in {manifest}")
                for row in reader:
                    image_path = image_dir / row["filename"]
                    if not image_path.is_file():
                        missing += 1
                        continue
                    samples.append((image_path, encode_code(row["number"], CODE_LENGTH)))

        if missing:
            print(f"Warning: skipped {missing} missing images referenced by {self.manifest}")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, target = self.samples[index]
        if self.crop_to_digits:
            from models.classical import CROPPED_IMAGE_SIZE, load_cropped_image

            image = load_cropped_image(path, CROPPED_IMAGE_SIZE)
        else:
            image = load_image(path, IMAGE_SIZE)
        if self.augment:
            image = self._augment(image)
        return image, target

    @staticmethod
    def _augment(image: torch.Tensor) -> torch.Tensor:
        # Rotation range widened from +/-5 to +/-30 degrees: real photos
        # (e.g. the professor's held-out set) showed camera-angle tilt well
        # beyond +/-5, which the model had never been trained to handle.
        # +/-30 comfortably covers the worst tilt observed (~20 deg) without
        # going so far that digits become ambiguous (e.g. a heavily rotated
        # 6 starting to look like a 9).
        if torch.rand(()) < 0.5:
            image = TF.affine(
                image,
                angle=random.uniform(-30, 30),
                translate=(random.uniform(-5, 5), random.uniform(-5, 5)),
                scale=random.uniform(0.95, 1.05),
                shear=0.0,
                fill=0.0,
            )
        if torch.rand(()) < 0.5:
            image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
        if torch.rand(()) < 0.5:
            image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
        if torch.rand(()) < 0.5:
            image = torch.clamp(image + torch.randn_like(image) * 0.02, 0.0, 1.0)
        return image