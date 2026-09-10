from __future__ import annotations

import random
import csv
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from models.classical import DIGIT_CANVAS_SIZE, normalize_digit, segment_digits


DIGIT_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def load_digit_tensor(image_path: str | Path) -> torch.Tensor:
	"""Load one digit with the same normalization used by segmented seal crops."""
	import cv2
	import numpy as np

	image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
	if image is None:
		raise FileNotFoundError(f"Could not read digit image: {image_path}")
	normalized = normalize_digit(image, DIGIT_CANVAS_SIZE)
	return torch.from_numpy(normalized.astype(np.float32) / 255.0).unsqueeze(0)


def digit_crops_from_seal(image_path: str | Path, code_length: int = 7) -> list[torch.Tensor]:
	"""Segment and normalize a seal's digits in left-to-right order."""
	import numpy as np

	crops = segment_digits(image_path, code_length)
	return [
		torch.from_numpy(normalize_digit(crop, DIGIT_CANVAS_SIZE).astype(np.float32) / 255.0).unsqueeze(0)
		for crop in crops
	]


class DigitDataset(Dataset):
	def __init__(self, root: str | Path, augment: bool = False, max_samples: int | None = None) -> None:
		self.root = Path(root)
		self.augment = augment
		paths = sorted(
			path for path in self.root.iterdir()
			if path.is_file() and path.suffix.lower() in DIGIT_EXTENSIONS
		)
		if max_samples is not None:
			if max_samples <= 0:
				raise ValueError("max_samples must be positive or None")
			paths = paths[:max_samples]
		self.samples = [(path, self._label(path)) for path in paths]
		if not self.samples:
			raise ValueError(f"No digit images found in {self.root}")

	@staticmethod
	def _label(path: Path) -> int:
		label = path.stem.rsplit("_", maxsplit=1)[-1]
		if not label.isdigit() or not 0 <= int(label) <= 9:
			raise ValueError(f"Could not parse digit label from {path.name}")
		return int(label)

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
		path, label = self.samples[index]
		image = load_digit_tensor(path)
		if self.augment:
			if torch.rand(()) < 0.5:
				image = TF.affine(
					image,
					angle=random.uniform(-8, 8),
					translate=(random.uniform(-2, 2), random.uniform(-2, 2)),
					scale=random.uniform(0.9, 1.1),
					shear=0.0,
					fill=0.0,
				)
		return image, torch.tensor(label, dtype=torch.long)


class SealDigitDataset(Dataset):
	"""Individual digit crops extracted from labeled seal photos."""

	def __init__(
		self,
		manifest: str | Path,
		image_dir: str | Path,
		augment: bool = False,
		max_images: int | None = None,
	) -> None:
		self.augment = augment
		manifest = Path(manifest)
		image_dir = Path(image_dir)
		self.samples: list[tuple[torch.Tensor, int]] = []
		missing = 0
		segmentation_failed = 0
		with manifest.open(newline="", encoding="utf-8") as handle:
			rows = list(csv.DictReader(handle, delimiter=";"))
		if max_images is not None:
			if max_images <= 0:
				raise ValueError("max_images must be positive or None")
			rows = rows[:max_images]
		total_rows = len(rows)
		for row_index, row in enumerate(rows, start=1):
			image_path = image_dir / row["filename"]
			if not image_path.is_file():
				missing += 1
				if row_index % 10 == 0 or row_index == total_rows:
					print(
						f"Preparing seal crops: {row_index}/{total_rows} images "
						f"({len(self.samples)} digit crops, {missing} skipped)",
						flush=True,
					)
				continue
			try:
				crops = digit_crops_from_seal(image_path, len(row["number"].strip()))
			except (FileNotFoundError, ValueError):
				segmentation_failed += 1
				if row_index % 10 == 0 or row_index == total_rows:
					print(
						f"Preparing seal crops: {row_index}/{total_rows} images "
						f"({len(self.samples)} digit crops, "
						f"{missing + segmentation_failed} skipped)",
						flush=True,
					)
				continue
			for crop, label in zip(crops, row["number"].strip().zfill(len(crops))):
				self.samples.append((crop, int(label)))
			if row_index % 10 == 0 or row_index == total_rows:
				print(
					f"Preparing seal crops: {row_index}/{total_rows} images "
					f"({len(self.samples)} digit crops, "
					f"{missing + segmentation_failed} skipped)",
					flush=True,
				)
		if missing:
			print(f"Warning: skipped {missing} missing seal images in {manifest}")
		if not self.samples:
			raise ValueError(f"No segmented digit crops found from {manifest}")

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
		image, label = self.samples[index]
		image = image.clone()
		if self.augment:
			if torch.rand(()) < 0.5:
				image = TF.affine(
					image,
					angle=random.uniform(-5, 5),
					translate=(random.uniform(-2, 2), random.uniform(-2, 2)),
					scale=random.uniform(0.95, 1.05),
					shear=0.0,
					fill=0.0,
				)
			if torch.rand(()) < 0.5:
				image = torch.clamp(image + torch.randn_like(image) * 0.02, 0.0, 1.0)
		return image, torch.tensor(label, dtype=torch.long)


class DigitCropDataset(Dataset):
	"""Dataset view over prepared digit tensors with optional augmentation."""

	def __init__(self, samples: list[tuple[torch.Tensor, int]], augment: bool = False) -> None:
		self.samples = samples
		self.augment = augment

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
		image, label = self.samples[index]
		image = image.clone()
		if self.augment:
			if torch.rand(()) < 0.5:
				image = TF.affine(
					image,
					angle=random.uniform(-5, 5),
					translate=(random.uniform(-2, 2), random.uniform(-2, 2)),
					scale=random.uniform(0.95, 1.05),
					shear=0.0,
					fill=0.0,
				)
			if torch.rand(()) < 0.5:
				image = torch.clamp(image + torch.randn_like(image) * 0.02, 0.0, 1.0)
		return image, torch.tensor(label, dtype=torch.long)