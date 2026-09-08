from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch


IMAGE_SIZE = (256, 160)  # width, height
CODE_LENGTH = 7


def letterbox(image: np.ndarray, size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
	"""Resize an image without changing its aspect ratio."""
	target_width, target_height = size
	height, width = image.shape[:2]
	scale = min(target_width / width, target_height / height)
	resized = cv2.resize(
		image,
		(max(1, round(width * scale)), max(1, round(height * scale))),
		interpolation=cv2.INTER_AREA,
	)
	canvas = np.zeros((target_height, target_width), dtype=np.uint8)
	offset_x = (target_width - resized.shape[1]) // 2
	offset_y = (target_height - resized.shape[0]) // 2
	canvas[offset_y : offset_y + resized.shape[0], offset_x : offset_x + resized.shape[1]] = resized
	return canvas


def load_image(path: str | Path, size: tuple[int, int] = IMAGE_SIZE) -> torch.Tensor:
	image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
	if image is None:
		raise FileNotFoundError(f"Could not read image: {path}")
	image = letterbox(image, size)
	return torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)


def encode_code(value: str | int, length: int = CODE_LENGTH) -> torch.Tensor:
	text = str(value).strip()
	if not text.isdigit():
		raise ValueError(f"Code must contain only digits, got {value!r}")
	if len(text) > length:
		raise ValueError(f"Code {text!r} is longer than the configured {length} digits")
	return torch.tensor([int(char) for char in text.zfill(length)], dtype=torch.long)


def decode_code(digits: torch.Tensor | list[int]) -> str:
	values = digits.detach().cpu().flatten().tolist() if isinstance(digits, torch.Tensor) else digits
	return "".join(str(int(value)) for value in values)
