from __future__ import annotations

import torch
from torch import nn


class SealCodeCNN(nn.Module):
	"""Small baseline CNN that predicts each digit in a seal code."""

	def __init__(self, code_length: int = 7, classes: int = 10) -> None:
		super().__init__()
		self.code_length = code_length
		self.features = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=5, padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Conv2d(16, 32, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.AdaptiveAvgPool2d((1, code_length)),
		)
		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Dropout(0.2),
			nn.Linear(64 * code_length, code_length * classes),
		)
		self.classes = classes

	def forward(self, images: torch.Tensor) -> torch.Tensor:
		logits = self.classifier(self.features(images))
		return logits.view(images.shape[0], self.code_length, self.classes)
