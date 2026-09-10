from __future__ import annotations

import torch
from torch import nn


class DigitCNN(nn.Module):
	"""Small ten-class CNN that recognizes one cropped digit at a time."""

	def __init__(self, classes: int = 10) -> None:
		super().__init__()
		self.features = nn.Sequential(
			nn.Conv2d(1, 16, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Conv2d(16, 32, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.AdaptiveAvgPool2d((1, 1)),
		)
		self.classifier = nn.Sequential(
			nn.Flatten(),
			nn.Dropout(0.2),
			nn.Linear(64, classes),
		)

	def forward(self, images: torch.Tensor) -> torch.Tensor:
		return self.classifier(self.features(images))