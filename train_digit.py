from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Subset

from digit_pipeline import DigitCropDataset, DigitDataset, SealDigitDataset
from models.digit_cnn import DigitCNN
from pipeline_config import DEFAULT_CONFIG, config_path, load_config, resolve_device


ROOT = Path(__file__).resolve().parent


def run_epoch(model, loader, criterion, device, optimizer=None) -> tuple[float, float]:
	training = optimizer is not None
	model.train(training)
	total_loss = 0.0
	correct = 0
	total = 0
	context = torch.enable_grad() if training else torch.no_grad()
	with context:
		for images, targets in loader:
			images, targets = images.to(device), targets.to(device)
			if training:
				optimizer.zero_grad(set_to_none=True)
			logits = model(images)
			loss = criterion(logits, targets)
			if training:
				loss.backward()
				optimizer.step()
			total_loss += loss.item() * targets.size(0)
			correct += (logits.argmax(dim=1) == targets).sum().item()
			total += targets.size(0)
	return total_loss / total, correct / total


def main() -> None:
	parser = argparse.ArgumentParser(description="Train a CNN to classify individual digits.")
	parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
	parser.add_argument("--dataset-root", type=Path)
	parser.add_argument("--manifest-root", type=Path)
	parser.add_argument("--data-root", type=Path)
	parser.add_argument("--source", choices=("seal", "clean", "mixed"))
	parser.add_argument("--output", type=Path)
	parser.add_argument("--samples", type=int)
	parser.add_argument("--epochs", type=int)
	parser.add_argument("--batch-size", type=int)
	parser.add_argument("--learning-rate", type=float)
	parser.add_argument("--device", choices=("cpu", "gpu"))
	args = parser.parse_args()

	config = load_config(args.config)
	config_root = args.config.resolve().parent
	dataset_root = config_path(
		args.dataset_root or config.get("digit_dataset_root", "data/ground_truth_chars_balanced"),
		config_root,
	)
	manifest_root = config_path(
		args.manifest_root or config.get("manifest_root", "data/splits/split_seals"),
		config_root,
	)
	data_root = config_path(args.data_root or config.get("data_root", "data"), config_root)
	output = config_path(
		args.output or config.get("digit_checkpoint", "artifacts/digit_cnn.pt"),
		config_root,
	)
	samples = args.samples if args.samples is not None else config.get("digit_samples")
	source = args.source or config.get("digit_source", "seal")
	epochs = args.epochs if args.epochs is not None else config.get("digit_epochs", 10)
	batch_size = args.batch_size if args.batch_size is not None else config.get("batch_size", 32)
	learning_rate = args.learning_rate if args.learning_rate is not None else config.get("learning_rate", 1e-3)
	device = resolve_device(args.device or config.get("device", "gpu"))

	torch.manual_seed(42)
	random.seed(42)
	if source == "clean":
		base_dataset = DigitDataset(dataset_root, augment=False, max_samples=samples)
		train_dataset = DigitDataset(dataset_root, augment=True, max_samples=samples)
		val_dataset = DigitDataset(dataset_root, augment=False, max_samples=samples)
	elif source == "seal":
		base_dataset = SealDigitDataset(manifest_root / "train.csv", data_root / "train", max_images=samples)
		train_dataset = DigitCropDataset(base_dataset.samples, augment=True)
		val_dataset = DigitCropDataset(base_dataset.samples, augment=False)
	else:
		clean_base = DigitDataset(dataset_root, augment=False, max_samples=samples)
		seal_base = SealDigitDataset(manifest_root / "train.csv", data_root / "train", max_images=samples)
		base_dataset = ConcatDataset([clean_base, seal_base])
		clean_train = DigitDataset(dataset_root, augment=True, max_samples=samples)
		seal_train = SealDigitDataset(manifest_root / "train.csv", data_root / "train", augment=True, max_images=samples)
		clean_val = DigitDataset(dataset_root, augment=False, max_samples=samples)
		seal_val = SealDigitDataset(manifest_root / "train.csv", data_root / "train", max_images=samples)
		train_dataset = ConcatDataset([clean_train, seal_train])
		val_dataset = ConcatDataset([clean_val, seal_val])
	validation_size = max(1, round(len(base_dataset) * 0.2))
	training_size = len(base_dataset) - validation_size
	indices = torch.randperm(
		len(base_dataset), generator=torch.Generator().manual_seed(42)
	).tolist()
	train_set = Subset(train_dataset, indices[:training_size])
	val_set = Subset(val_dataset, indices[training_size:])
	train_loader = DataLoader(train_set, batch_size, shuffle=True)
	val_loader = DataLoader(val_set, batch_size, shuffle=False)
	model = DigitCNN().to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	criterion = nn.CrossEntropyLoss()
	best_accuracy = -1.0

	print(f"Training digit CNN on {device} with {len(train_set)} train and {len(val_set)} validation crops")
	for epoch in range(1, epochs + 1):
		train_loss, train_accuracy = run_epoch(model, train_loader, criterion, device, optimizer)
		val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device)
		print(
			f"Epoch {epoch:02d} | train loss {train_loss:.4f} accuracy {train_accuracy:.3%} | "
			f"val loss {val_loss:.4f} accuracy {val_accuracy:.3%}"
		)
		if val_accuracy > best_accuracy:
			best_accuracy = val_accuracy
			output.parent.mkdir(parents=True, exist_ok=True)
			torch.save({"model": model.state_dict()}, output)
			print(f"Saved best digit checkpoint to {output}")


if __name__ == "__main__":
	main()