from __future__ import annotations

import argparse
import datetime
import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data_pipeline import SealDataset
from models.cnn import SealCodeCNN
from pipeline_config import DEFAULT_CONFIG, config_path, load_config, resolve_device
from preprocess import CODE_LENGTH


ROOT = Path(__file__).resolve().parent


def run_epoch(model, loader, criterion, device, optimizer=None):
	training = optimizer is not None
	model.train(training)
	total_loss = 0.0
	digit_correct = 0
	exact_correct = 0
	total = 0
	context = torch.enable_grad() if training else torch.no_grad()
	with context:
		for images, targets in loader:
			images, targets = images.to(device), targets.to(device)
			if training:
				optimizer.zero_grad(set_to_none=True)
			logits = model(images)
			loss = criterion(logits.transpose(1, 2), targets)
			if training:
				loss.backward()
				optimizer.step()
			predictions = logits.argmax(dim=-1)
			batch_size = targets.size(0)
			total_loss += loss.item() * batch_size
			digit_correct += (predictions == targets).sum().item()
			exact_correct += (predictions == targets).all(dim=1).sum().item()
			total += batch_size
	return total_loss / total, digit_correct / (total * CODE_LENGTH), exact_correct / total


def main() -> None:
	parser = argparse.ArgumentParser(description="Train the baseline seal-code CNN on train/val data.")
	parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
	parser.add_argument("--data-root", type=Path)
	parser.add_argument("--manifest-root", type=Path)
	parser.add_argument("--output", type=Path)
	parser.add_argument("--train-samples", type=int)
	parser.add_argument("--validation-samples", type=int)
	parser.add_argument("--epochs", type=int)
	parser.add_argument("--batch-size", type=int)
	parser.add_argument("--learning-rate", type=float)
	parser.add_argument("--device", choices=("cpu", "gpu"))
	parser.add_argument(
		"--crop-to-digits",
		action="store_true",
		help="Train on photos cropped to the digit row (via the classical approach's segmentation) instead of the full frame",
	)
	parser.add_argument(
		"--resume",
		action="store_true",
		help="Continue training from the existing checkpoint at --output, if present",
	)
	parser.add_argument(
		"--num-workers",
		type=int,
		default=min(8, os.cpu_count() or 1),
		help="Parallel data-loading processes; loads the next batch while the GPU is busy with the current one",
	)
	parser.add_argument(
		"--backup-every",
		type=int,
		default=5,
		help="Save a timestamped backup copy of the checkpoint every N epochs (0 disables). "
		"Backups never overwrite each other or the main checkpoint.",
	)
	args = parser.parse_args()
	config = load_config(args.config)
	config_root = args.config.resolve().parent
	data_root = config_path(args.data_root or config.get("data_root", "data"), config_root)
	manifest_root = config_path(
		args.manifest_root or config.get("manifest_root", "data/splits/split_seals"),
		config_root,
	)
	default_checkpoint = "artifacts/seal_code_cnn_cropped.pt" if args.crop_to_digits else config.get(
		"checkpoint", "artifacts/seal_code_cnn.pt"
	)
	output = config_path(args.output or default_checkpoint, config_root)
	train_samples = args.train_samples if args.train_samples is not None else config.get("train_samples")
	validation_samples = (
		args.validation_samples
		if args.validation_samples is not None
		else config.get("validation_samples")
	)
	epochs = args.epochs if args.epochs is not None else config.get("epochs", 10)
	batch_size = args.batch_size if args.batch_size is not None else config.get("batch_size", 32)
	learning_rate = args.learning_rate if args.learning_rate is not None else config.get("learning_rate", 1e-3)
	device = resolve_device(args.device or config.get("device", "gpu"))

	train_set = SealDataset(
		manifest_root / "train.csv",
		data_root / "train",
		augment=True,
		max_samples=train_samples,
		crop_to_digits=args.crop_to_digits,
	)
	val_set = SealDataset(
		manifest_root / "val.csv",
		data_root / "val",
		max_samples=validation_samples,
		crop_to_digits=args.crop_to_digits,
	)
	loader_kwargs = {
		"num_workers": args.num_workers,
		"persistent_workers": args.num_workers > 0,
	}
	train_loader = DataLoader(train_set, batch_size, shuffle=True, **loader_kwargs)
	val_loader = DataLoader(val_set, batch_size, shuffle=False, **loader_kwargs)
	model = SealCodeCNN(CODE_LENGTH).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	criterion = nn.CrossEntropyLoss()
	best_exact = -1.0

	if args.resume and output.is_file():
		checkpoint_data = torch.load(output, map_location=device, weights_only=True)
		model.load_state_dict(checkpoint_data["model"])
		_, _, best_exact = run_epoch(model, val_loader, criterion, device)
		print(f"Resumed from {output} (starting val exact accuracy {best_exact:.3%})")

	backup_dir = output.parent / "backups"
	run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

	print(f"Training on {device} with {len(train_set)} train and {len(val_set)} validation samples")
	for epoch in range(1, epochs + 1):
		train_loss, train_digit, train_exact = run_epoch(model, train_loader, criterion, device, optimizer)
		val_loss, val_digit, val_exact = run_epoch(model, val_loader, criterion, device)
		print(
			f"Epoch {epoch:02d} | train loss {train_loss:.4f} digit {train_digit:.3%} exact {train_exact:.3%} | "
			f"val loss {val_loss:.4f} digit {val_digit:.3%} exact {val_exact:.3%}"
		)
		if val_exact > best_exact:
			best_exact = val_exact
			output.parent.mkdir(parents=True, exist_ok=True)
			torch.save({"model": model.state_dict(), "code_length": CODE_LENGTH}, output)
			print(f"Saved best checkpoint to {output}")
		if args.backup_every and epoch % args.backup_every == 0:
			backup_dir.mkdir(parents=True, exist_ok=True)
			backup_path = backup_dir / f"{output.stem}_{run_id}_epoch{epoch:03d}.pt"
			torch.save(
				{"model": model.state_dict(), "code_length": CODE_LENGTH, "val_exact": val_exact},
				backup_path,
			)
			print(f"Backed up checkpoint to {backup_path}")


if __name__ == "__main__":
	main()
