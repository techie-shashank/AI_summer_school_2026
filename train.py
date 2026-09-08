from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data_pipeline import SealDataset
from models.cnn import SealCodeCNN
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
	parser.add_argument("--data-root", type=Path, default=ROOT / "data")
	parser.add_argument("--manifest-root", type=Path, default=ROOT / "data/splits/split_seals")
	parser.add_argument("--output", type=Path, default=ROOT / "artifacts/seal_code_cnn.pt")
	parser.add_argument("--epochs", type=int, default=10)
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--learning-rate", type=float, default=1e-3)
	parser.add_argument("--workers", type=int, default=0)
	args = parser.parse_args()

	train_set = SealDataset(args.manifest_root / "train.csv", args.data_root / "train", augment=True)
	val_set = SealDataset(args.manifest_root / "val.csv", args.data_root / "val")
	train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.workers)
	val_loader = DataLoader(val_set, args.batch_size, shuffle=False, num_workers=args.workers)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = SealCodeCNN(CODE_LENGTH).to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
	criterion = nn.CrossEntropyLoss()
	best_exact = -1.0

	print(f"Training on {device} with {len(train_set)} train and {len(val_set)} validation samples")
	for epoch in range(1, args.epochs + 1):
		train_loss, train_digit, train_exact = run_epoch(model, train_loader, criterion, device, optimizer)
		val_loss, val_digit, val_exact = run_epoch(model, val_loader, criterion, device)
		print(
			f"Epoch {epoch:02d} | train loss {train_loss:.4f} digit {train_digit:.3%} exact {train_exact:.3%} | "
			f"val loss {val_loss:.4f} digit {val_digit:.3%} exact {val_exact:.3%}"
		)
		if val_exact > best_exact:
			best_exact = val_exact
			args.output.parent.mkdir(parents=True, exist_ok=True)
			torch.save({"model": model.state_dict(), "code_length": CODE_LENGTH}, args.output)
			print(f"Saved best checkpoint to {args.output}")


if __name__ == "__main__":
	main()
