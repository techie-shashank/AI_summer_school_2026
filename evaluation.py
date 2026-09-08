from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data_pipeline import SealDataset
from models.cnn import SealCodeCNN
from preprocess import CODE_LENGTH, decode_code


ROOT = Path(__file__).resolve().parent


def main() -> None:
	parser = argparse.ArgumentParser(description="Evaluate a seal-code CNN on the validation set.")
	parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/seal_code_cnn.pt")
	parser.add_argument("--manifest", type=Path, default=ROOT / "data/splits/split_seals/val.csv")
	parser.add_argument("--image-dir", type=Path, default=ROOT / "data/val")
	parser.add_argument("--batch-size", type=int, default=32)
	args = parser.parse_args()

	checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
	model = SealCodeCNN(checkpoint.get("code_length", CODE_LENGTH))
	model.load_state_dict(checkpoint["model"])
	model.eval()
	loader = DataLoader(SealDataset(args.manifest, args.image_dir), args.batch_size, shuffle=False)
	exact = digit = total = 0
	examples = []
	with torch.no_grad():
		for images, targets in loader:
			predictions = model(images).argmax(dim=-1)
			exact += (predictions == targets).all(dim=1).sum().item()
			digit += (predictions == targets).sum().item()
			total += targets.size(0)
			for prediction, target in zip(predictions, targets):
				if len(examples) < 5 and not torch.equal(prediction, target):
					examples.append((decode_code(target), decode_code(prediction)))
	print(f"Samples: {total}")
	print(f"Exact-code accuracy: {exact / total:.3%}")
	print(f"Digit accuracy: {digit / (total * CODE_LENGTH):.3%}")
	if examples:
		print("Example errors (target -> prediction):")
		print("\n".join(f"{target} -> {prediction}" for target, prediction in examples))


if __name__ == "__main__":
	main()
