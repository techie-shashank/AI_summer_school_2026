from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data_pipeline import SealDataset
from models.cnn import SealCodeCNN
from pipeline_config import DEFAULT_CONFIG, config_path, load_config, resolve_device
from preprocess import CODE_LENGTH, decode_code


ROOT = Path(__file__).resolve().parent


def main() -> None:
	parser = argparse.ArgumentParser(description="Evaluate a seal-code CNN on the validation set.")
	parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
	parser.add_argument("--checkpoint", type=Path)
	parser.add_argument("--manifest", type=Path)
	parser.add_argument("--image-dir", type=Path)
	parser.add_argument("--validation-samples", type=int)
	parser.add_argument("--batch-size", type=int)
	parser.add_argument("--device", choices=("cpu", "gpu"))
	args = parser.parse_args()
	config = load_config(args.config)
	config_root = args.config.resolve().parent
	checkpoint = config_path(args.checkpoint or config.get("checkpoint", "artifacts/seal_code_cnn.pt"), config_root)
	manifest = config_path(
		args.manifest or Path(config.get("manifest_root", "data/splits/split_seals")) / "val.csv",
		config_root,
	)
	image_dir = config_path(args.image_dir or Path(config.get("data_root", "data")) / "val", config_root)
	validation_samples = (
		args.validation_samples
		if args.validation_samples is not None
		else config.get("validation_samples")
	)
	batch_size = args.batch_size if args.batch_size is not None else config.get("batch_size", 32)
	device = resolve_device(args.device or config.get("device", "gpu"))

	checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=True)
	model = SealCodeCNN(checkpoint_data.get("code_length", CODE_LENGTH))
	model.load_state_dict(checkpoint_data["model"])
	model.to(device).eval()
	loader = DataLoader(
		SealDataset(manifest, image_dir, max_samples=validation_samples),
		batch_size,
		shuffle=False,
	)
	exact = digit = total = 0
	examples = []
	with torch.no_grad():
		for images, targets in loader:
			images, targets = images.to(device), targets.to(device)
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
