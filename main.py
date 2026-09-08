from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from models.cnn import SealCodeCNN
from pipeline_config import DEFAULT_CONFIG, config_path, load_config, resolve_device
from preprocess import CODE_LENGTH, decode_code, load_image


ROOT = Path(__file__).resolve().parent


def main() -> None:
	parser = argparse.ArgumentParser(description="Predict numerical codes for seal images.")
	parser.add_argument("--input-dir", type=Path, required=True)
	parser.add_argument("--output-dir", type=Path, required=True)
	parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
	parser.add_argument("--checkpoint", type=Path)
	parser.add_argument("--device", choices=("cpu", "gpu"))
	parser.add_argument("--output-name", default="predictions.csv")
	args = parser.parse_args()

	config = load_config(args.config)
	config_root = args.config.resolve().parent
	checkpoint = config_path(args.checkpoint or config.get("checkpoint", "artifacts/seal_code_cnn.pt"), config_root)
	device = resolve_device(args.device or config.get("device", "gpu"))
	checkpoint_data = torch.load(checkpoint, map_location=device, weights_only=True)
	model = SealCodeCNN(checkpoint_data.get("code_length", CODE_LENGTH))
	model.load_state_dict(checkpoint_data["model"])
	model.to(device).eval()
	images = sorted(args.input_dir.glob("*.png"))
	args.output_dir.mkdir(parents=True, exist_ok=True)
	with (args.output_dir / args.output_name).open("w", newline="", encoding="utf-8") as handle:
		writer = csv.writer(handle, delimiter=";")
		writer.writerow(["filename", "number"])
		with torch.no_grad():
			for path in images:
				prediction = model(load_image(path).unsqueeze(0).to(device)).argmax(dim=-1)[0]
				writer.writerow([path.name, decode_code(prediction)])
	print(f"Wrote {len(images)} predictions to {args.output_dir / args.output_name}")


if __name__ == "__main__":
	main()
