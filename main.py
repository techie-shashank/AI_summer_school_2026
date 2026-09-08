from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from models.cnn import SealCodeCNN
from preprocess import CODE_LENGTH, decode_code, load_image


ROOT = Path(__file__).resolve().parent


def main() -> None:
	parser = argparse.ArgumentParser(description="Predict numerical codes for seal images.")
	parser.add_argument("--input-dir", type=Path, required=True)
	parser.add_argument("--output-dir", type=Path, required=True)
	parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/seal_code_cnn.pt")
	parser.add_argument("--output-name", default="predictions.csv")
	args = parser.parse_args()

	checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
	model = SealCodeCNN(checkpoint.get("code_length", CODE_LENGTH))
	model.load_state_dict(checkpoint["model"])
	model.eval()
	images = sorted(args.input_dir.glob("*.png"))
	args.output_dir.mkdir(parents=True, exist_ok=True)
	with (args.output_dir / args.output_name).open("w", newline="", encoding="utf-8") as handle:
		writer = csv.writer(handle, delimiter=";")
		writer.writerow(["filename", "number"])
		with torch.no_grad():
			for path in images:
				prediction = model(load_image(path).unsqueeze(0)).argmax(dim=-1)[0]
				writer.writerow([path.name, decode_code(prediction)])
	print(f"Wrote {len(images)} predictions to {args.output_dir / args.output_name}")


if __name__ == "__main__":
	main()
