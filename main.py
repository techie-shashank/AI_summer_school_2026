from __future__ import annotations

import argparse
import csv
from pathlib import Path

from inference import run_inference
from pipeline_config import DEFAULT_CONFIG, load_config


ROOT = Path(__file__).resolve().parent
TEAM_NAME = "PinkOps"


def main() -> None:
	parser = argparse.ArgumentParser(description="Predict numerical codes for seal images from the default test split.")
	parser.add_argument("--input-dir", type=Path, help="Directory of seal images to predict. Defaults to the configured test split.")
	parser.add_argument("--output-dir", type=Path, required=True)
	parser.add_argument("--approach", choices=("cnn", "digit-cnn", "classical", "vlm"), default="cnn")
	parser.add_argument(
		"--voting",
		action="store_true",
		help="With --approach vlm: query the model 3x and keep the majority answer",
	)
	parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
	parser.add_argument("--checkpoint", type=Path)
	parser.add_argument("--device", choices=("cpu", "gpu"))
	parser.add_argument("--split", choices=("test", "val"), default="test", help="Default image directory when --input-dir is not provided")
	parser.add_argument("--output-name", default=f"{TEAM_NAME}.csv")
	args = parser.parse_args()

	config = load_config(args.config)
	config_root = args.config.resolve().parent
	data_root = Path(config.get("data_root", "data"))
	input_dir = args.input_dir or config_root / data_root / args.split
	images = sorted(input_dir.glob("*.png"))
	if not images:
		raise FileNotFoundError(f"No PNG images found in {input_dir}; set --input-dir or --split to a populated dataset folder")
	result = run_inference(
		args.approach,
		images,
		config,
		config_root,
		checkpoint=args.checkpoint,
		device_name=args.device,
		batch_size=config.get("batch_size", 32),
		voting=args.voting,
	)

	args.output_dir.mkdir(parents=True, exist_ok=True)
	with (args.output_dir / args.output_name).open("w", newline="", encoding="utf-8") as handle:
		writer = csv.writer(handle, delimiter=";")
		writer.writerow(["filename", "number"])
		for path, code in zip(images, result.predictions):
			writer.writerow([path.name, code or ""])
	print(f"Wrote {len(images)} predictions to {args.output_dir / args.output_name}")


if __name__ == "__main__":
	main()
