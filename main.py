from __future__ import annotations

import argparse
import csv
from pathlib import Path

from inference import run_inference
from pipeline_config import DEFAULT_CONFIG, load_config


ROOT = Path(__file__).resolve().parent
TEAM_NAME = "PinkOps"


def main() -> None:
	parser = argparse.ArgumentParser(description="Predict numerical codes for seal images.")
	parser.add_argument("--input-dir", type=Path, required=True)
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
	parser.add_argument("--output-name", default=f"{TEAM_NAME}.csv")
	args = parser.parse_args()

	images = sorted(args.input_dir.glob("*.png"))
	config = load_config(args.config)
	result = run_inference(
		args.approach,
		images,
		config,
		args.config.resolve().parent,
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
