from __future__ import annotations

import argparse
import csv
from pathlib import Path

from inference import run_inference
from pipeline_config import DEFAULT_CONFIG, config_path, load_config
from preprocess import CODE_LENGTH


def load_validation_rows(manifest, image_dir, max_samples):
	rows = []
	missing = 0
	with manifest.open(newline="", encoding="utf-8") as handle:
		reader = csv.DictReader(handle, delimiter=";")
		if reader.fieldnames != ["filename", "number"]:
			raise ValueError(f"Expected columns filename;number in {manifest}")
		for row in reader:
			image_path = image_dir / row["filename"]
			if not image_path.is_file():
				missing += 1
				continue
			rows.append((image_path, str(row["number"]).zfill(CODE_LENGTH)))
	if max_samples is not None:
		if max_samples <= 0:
			raise ValueError("validation_samples must be positive or None")
		rows = rows[:max_samples]
	if missing:
		print(f"Warning: skipped {missing} missing images referenced by {manifest}")
	if not rows:
		raise ValueError(f"No available validation images found in {image_dir}")
	return rows


def evaluate_predictions(targets, predictions, inference_times, show_examples=True):
	exact = 0
	valid_predictions = 0
	examples = []
	for target, prediction in zip(targets, predictions):
		if prediction is not None and len(prediction) == CODE_LENGTH and prediction.isdigit():
			valid_predictions += 1
			if target == prediction:
				exact += 1
			elif len(examples) < 5:
				examples.append((target, prediction))
	targets_counted = len(targets)
	print(f"Samples: {targets_counted}")
	print(f"Valid predictions: {valid_predictions}/{targets_counted}")
	print(f"Exact-code accuracy: {exact / targets_counted:.3%}")
	if inference_times:
		total_time = sum(inference_times)
		average_time = total_time / len(inference_times)
		throughput = len(inference_times) / total_time if total_time else 0.0
	else:
		total_time = average_time = throughput = 0.0
	print(f"Total inference time: {total_time:.4f} s")
	print(f"Average inference time: {average_time:.4f} s/image")
	print(f"Throughput: {throughput:.2f} images/s")
	if show_examples and examples:
		print("Example errors (target -> prediction):")
		print("\n".join(f"{target} -> {prediction or '<empty>'}" for target, prediction in examples))
	return {
		"valid": valid_predictions,
		"exact": exact / targets_counted,
		"total_time": total_time,
		"average_time": average_time,
		"throughput": throughput,
	}


def main() -> None:
	parser = argparse.ArgumentParser(description="Evaluate a recognition approach on the validation set.")
	parser.add_argument(
		"--approach",
		choices=("cnn", "digit-cnn", "classical", "vlm", "all"),
		default="cnn",
		help="Approach to evaluate, or all approaches for a comparison benchmark",
	)
	parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
	parser.add_argument("--checkpoint", type=Path)
	parser.add_argument("--manifest", type=Path)
	parser.add_argument("--image-dir", type=Path)
	parser.add_argument("--validation-samples", type=int)
	parser.add_argument("--batch-size", type=int)
	parser.add_argument("--device", choices=("cpu", "gpu"))
	parser.add_argument("--voting", action="store_true", help="Query the VLM three times per image")
	args = parser.parse_args()

	config = load_config(args.config)
	config_root = args.config.resolve().parent
	manifest = config_path(
		args.manifest or Path(config.get("manifest_root", "data/splits/split_seals")) / "val.csv",
		config_root,
	)
	image_dir = config_path(args.image_dir or Path(config.get("data_root", "data")) / "val", config_root)
	validation_samples = args.validation_samples if args.validation_samples is not None else config.get("validation_samples")
	batch_size = args.batch_size if args.batch_size is not None else config.get("batch_size", 32)
	rows = load_validation_rows(manifest, image_dir, validation_samples)
	targets = [target for _, target in rows]
	approaches = ("cnn", "digit-cnn", "classical", "vlm") if args.approach == "all" else (args.approach,)
	benchmark = []
	for approach in approaches:
		print(f"\nApproach: {approach}")
		try:
			result = run_inference(
				approach,
				[image_path for image_path, _ in rows],
				config,
				config_root,
				checkpoint=args.checkpoint if args.approach != "all" else None,
				device_name=args.device,
				batch_size=batch_size,
				voting=args.voting,
			)
			metrics = evaluate_predictions(
				targets,
				result.predictions,
				result.inference_times,
				show_examples=args.approach != "all",
			)
			benchmark.append((approach, metrics))
		except (FileNotFoundError, ValueError) as error:
			print(f"Unavailable: {error}")

	if args.approach == "all":
		print("\nBenchmark summary")
		print("Approach        Exact       Valid       Avg s/image   Images/s")
		print("-" * 65)
		for approach, metrics in benchmark:
			print(
				f"{approach:<15} {metrics['exact']:>7.2%} "
				f"{metrics['valid']:>5}/{len(targets):<5} "
				f"{metrics['average_time']:>12.4f} {metrics['throughput']:>10.2f}"
			)


if __name__ == "__main__":
	main()
