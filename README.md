# AI Summer School 2026

Baseline implementation for numerical code recognition on industrial seal
images. The current repository contains a small PyTorch CNN pipeline that
predicts the seven digits of a seal code.

## Current Scope

The implemented pipeline includes:

- grayscale image loading and aspect-ratio-preserving resize;
- a compact CNN with one classification output for each code digit;
- training on the train split;
- validation after every epoch;
- checkpointing by best validation exact-code accuracy;
- validation evaluation and directory-based inference.

The pipeline currently uses only the train and validation splits. Classical
computer vision and multimodal LLM approaches are not implemented yet.

## Requirements

- Python 3.10 or newer
- A CUDA-capable GPU is recommended but not required
- The full image dataset is large; a partial download is supported for smoke
	runs

The required Python packages are listed in [requirements.txt](requirements.txt).
Training and validation defaults are stored in [config.json](config.json).

## Installation

From the repository root, create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On a machine that needs a specific CUDA build of PyTorch, install the matching
PyTorch, TorchVision, and TorchAudio wheels from the official PyTorch index
before installing the remaining packages.

## Dataset Layout

The baseline expects this structure:

```text
data/
	train/
		00000.png
		...
	val/
		00000.png
		...
	splits/
		split_seals/
			train.csv
			val.csv
```

Each CSV uses a semicolon delimiter and has this format:

```text
filename;number
00000.png;1585678
```

The `filename` in each manifest must correspond to an image in the matching
`data/train` or `data/val` directory.

If only part of the dataset has been downloaded, manifest entries whose image
files are missing are skipped with a warning. This allows a smoke run, but the
resulting accuracy is not representative. Add the remaining images to obtain
meaningful results.

## Training

Edit `config.json` to choose how many locally available samples to use:

```json
{
	"train_samples": 10,
	"validation_samples": 5,
	"epochs": 10,
	"batch_size": 32,
	"learning_rate": 0.001,
	"device": "gpu"
}
```

Set either value to `null` to use every available image in that split. The
limits are applied after missing manifest images are skipped. When the full
dataset is downloaded, setting both values to `null` uses all samples.

Activate the virtual environment, then run a short smoke training run:

```bash
python train.py --config config.json --epochs 1
```

Run a normal baseline training session with:

```bash
python train.py --config config.json
```

The `device` setting controls execution. Use `"gpu"` to use CUDA when it is
available, or `"cpu"` to force CPU execution. If `"gpu"` is configured but
CUDA is unavailable, the pipeline warns and falls back to CPU. The script
prints training and validation loss, per-digit accuracy, and exact-code
accuracy after each epoch. The checkpoint is saved whenever validation
exact-code accuracy improves.

Command-line options such as `--train-samples`, `--validation-samples`,
`--epochs`, and `--batch-size` override values from `config.json` for a single
run.

Useful training options:

```text
--data-root       Root containing train/ and val/
--manifest-root   Directory containing train.csv and val.csv
--output          Checkpoint path
--epochs          Number of training epochs
--batch-size      Number of images per batch
--learning-rate   Adam learning rate
--device          Execution device: cpu or gpu
```

For help:

```bash
python train.py --help
```

## Evaluation

Evaluate a saved checkpoint on the available validation images:

```bash
python evaluation.py --config config.json
```

The evaluator reports:

- `Digit accuracy`: percentage of correctly predicted individual digits;
- `Exact-code accuracy`: percentage of samples where all seven digits are
	correct;
- several example errors.

Custom validation paths can be supplied with `--manifest` and `--image-dir`.

## Inference

Run the trained model on every PNG in an input directory:

```bash
python main.py \
	--input-dir data/val \
	--output-dir artifacts \
	--checkpoint artifacts/seal_code_cnn.pt
```

This creates `artifacts/predictions.csv` with semicolon-separated columns:

```text
filename;number
00000.png;1585844
```

Use `--output-name` to choose a different result filename:

```bash
python main.py \
	--input-dir data/val \
	--output-dir artifacts \
	--checkpoint artifacts/seal_code_cnn.pt \
	--output-name team1.csv
```

For help on any entry point:

```bash
python train.py --help
python evaluation.py --help
python main.py --help
```

## Troubleshooting

If training reports missing image files, compare the manifests with the local
folders. The loader will continue with the files that are present. If a split
has no available images, download at least one image for that split before
running training or evaluation.

If `torch.cuda.is_available()` is false, training still works on CPU but will
be slower. Confirm that the virtual environment is active and that the local
PyTorch installation matches the available CUDA driver when GPU execution is
expected.

The repository currently has no automated test suite. The practical smoke
check is a one-epoch training run followed by evaluation.

## Project Files

```text
preprocess.py       Image preprocessing and code encoding
data_pipeline.py    Manifest-backed PyTorch dataset
models/cnn.py       Baseline CNN model
train.py            Training and validation loop
evaluation.py       Validation metrics and example errors
main.py             Batch inference and CSV generation
requirements.txt    Python dependencies
```