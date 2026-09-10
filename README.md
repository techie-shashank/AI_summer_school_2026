# AI Summer School 2026 — Team PinkOps

Numerical code recognition on industrial seal images. The repository
implements and compares three independent approaches to reading the
7-digit code stamped on a seal: a PyTorch CNN, a classical (non-neural)
image-processing + SVM pipeline, and a vision-language model (VLM) queried
through prompting.

## Current Scope

All three approaches described in the assignment are implemented:

- **CNN** ([models/cnn.py](models/cnn.py), trained via [train.py](train.py)): a
	compact convolutional network trained end-to-end on the full seal photo,
	predicting all 7 digits at once. Trained on the full train split with
	augmentation (rotation, translation, scale, brightness/contrast, noise).
- **VLM** ([models/vlm.py](models/vlm.py)): asks a pretrained vision-language
	model (the KKY Ollama server, or OpenAI) to read the code directly, using
	a structured response format, a plausibility check (exactly 7 digits),
	and an optional majority vote across repeated queries.
- **Classical** ([models/classical.py](models/classical.py)): locates the 7
	digits with normal/inverse Otsu and adaptive thresholding, then uses
	connected-component analysis grouped by matching height, describes each digit with HOG features, and classifies
	them with an SVM trained on both a clean reference digit set
	(`data/ground_truth_chars_balanced`) and digits self-labeled by
	segmenting real seal photos with a known code.
- **Segmented-digit CNN** ([models/digit_cnn.py](models/digit_cnn.py), trained
	via [train_digit.py](train_digit.py)): segments the seven digits from a seal,
	classifies each crop with one reusable ten-class CNN, and joins the predictions
	from left to right.

An experimental fourth variant — the same CNN architecture trained on a crop
of just the digit row instead of the whole photo (`train.py
--crop-to-digits`) — is still being evaluated against the whole-image CNN.

### Current results (validation set, `data/val`)

| Approach | Exact-code accuracy | Per-digit accuracy | Speed / image |
|---|---|---|---|
| CNN | ~84% (still improving) | ~97% | ~36 ms |
| VLM (single query) | 91% | – | ~2.4 s |
| VLM (with voting, 3×) | 99% | – | ~6.9 s |
| Classical | ~10% | ~31% | ~26 ms |

CNN and Classical run entirely locally with no network dependency; VLM
requires network access to the configured backend. These numbers come from
`data/val`, which was also used to pick the best CNN checkpoint — a final,
unbiased comparison should use the untouched `data/splits/split_seals/test.csv`
split instead.

## Requirements

- Python 3.12 (developed against 3.12.7; 3.10+ should work)
- A CUDA-capable GPU is recommended for CNN training but not required
	(falls back to Apple MPS, then CPU)
- The full image dataset is large; a partial download is supported for smoke
	runs
- The VLM approach needs network access and credentials (see
	[VLM Setup](#vlm-setup))

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
	test/                          # held out, used only for final evaluation
		00000.png
		...
	ground_truth_chars_balanced/   # individual, pre-cropped digit images
		102423_7.tif                # filename encodes the label (here: 7)
		...
	splits/
		split_seals/
			train.csv
			val.csv
			test.csv
```

Each seal-level CSV uses a semicolon delimiter and has this format:

```text
filename;number
00000.png;1585678
```

The `filename` in each manifest must correspond to an image in the matching
`data/train`, `data/val`, or `data/test` directory.

If only part of the dataset has been downloaded, manifest entries whose image
files are missing are skipped with a warning. This allows a smoke run, but the
resulting accuracy is not representative. Add the remaining images to obtain
meaningful results.

## Training the CNN

Edit `config.json` to choose how many locally available samples to use:

```json
{
	"train_samples": null,
	"validation_samples": null,
	"epochs": 50,
	"batch_size": 32,
	"learning_rate": 0.001,
	"device": "gpu"
}
```

Set either sample value to `null` to use every available image in that split
(the current default). The limits are applied after missing manifest images
are skipped.

Activate the virtual environment, then run a short smoke training run:

```bash
python train.py --config config.json --epochs 1
```

Run a full training session with:

```bash
python train.py --config config.json
```

The `device` setting controls execution: `"gpu"` uses CUDA if available, then
Apple MPS, then falls back to CPU with a warning; `"cpu"` forces CPU. The
script prints training and validation loss, per-digit accuracy, and
exact-code accuracy after each epoch. The checkpoint is saved whenever
validation exact-code accuracy improves.

Command-line options such as `--train-samples`, `--validation-samples`,
`--epochs`, and `--batch-size` override values from `config.json` for a single
run.

Useful training options:

```text
--data-root         Root containing train/ and val/
--manifest-root     Directory containing train.csv and val.csv
--output            Checkpoint path
--epochs            Number of training epochs
--batch-size        Number of images per batch
--learning-rate     Adam learning rate
--device            Execution device: cpu or gpu
--resume            Continue training from the existing checkpoint at --output,
                    instead of starting over from random weights
--crop-to-digits    Train on the digit row (via the classical approach's
                    segmentation) instead of the full photo; saves to
                    artifacts/seal_code_cnn_cropped.pt by default
```

For help:

```bash
python train.py --help
```

## Training the Classical Approach

The classical approach's digit classifier is trained by running its module
directly:

```bash
python models/classical.py
```

This loads the clean reference digits from `ground_truth_chars_balanced`,
segments all seal photos in the training split to self-label additional
digit crops, fits an SVM, and saves it to
`artifacts/classical_digit_svm.pkl`.

## Training the Segmented-Digit CNN

This alternative CNN is trained by segmenting labeled seal photos from
`data/train`, then pairing each crop with the corresponding digit from the seal
manifest. This matches the images the model receives during inference. The
clean individually labeled images in `data/ground_truth_chars_balanced` remain
available as an optional training source.

Train it with:

```bash
python train_digit.py --config config.json
```

The relevant configuration keys are:

```json
{
	"digit_dataset_root": "data/ground_truth_chars_balanced",
	"digit_checkpoint": "artifacts/digit_cnn.pt",
	"digit_source": "seal",
	"digit_samples": null,
	"digit_epochs": 10
}
```

Set `digit_samples` to an integer for a quick smoke run or `null` to use all
available source images. Set `digit_source` to `clean` for the reference digit
dataset or `mixed` to combine both sources. The digit training command creates
an internal, deterministic 80/20 crop split and saves the checkpoint with the
best validation digit accuracy.

## VLM Setup

The VLM approach needs credentials for either the KKY Ollama server or
OpenAI. Copy the template and fill in your own values:

```bash
cp .env.example .env
```

```text
KKY_OLLAMA_UNAME=...
KKY_OLLAMA_PASSWORD=...
KKY_OLLAMA_SERVER=https://ollama.kky.zcu.cz
OPENAI_API_KEY=...
```

`.env` is listed in `.gitignore` and is never committed. `models/vlm.py`
loads it automatically via `python-dotenv`.

## Evaluation

`evaluation.py` supports every recognition approach through `--approach`.
All approaches are compared against the ground-truth codes in the validation
manifest and report valid predictions, exact-code accuracy, and inference-time
statistics.

Benchmark all approaches in one run:

```bash
python evaluation.py --approach all --config config.json
```

This prints a comparison table containing exact-code accuracy, valid
predictions, average latency, and throughput. Missing checkpoints are reported
as unavailable and do not stop the other approaches from being benchmarked.
While each approach runs, progress is printed approximately every 10% with the
number of processed images, elapsed time, and current average time per image.

Evaluate the whole-image CNN:

```bash
python evaluation.py --approach cnn --config config.json
```

Evaluate the segmented-digit CNN:

```bash
python evaluation.py --approach digit-cnn --config config.json
```

Evaluate the classical SVM approach after creating its classifier checkpoint:

```bash
python evaluation.py --approach classical --config config.json
```

Evaluate the VLM approach using the configured Ollama backend:

```bash
python evaluation.py --approach vlm --config config.json
```

Add `--voting` to query the VLM three times per image. VLM evaluation requires
network access and the credentials described in [VLM Setup](#vlm-setup).

The evaluator reports:

- `Exact-code accuracy`: percentage of samples where all seven digits are
	correct;
- total, average, minimum, and maximum inference time;
- inference throughput in images per second;
- several example errors.

Custom validation paths can be supplied with `--manifest` and `--image-dir`.
Use `--validation-samples` to limit one evaluation run without editing the
configuration file. Use `--device cpu` to force CPU inference.

## Inference

Run a recognition approach on every PNG in an input directory:

```bash
python main.py \
	--input-dir data/val \
	--output-dir artifacts \
	--approach cnn
```

`--approach` selects `cnn` (default), `digit-cnn`, or `vlm`. The `digit-cnn`
approach requires a trained `digit_cnn.pt` checkpoint and uses classical
segmentation before classification. For the VLM approach, add
`--voting` to query the model three times per image and keep the majority
answer (slower, more accurate).

Run the segmented-digit CNN with:

```bash
python main.py \
	--approach digit-cnn \
	--input-dir data/val \
	--output-dir artifacts \
	--config config.json
```

If segmentation cannot find seven digits, the image receives an empty code in
the output CSV and a warning is printed.

This creates `artifacts/PinkOps.csv` with semicolon-separated columns:

```text
filename;number
00000.png;1585844
```

Use `--output-name` to choose a different result filename. For help on any
entry point:

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

If neither CUDA nor MPS is available, training still works on CPU but will
be slower. Confirm that the virtual environment is active and that the local
PyTorch installation matches the available CUDA driver when GPU execution is
expected.

The repository currently has no automated test suite. The practical smoke
check is a one-epoch training run followed by evaluation.

## Project Files

```text
preprocess.py           Image preprocessing and code encoding
data_pipeline.py        Manifest-backed PyTorch dataset (whole-image or cropped)
models/cnn.py           Baseline CNN model
models/digit_cnn.py    Single-digit CNN model
models/vlm.py           Vision-language-model recognition
models/classical.py     Classical image-processing + SVM recognition
llm/common_llm.py        OpenAI / KKY Ollama API helpers used by models/vlm.py
inference.py            Shared prediction and timing logic for all approaches
train.py                Training and validation loop for the CNN
train_digit.py          Training loop for the single-digit CNN
digit_pipeline.py       Digit dataset and segmentation-crop utilities
evaluation.py           Validation metrics and example errors
main.py                 Batch inference and CSV generation
requirements.txt        Python dependencies
.env.example             Template for VLM credentials (copy to .env)
```
