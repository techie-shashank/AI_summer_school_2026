# AI_Summer_School_2026
Repository for Bavarian-Czech Summer School “AI and Industry” 2026

## Baseline CNN pipeline

The first baseline uses the train and validation seal splits only. Images are
converted to grayscale, letterboxed to 256x160, and passed to a small CNN that
predicts the seven digits independently. The best checkpoint is selected by
validation exact-code accuracy.

The expected local layout is:

```text
data/train/*.png
data/val/*.png
data/splits/split_seals/train.csv
data/splits/split_seals/val.csv
```

Train and save a checkpoint with:

```bash
python3 train.py --epochs 10 --output artifacts/seal_code_cnn.pt
```

Evaluate the validation split with:

```bash
python3 evaluation.py --checkpoint artifacts/seal_code_cnn.pt
```

Run inference on a directory of PNG files with:

```bash
python3 main.py --input-dir data/val --output-dir artifacts \
	--checkpoint artifacts/seal_code_cnn.pt
```

The current workspace must contain all images named by the manifests before
training; the CSV files reference more images than the checked-out image subset.

### Notes

#### Image to Base64 

```Python
image = cv2.imread(IMAGE_PATH)
_, buffer = cv2.imencode('.png', image)
img_base64 = base64.b64encode(buffer).decode('utf-8')
```

#### UV Python package manager instalation

https://docs.astral.sh/uv/getting-started/installation/

#### Competition laptop HW specs

- **CPU:** Intel Core i7-9750 2.6 GHz x 12
- **RAM:** 24 GB
- **GPU:** NVIDIA GeForce RTX 2060 Mobile. 6 GB VRAM 