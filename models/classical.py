from __future__ import annotations

import csv
import pickle
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from skimage.feature import hog
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from preprocess import CODE_LENGTH, letterbox


ROOT = Path(__file__).resolve().parent.parent
DIGIT_CANVAS_SIZE = 32
DEFAULT_CLASSIFIER_PATH = ROOT / "artifacts" / "classical_digit_svm.pkl"
DEFAULT_DIGIT_DATASET = ROOT / "data" / "ground_truth_chars_balanced"
DEFAULT_SEAL_MANIFEST = ROOT / "data" / "splits" / "split_seals" / "train.csv"
DEFAULT_SEAL_IMAGE_DIR = ROOT / "data" / "train"
DIGIT_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
CROPPED_IMAGE_SIZE = (256, 64)  # width, height -- matches the digit row's aspect ratio
CROP_PADDING = 12


def normalize_digit(digit: np.ndarray, size: int = DIGIT_CANVAS_SIZE) -> np.ndarray:
	"""Binarize a cropped digit and center it on a fixed square canvas.

	Matches the "Recommended Preprocessing" from the assignment: binary
	foreground/background, tight crop, aspect-ratio-preserving resize onto a
	fixed canvas, consistent polarity (digit = 1, background = 0).
	"""
	_, binary = cv2.threshold(digit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	if np.mean(binary) > 127:
		binary = cv2.bitwise_not(binary)
	ys, xs = np.where(binary > 0)
	if len(xs) == 0:
		return np.zeros((size, size), dtype=np.uint8)
	cropped = binary[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
	h, w = cropped.shape
	scale = (size - 4) / max(h, w)
	resized = cv2.resize(
		cropped,
		(max(1, round(w * scale)), max(1, round(h * scale))),
		interpolation=cv2.INTER_AREA,
	)
	canvas = np.zeros((size, size), dtype=np.uint8)
	off_y = (size - resized.shape[0]) // 2
	off_x = (size - resized.shape[1]) // 2
	canvas[off_y : off_y + resized.shape[0], off_x : off_x + resized.shape[1]] = resized
	return canvas


def _hole_features(normalized: np.ndarray) -> np.ndarray:
	"""Number of enclosed holes and their average vertical position
	(0 = top, 1 = bottom). This is the feature that most reliably tells
	apart visually similar digits that HOG alone confuses -- e.g. 0 vs 6
	vs 8 vs 9 differ mainly in how many holes they have and where (see the
	assignment's "Topological Features" section).
	"""
	contours, hierarchy = cv2.findContours(normalized, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
	if hierarchy is None:
		return np.array([0.0, 0.5], dtype=np.float32)
	holes = [
		contours[i]
		for i, h in enumerate(hierarchy[0])
		if h[3] != -1 and cv2.contourArea(contours[i]) > 4
	]
	if not holes:
		return np.array([0.0, 0.5], dtype=np.float32)
	moments = [cv2.moments(c) for c in holes]
	ys = [m["m01"] / m["m00"] for m in moments if m["m00"] > 0]
	hole_y = (sum(ys) / len(ys) / normalized.shape[0]) if ys else 0.5
	return np.array([len(holes), hole_y], dtype=np.float32)


def extract_features(digit: np.ndarray) -> np.ndarray:
	"""HOG shape descriptor combined with hole-count/position features."""
	normalized = normalize_digit(digit)
	hog_features = hog(
		normalized,
		orientations=9,
		pixels_per_cell=(8, 8),
		cells_per_block=(2, 2),
		feature_vector=True,
	)
	return np.concatenate([hog_features, _hole_features(normalized)])


def _threshold_candidates(
	image: np.ndarray, threshold_type: int, adaptive_block: int | None = None, adaptive_c: int = 5
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
	if adaptive_block is None:
		blurred = cv2.GaussianBlur(image, (5, 5), 0)
		_, binary = cv2.threshold(blurred, 0, 255, threshold_type + cv2.THRESH_OTSU)
	else:
		binary = cv2.adaptiveThreshold(
			image,
			255,
			cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
			threshold_type,
			adaptive_block,
			adaptive_c,
		)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
	clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

	num_labels, _, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
	image_area = image.shape[0] * image.shape[1]
	candidates = []
	for i in range(1, num_labels):
		x, y, w, h, area = stats[i]
		if not (30 <= area <= 0.05 * image_area):
			continue
		if h <= w or h > 8 * w:
			continue
		candidates.append((x, y, w, h))
	return binary, candidates


def _select_digit_row(
	candidates: list[tuple[int, int, int, int]], code_length: int
) -> list[tuple[int, int, int, int]]:
	"""Select similarly sized, vertically aligned components as one digit row."""
	best_group: list[tuple[int, int, int, int]] = []
	best_key = (-1, float("-inf"), float("-inf"))
	for seed in candidates:
		seed_height = seed[3]
		height_group = [candidate for candidate in candidates if 1 / 1.25 <= candidate[3] / seed_height <= 1.25]
		median_height = float(np.median([candidate[3] for candidate in height_group]))
		y_centers = [candidate[1] + candidate[3] / 2 for candidate in height_group]
		median_y = float(np.median(y_centers))
		row = [
			candidate
			for candidate in height_group
			if abs(candidate[1] + candidate[3] / 2 - median_y) <= 0.35 * median_height
		]
		if len(row) < code_length:
			continue
		y_spread = max(candidate[1] + candidate[3] / 2 for candidate in row) - min(
			candidate[1] + candidate[3] / 2 for candidate in row
		)
		height_spread = max(candidate[3] for candidate in row) - min(candidate[3] for candidate in row)
		key = (len(row), -y_spread, -height_spread)
		if key > best_key:
			best_group = row
			best_key = key

	if len(best_group) < code_length:
		raise ValueError(f"Could not find a row of {code_length} similarly sized digits")

	median_height = float(np.median([candidate[3] for candidate in best_group]))
	best_group.sort(key=lambda candidate: abs(candidate[3] - median_height))
	return sorted(best_group[:code_length], key=lambda candidate: candidate[0])


def find_digit_boxes(
	image_path: str | Path, code_length: int = CODE_LENGTH
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
	"""Locate the row of `code_length` digits stamped on a seal photo.

	Digits are the only components that share both a similar height (same
	font/size) and appear in a horizontal row; other seal artwork (logos,
	borders, other text) reliably differs in one of those two ways. Both
	normal and inverse threshold polarity are tried, together with an
	adaptive-threshold fallback for uneven illumination.

	Returns the pre-morphology binary mask and each digit's (x, y, w, h)
	box, sorted left to right.
	"""
	image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
	if image is None:
		raise FileNotFoundError(f"Could not read image: {image_path}")

	options = []
	thresholds = [
		(threshold_type, None, 5)
		for threshold_type in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV)
	]
	thresholds.extend(
		(threshold_type, 81, 7)
		for threshold_type in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV)
	)
	for threshold_type, adaptive_block, adaptive_c in thresholds:
		binary, candidates = _threshold_candidates(image, threshold_type, adaptive_block, adaptive_c)
		try:
			chosen = _select_digit_row(candidates, code_length)
			median_height = float(np.median([h for x, y, w, h in chosen]))
			y_spread = max(y + h / 2 for x, y, w, h in chosen) - min(
				y + h / 2 for x, y, w, h in chosen
			)
			options.append((len(chosen), median_height, -y_spread, binary, chosen))
		except ValueError:
			continue

	if not options:
		raise ValueError(f"Could not find a row of {code_length} similarly sized digits in {image_path}")

	_, _, _, binary, chosen = max(options, key=lambda option: option[:3])
	return binary, chosen


def segment_digits(image_path: str | Path, code_length: int = CODE_LENGTH) -> list[np.ndarray]:
	"""Crop each of the `code_length` digits located by `find_digit_boxes`.

	Crops from the pre-morphology mask: OPEN/CLOSE are only needed to get
	clean, well-separated components for detection, but at small digit
	scales they can erode thin strokes or seal shut a "0"/"6"'s hole --
	exactly the detail that tells digits apart.
	"""
	binary, chosen = find_digit_boxes(image_path, code_length)
	return [binary[y : y + h, x : x + w] for x, y, w, h in chosen]


def load_cropped_image(
	image_path: str | Path,
	size: tuple[int, int] = CROPPED_IMAGE_SIZE,
	padding: int = CROP_PADDING,
) -> torch.Tensor:
	"""Load a seal photo cropped tightly around its digit row.

	Feeding a neural network the small region that actually contains the
	code -- instead of the whole photo shrunk down -- lets it work with far
	more effective resolution on the part that matters. Falls back to the
	full, uncropped frame when the digit row can't be located, so this
	always returns a usable, fixed-size tensor.
	"""
	image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
	if image is None:
		raise FileNotFoundError(f"Could not read image: {image_path}")

	try:
		_, boxes = find_digit_boxes(image_path)
	except ValueError:
		boxes = None

	if boxes:
		height, width = image.shape
		x0 = max(0, min(x for x, y, w, h in boxes) - padding)
		y0 = max(0, min(y for x, y, w, h in boxes) - padding)
		x1 = min(width, max(x + w for x, y, w, h in boxes) + padding)
		y1 = min(height, max(y + h for x, y, w, h in boxes) + padding)
		image = image[y0:y1, x0:x1]

	resized = letterbox(image, size)
	return torch.from_numpy(resized.astype(np.float32) / 255.0).unsqueeze(0)


class SealCodeClassical:
	"""Recognizes a seal's code without any neural network: segment the
	digits with classical image processing, describe each with HOG features,
	and classify them with a trained SVM."""

	def __init__(self, classifier_path: str | Path = DEFAULT_CLASSIFIER_PATH) -> None:
		with open(classifier_path, "rb") as handle:
			saved = pickle.load(handle)
		self.scaler = saved["scaler"]
		self.classifier = saved["classifier"]

	def predict(self, image_path: str | Path) -> Optional[str]:
		try:
			digits = segment_digits(image_path)
		except ValueError:
			return None
		features = np.stack([extract_features(digit) for digit in digits])
		features = self.scaler.transform(features)
		predictions = self.classifier.predict(features)
		return "".join(str(label) for label in predictions)


def _load_digit_dataset(dataset_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
	features, labels = [], []
	paths = [p for p in sorted(Path(dataset_dir).iterdir()) if p.suffix.lower() in DIGIT_EXTENSIONS]
	for i, path in enumerate(paths, start=1):
		label = path.stem.rsplit("_", 1)[1]
		image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
		features.append(extract_features(image))
		labels.append(label)
		if i % 200 == 0 or i == len(paths):
			print(f"Loaded {i}/{len(paths)} digit images")
	return np.stack(features), np.array(labels)


def _load_manifest_digit_dataset(
	manifest_path: str | Path,
	image_dir: str | Path,
	code_length: int = CODE_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
	"""Self-label digits by segmenting seal photos with a known ground-truth
	code, so the classifier also trains on digits that look like the ones it
	will actually see at inference time (not just the clean reference set).
	"""
	with open(manifest_path, newline="", encoding="utf-8") as handle:
		rows = list(csv.DictReader(handle, delimiter=";"))

	features, labels = [], []
	skipped = 0
	for i, row in enumerate(rows, start=1):
		image_path = Path(image_dir) / row["filename"]
		code = row["number"].strip().zfill(code_length)
		if image_path.is_file():
			try:
				digits = segment_digits(image_path, code_length)
			except ValueError:
				digits = None
		else:
			digits = None
		if digits is None:
			skipped += 1
		else:
			for digit_image, label in zip(digits, code):
				features.append(extract_features(digit_image))
				labels.append(label)
		if i % 1000 == 0 or i == len(rows):
			print(f"Segmented {i}/{len(rows)} seal photos ({skipped} skipped so far)")
	return np.stack(features), np.array(labels)


def train_classifier(
	dataset_dir: str | Path = DEFAULT_DIGIT_DATASET,
	seal_manifest: str | Path = DEFAULT_SEAL_MANIFEST,
	seal_image_dir: str | Path = DEFAULT_SEAL_IMAGE_DIR,
	output_path: str | Path = DEFAULT_CLASSIFIER_PATH,
) -> None:
	"""Fit the digit classifier on clean reference digits plus digits
	self-labeled by segmenting real seal photos with a known code."""
	clean_features, clean_labels = _load_digit_dataset(dataset_dir)
	print(f"Loaded {len(clean_labels)} clean reference digits from {dataset_dir}")

	seal_features, seal_labels = _load_manifest_digit_dataset(seal_manifest, seal_image_dir)
	print(f"Segmented {len(seal_labels)} digits from real seal photos in {seal_manifest}")

	features = np.concatenate([clean_features, seal_features])
	labels = np.concatenate([clean_labels, seal_labels])

	x_train, x_val, y_train, y_val = train_test_split(
		features, labels, test_size=0.2, stratify=labels, random_state=42
	)

	# Scaling so the 2-dim hole features aren't swamped by the 324-dim HOG
	# vector; kept even though a quick test showed it barely changes RBF's
	# accuracy, since it doesn't hurt and is good practice regardless.
	scaler = StandardScaler()
	x_train = scaler.fit_transform(x_train)
	x_val = scaler.transform(x_val)

	# A quick side-by-side test (800 photos) showed RBF clearly beating a
	# linear kernel here (45% vs. 40%), despite the assignment suggesting
	# linear for HOG -- so we keep RBF.
	classifier = SVC(kernel="rbf", C=10, gamma="scale")
	classifier.fit(x_train, y_train)
	accuracy = accuracy_score(y_val, classifier.predict(x_val))
	print(f"Validation accuracy on held-out digits: {accuracy:.3%}")

	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("wb") as handle:
		pickle.dump({"scaler": scaler, "classifier": classifier}, handle)
	print(f"Saved digit classifier to {output_path}")


if __name__ == "__main__":
	train_classifier()
