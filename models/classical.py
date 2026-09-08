from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from skimage.feature import hog
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from preprocess import CODE_LENGTH


ROOT = Path(__file__).resolve().parent.parent
DIGIT_CANVAS_SIZE = 32
DEFAULT_CLASSIFIER_PATH = ROOT / "artifacts" / "classical_digit_svm.pkl"
DEFAULT_DIGIT_DATASET = ROOT / "data" / "ground_truth_chars_balanced"
DIGIT_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


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


def extract_features(digit: np.ndarray) -> np.ndarray:
	"""Histogram-of-Oriented-Gradients features for one normalized digit."""
	normalized = normalize_digit(digit)
	return hog(
		normalized,
		orientations=9,
		pixels_per_cell=(8, 8),
		cells_per_block=(2, 2),
		feature_vector=True,
	)


def segment_digits(image_path: str | Path, code_length: int = CODE_LENGTH) -> list[np.ndarray]:
	"""Locate the row of `code_length` digits stamped on a seal photo.

	Digits are the only components that share both a similar height (same
	font/size) and appear in a horizontal row; other seal artwork (logos,
	borders, other text) reliably differs in one of those two ways.
	"""
	image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
	if image is None:
		raise FileNotFoundError(f"Could not read image: {image_path}")

	blurred = cv2.GaussianBlur(image, (5, 5), 0)
	_, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
	clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

	num_labels, _, stats, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
	image_area = image.shape[0] * image.shape[1]

	candidates = []
	for i in range(1, num_labels):
		x, y, w, h, area = stats[i]
		if not (0.0005 * image_area <= area <= 0.02 * image_area):
			continue
		if h <= w:  # digits are taller than they are wide
			continue
		candidates.append((x, y, w, h))

	if len(candidates) < code_length:
		raise ValueError(
			f"Found only {len(candidates)} digit-like components in {image_path}, need {code_length}"
		)

	# Group components with similar height; unrelated seal artwork rarely
	# matches the digits' height closely enough to land in the same group.
	candidates.sort(key=lambda c: c[3])
	best_group: list[tuple[int, int, int, int]] = []
	group = [candidates[0]]
	for c in candidates[1:]:
		if c[3] <= group[-1][3] * 1.15:
			group.append(c)
		else:
			if len(group) > len(best_group):
				best_group = group
			group = [c]
	if len(group) > len(best_group):
		best_group = group

	if len(best_group) < code_length:
		raise ValueError(
			f"Could not find a row of {code_length} similarly sized digits in {image_path}"
		)

	# If more than `code_length` boxes made it into the group, keep the ones
	# closest to the group's median height, then order them left to right.
	median_h = sorted(c[3] for c in best_group)[len(best_group) // 2]
	best_group.sort(key=lambda c: abs(c[3] - median_h))
	chosen = sorted(best_group[:code_length], key=lambda c: c[0])

	return [clean[y : y + h, x : x + w] for x, y, w, h in chosen]


class SealCodeClassical:
	"""Recognizes a seal's code without any neural network: segment the
	digits with classical image processing, describe each with HOG features,
	and classify them with a trained SVM."""

	def __init__(self, classifier_path: str | Path = DEFAULT_CLASSIFIER_PATH) -> None:
		with open(classifier_path, "rb") as handle:
			self.classifier = pickle.load(handle)

	def predict(self, image_path: str | Path) -> Optional[str]:
		try:
			digits = segment_digits(image_path)
		except ValueError:
			return None
		features = np.stack([extract_features(digit) for digit in digits])
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


def train_classifier(
	dataset_dir: str | Path = DEFAULT_DIGIT_DATASET,
	output_path: str | Path = DEFAULT_CLASSIFIER_PATH,
) -> None:
	"""Fit the digit classifier on the individually-cropped digit dataset."""
	features, labels = _load_digit_dataset(dataset_dir)
	x_train, x_val, y_train, y_val = train_test_split(
		features, labels, test_size=0.2, stratify=labels, random_state=42
	)
	classifier = SVC(kernel="rbf", C=10, gamma="scale")
	classifier.fit(x_train, y_train)
	accuracy = accuracy_score(y_val, classifier.predict(x_val))
	print(f"Validation accuracy on held-out digits: {accuracy:.3%}")

	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("wb") as handle:
		pickle.dump(classifier, handle)
	print(f"Saved digit classifier to {output_path}")


if __name__ == "__main__":
	train_classifier()
