from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch

from digit_pipeline import digit_crops_from_seal
from models.cnn import SealCodeCNN
from models.digit_cnn import DigitCNN
from pipeline_config import config_path, resolve_device
from preprocess import CODE_LENGTH, decode_code, load_image


@dataclass
class PredictionResult:
    predictions: list[str | None]
    inference_times: list[float]


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    return torch.load(path, map_location=device, weights_only=True)


def _progress(approach: str, processed: int, total: int, started: float) -> None:
    elapsed = time.perf_counter() - started
    average = elapsed / processed if processed else 0.0
    print(
        f"{approach}: processed {processed}/{total} images "
        f"({average:.3f} s/image, {elapsed:.1f} s elapsed)",
        flush=True,
    )


def _predict_cnn(
    images: list[Path], checkpoint: Path, batch_size: int, device: torch.device
) -> PredictionResult:
    checkpoint_data = _load_checkpoint(checkpoint, device)
    model = SealCodeCNN(checkpoint_data.get("code_length", CODE_LENGTH))
    model.load_state_dict(checkpoint_data["model"])
    model.to(device).eval()
    predictions: list[str] = []
    inference_times: list[float] = []
    started_total = time.perf_counter()
    progress_step = max(1, len(images) // 10)
    with torch.no_grad():
        for start_index in range(0, len(images), batch_size):
            batch_paths = images[start_index : start_index + batch_size]
            started = time.perf_counter()
            batch = torch.stack([load_image(path) for path in batch_paths]).to(device)
            _synchronize(device)
            logits = model(batch)
            _synchronize(device)
            elapsed = (time.perf_counter() - started) / len(batch_paths)
            predictions.extend(decode_code(prediction) for prediction in logits.argmax(dim=-1))
            inference_times.extend([elapsed] * len(batch_paths))
            processed = start_index + len(batch_paths)
            if processed % progress_step == 0 or processed == len(images):
                _progress("cnn", processed, len(images), started_total)
    return PredictionResult(predictions, inference_times)


def _predict_digit_cnn(
    images: list[Path], checkpoint: Path, device: torch.device
) -> PredictionResult:
    checkpoint_data = _load_checkpoint(checkpoint, device)
    model = DigitCNN().to(device)
    model.load_state_dict(checkpoint_data["model"])
    model.eval()
    predictions: list[str | None] = []
    inference_times: list[float] = []
    started_total = time.perf_counter()
    progress_step = max(1, len(images) // 10)
    for image_path in images:
        started = time.perf_counter()
        try:
            crops = digit_crops_from_seal(image_path, CODE_LENGTH)
            _synchronize(device)
            with torch.no_grad():
                digits = model(torch.stack(crops).to(device)).argmax(dim=1)
            _synchronize(device)
            predictions.append(decode_code(digits))
        except (FileNotFoundError, ValueError) as error:
            print(f"Warning: could not predict {image_path.name}: {error}")
            predictions.append(None)
        inference_times.append(time.perf_counter() - started)
        processed = len(predictions)
        if processed % progress_step == 0 or processed == len(images):
            _progress("digit-cnn", processed, len(images), started_total)
    return PredictionResult(predictions, inference_times)


def _predict_classical(images: list[Path], checkpoint: Path) -> PredictionResult:
    from models.classical import SealCodeClassical

    model = SealCodeClassical(checkpoint)
    predictions: list[str | None] = []
    inference_times: list[float] = []
    started_total = time.perf_counter()
    progress_step = max(1, len(images) // 10)
    for image_path in images:
        started = time.perf_counter()
        try:
            predictions.append(model.predict(image_path))
        except (FileNotFoundError, ValueError) as error:
            print(f"Warning: could not predict {image_path.name}: {error}")
            predictions.append(None)
        inference_times.append(time.perf_counter() - started)
        processed = len(predictions)
        if processed % progress_step == 0 or processed == len(images):
            _progress("classical", processed, len(images), started_total)
    return PredictionResult(predictions, inference_times)


def _predict_vlm(images: list[Path], voting: bool) -> PredictionResult:
    from models.vlm import SealCodeVLM

    model = SealCodeVLM(openai=False)
    predictions: list[str | None] = []
    inference_times: list[float] = []
    started_total = time.perf_counter()
    progress_step = max(1, len(images) // 10)
    for image_path in images:
        started = time.perf_counter()
        try:
            predictions.append(model.predict_with_voting(image_path) if voting else model.predict(image_path))
        except Exception as error:
            print(f"Warning: VLM failed for {image_path.name}: {error}")
            predictions.append(None)
        inference_times.append(time.perf_counter() - started)
        processed = len(predictions)
        if processed % progress_step == 0 or processed == len(images):
            _progress("vlm", processed, len(images), started_total)
    return PredictionResult(predictions, inference_times)


def run_inference(
    approach: str,
    images: list[Path],
    config: dict,
    config_root: Path,
    checkpoint: Path | None = None,
    device_name: str | None = None,
    batch_size: int = 32,
    voting: bool = False,
) -> PredictionResult:
    device = resolve_device(device_name or config.get("device", "gpu"))
    if approach == "cnn":
        checkpoint_path = config_path(
            checkpoint or config.get("checkpoint", "artifacts/seal_code_cnn.pt"), config_root
        )
        return _predict_cnn(images, checkpoint_path, batch_size, device)
    if approach == "digit-cnn":
        checkpoint_path = config_path(
            checkpoint or config.get("digit_checkpoint", "artifacts/digit_cnn.pt"), config_root
        )
        digit_result = _predict_digit_cnn(images, checkpoint_path, device)
        failed_indexes = [index for index, prediction in enumerate(digit_result.predictions) if prediction is None]
        if not failed_indexes:
            return digit_result

        cnn_checkpoint_path = config_path(
            checkpoint or config.get("checkpoint", "artifacts/seal_code_cnn.pt"),
            config_root,
        )
        fallback_images = [images[index] for index in failed_indexes]
        fallback_result = _predict_cnn(fallback_images, cnn_checkpoint_path, batch_size, device)
        merged_predictions = list(digit_result.predictions)
        merged_times = list(digit_result.inference_times)
        for failed_index, fallback_prediction in zip(failed_indexes, fallback_result.predictions):
            merged_predictions[failed_index] = fallback_prediction
        for failed_index, fallback_time in zip(failed_indexes, fallback_result.inference_times):
            merged_times[failed_index] = fallback_time
        print(
            f"digit-cnn fallback: used whole-image CNN for {len(failed_indexes)}/{len(images)} images "
            f"after digit segmentation or digit-CNN failure",
            flush=True,
        )
        return PredictionResult(merged_predictions, merged_times)
    if approach == "classical":
        checkpoint_path = config_path(
            checkpoint or config.get("classical_checkpoint", "artifacts/classical_digit_svm.pkl"),
            config_root,
        )
        return _predict_classical(images, checkpoint_path)
    if approach == "vlm":
        return _predict_vlm(images, voting)
    raise ValueError(f"Unknown approach: {approach}")
