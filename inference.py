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

import torchvision.transforms.functional as TF


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


@torch.no_grad()
def _cnn_predictions_with_confidence(
    model: torch.nn.Module,
    batch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        predictions: [B, 7]
        confidence:  [B, 7] - probability of predicted digit
        margin:      [B, 7] - difference between top-1 and top-2 probabilities
    """
    logits = model(batch)
    probabilities = torch.softmax(logits, dim=-1)

    top2 = torch.topk(probabilities, k=2, dim=-1)

    predictions = top2.indices[..., 0]
    confidence = top2.values[..., 0]
    margin = top2.values[..., 0] - top2.values[..., 1]

    return predictions, confidence, margin


def _needs_vlm(
    predictions: torch.Tensor,
    confidence: torch.Tensor,
    margin: torch.Tensor,
) -> torch.Tensor:
    """
    Returns a boolean tensor [B] indicating which images should be
    checked by the VLM.
    """

    # Any digit with very low confidence.
    low_confidence = (confidence < 0.93).any(dim=1)

    # Any digit where the top prediction is very close to the runner-up.
    low_margin = (margin < 0.15).any(dim=1)

    return low_confidence | low_margin


def _predict_cnn(
    images: list[Path],
    checkpoint: Path,
    batch_size: int,
    device: torch.device,
    use_vlm: bool = True,
) -> PredictionResult:
    checkpoint_data = _load_checkpoint(checkpoint, device)

    model = SealCodeCNN(
        checkpoint_data.get("code_length", CODE_LENGTH)
    )
    model.load_state_dict(checkpoint_data["model"])
    model.to(device).eval()

    vlm = None
    if use_vlm:
        from models.vlm import SealCodeVLM
        vlm = SealCodeVLM(openai=False)

    predictions: list[str | None] = []
    inference_times: list[float] = []

    started_total = time.perf_counter()
    progress_step = max(1, len(images) // 10)

    vlm_count = 0

    with torch.no_grad():
        for start_index in range(0, len(images), batch_size):
            batch_paths = images[start_index : start_index + batch_size]

            started = time.perf_counter()

            batch = torch.stack(
                [load_image(path) for path in batch_paths]
            ).to(device)

            _synchronize(device)

            # CNN prediction + confidence
            cnn_digits, confidence, margin = (
                _cnn_predictions_with_confidence(
                    model,
                    batch,
                )
            )

            _synchronize(device)

            batch_predictions = [
                decode_code(digits)
                for digits in cnn_digits
            ]

            if use_vlm:

                # Which images look suspicious?
                suspicious = _needs_vlm(
                    cnn_digits,
                    confidence,
                    margin,
                )

                # VLM fallback for suspicious images
                for local_index, is_suspicious in enumerate(
                    suspicious.tolist()
                ):

                    image_path = batch_paths[local_index]

                    print(
                        f"CNN decision: {image_path.name} | "
                        f"prediction={batch_predictions[local_index]} | "
                        f"confidence={confidence[local_index].min().item():.2f} | "
                        f"margin={margin[local_index].min().item():.2f} | "
                        f"VLM={'YES' if is_suspicious else 'NO'}",
                        flush=True,
                    )
                    
                    if not is_suspicious:
                        continue

                    image_path = batch_paths[local_index]

                    try:
                        vlm_prediction = vlm.predict(image_path)

                        if vlm_prediction is not None:
                            batch_predictions[local_index] = vlm_prediction
                            vlm_count += 1

                    except Exception as error:
                        print(
                            f"Warning: VLM failed for "
                            f"{image_path.name}: {error}"
                        )

            _synchronize(device)

            elapsed = (
                time.perf_counter() - started
            ) / len(batch_paths)

            predictions.extend(batch_predictions)
            inference_times.extend(
                [elapsed] * len(batch_paths)
            )

            processed = start_index + len(batch_paths)

            if (
                processed % progress_step == 0
                or processed == len(images)
            ):
                _progress(
                    "cnn+vlm",
                    processed,
                    len(images),
                    started_total,
                )

    if use_vlm:
        print(
            f"CNN+VLM fallback: VLM used for "
            f"{vlm_count}/{len(images)} images",
            flush=True,
        )

    return PredictionResult(
        predictions,
        inference_times,
    )


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
