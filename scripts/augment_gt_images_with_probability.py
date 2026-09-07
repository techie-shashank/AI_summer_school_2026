from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = (
    DEFAULT_PROJECT_ROOT
    / "seals-dataset/ground_truth/01_510_800_820_280_114"
)
DEFAULT_OUTPUT_DIR = (
    DEFAULT_PROJECT_ROOT
    / "seals-dataset/ground_truth/01_510_800_820_280_114_augmented"
)

IMAGE_EXTENSIONS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


def collect_images(
    input_dir: Path,
    recursive: bool,
) -> list[Path]:

    iterator = (
        input_dir.rglob("*")
        if recursive
        else input_dir.iterdir()
    )

    return sorted(
        path
        for path in iterator
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def random_rotate_scale(
    image: np.ndarray,
    rng: np.random.Generator,
    max_rotation_degrees: float,
    scale_range: tuple[float, float],
) -> np.ndarray:

    height, width = image.shape[:2]

    angle = rng.uniform(
        -max_rotation_degrees,
        max_rotation_degrees,
    )

    scale = rng.uniform(
        scale_range[0],
        scale_range[1],
    )

    center = (
        (width - 1) / 2.0,
        (height - 1) / 2.0,
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        scale,
    )

    border_value = median_border_value(image)

    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def median_border_value(
    image: np.ndarray,
) -> int | tuple[int, ...]:

    if image.ndim == 2:
        border = np.concatenate(
            [
                image[0, :],
                image[-1, :],
                image[:, 0],
                image[:, -1],
            ]
        )

        return int(np.median(border))

    border = np.concatenate(
        [
            image[0, :, :],
            image[-1, :, :],
            image[:, 0, :],
            image[:, -1, :],
        ],
        axis=0,
    )

    return tuple(
        int(value)
        for value in np.median(
            border,
            axis=0,
        )
    )


def add_salt_and_pepper_noise(
    image: np.ndarray,
    rng: np.random.Generator,
    amount_range: tuple[float, float],
    salt_ratio: float,
) -> np.ndarray:

    amount = rng.uniform(
        amount_range[0],
        amount_range[1],
    )

    if amount <= 0:
        return image

    noisy = image.copy()

    height, width = image.shape[:2]

    mask = rng.random(
        (height, width)
    )

    pepper_threshold = (
        amount * (1.0 - salt_ratio)
    )

    salt_threshold = (
        1.0 - (amount * salt_ratio)
    )

    if image.ndim == 2:

        noisy[
            mask < pepper_threshold
        ] = 0

        noisy[
            mask > salt_threshold
        ] = 255

    else:

        noisy[
            mask < pepper_threshold,
            :,
        ] = 0

        noisy[
            mask > salt_threshold,
            :,
        ] = 255

    return noisy


def random_blur(
    image: np.ndarray,
    rng: np.random.Generator,
    max_kernel_size: int,
) -> np.ndarray:

    if max_kernel_size < 3:
        return image

    odd_kernel_sizes = list(
        range(
            3,
            max_kernel_size + 1,
            2,
        )
    )

    kernel_size = int(
        rng.choice(
            odd_kernel_sizes
        )
    )

    return cv2.GaussianBlur(
        image,
        (
            kernel_size,
            kernel_size,
        ),
        sigmaX=0,
    )


def add_random_rectangles(
    image: np.ndarray,
    rng: np.random.Generator,
    count_range: tuple[int, int],
    brightness_range: tuple[int, int],
) -> np.ndarray:

    height, width = image.shape[:2]

    if height < 2 or width < 2:
        return image

    augmented = image.copy()

    min_count, max_count = count_range

    count = int(
        rng.integers(
            min_count,
            max_count + 1,
        )
    )

    upper_half_height = max(
        1,
        height // 2,
    )

    for _ in range(count):

        rect_width = int(
            rng.integers(
                max(1, width // 12),
                max(2, width // 3) + 1,
            )
        )

        rect_height = int(
            rng.integers(
                max(1, upper_half_height // 12),
                max(2, upper_half_height // 3) + 1,
            )
        )

        x1 = int(
            rng.integers(
                0,
                max(
                    1,
                    width - rect_width + 1,
                ),
            )
        )

        y1 = int(
            rng.integers(
                0,
                max(
                    1,
                    upper_half_height - rect_height + 1,
                ),
            )
        )

        x2 = min(
            width,
            x1 + rect_width,
        )

        y2 = min(
            upper_half_height,
            y1 + rect_height,
        )

        brightness_delta = int(
            rng.integers(
                brightness_range[0],
                brightness_range[1] + 1,
            )
        )

        region = (
            augmented[y1:y2, x1:x2]
            .astype(np.int16)
            + brightness_delta
        )

        augmented[
            y1:y2,
            x1:x2,
        ] = np.clip(
            region,
            0,
            255,
        ).astype(np.uint8)

    return augmented


def add_gradient_brightness(
    image: np.ndarray,
    rng: np.random.Generator,
    strength_range: tuple[int, int],
) -> np.ndarray:

    height, width = image.shape[:2]

    if height < 1 or width < 1:
        return image

    yy, xx = np.mgrid[
        0:height,
        0:width,
    ]

    start_x = rng.uniform(
        0,
        max(1, width - 1),
    )

    start_y = rng.uniform(
        0,
        max(1, height - 1),
    )

    direction = rng.uniform(
        0,
        2 * np.pi,
    )

    projection = (
        (xx - start_x) * np.cos(direction)
        + (yy - start_y) * np.sin(direction)
    )

    projection -= projection.min()

    max_projection = projection.max()

    if max_projection > 0:
        projection /= max_projection

    strength = int(
        rng.integers(
            strength_range[0],
            strength_range[1] + 1,
        )
    )

    gradient = projection * strength

    if image.ndim == 3:
        gradient = gradient[:, :, None]

    augmented = (
        image.astype(np.float32)
        + gradient
    )

    return np.clip(
        augmented,
        0,
        255,
    ).astype(np.uint8)


def should_apply(
    rng: np.random.Generator,
    probability: float,
) -> bool:
    """
    Return True with the specified probability.
    """

    return bool(
        rng.random() < probability
    )


def augment_image(
    image: np.ndarray,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> np.ndarray:

    augmented = image

    # ---------------------------------------------------------------
    # Geometric augmentation
    # ---------------------------------------------------------------

    if should_apply(
        rng,
        args.geometry_probability,
    ):

        augmented = random_rotate_scale(
            augmented,
            rng,
            args.max_rotation_degrees,
            (
                args.min_scale,
                args.max_scale,
            ),
        )

    # ---------------------------------------------------------------
    # Gradient brightness
    # ---------------------------------------------------------------

    if should_apply(
        rng,
        args.gradient_probability,
    ):

        augmented = add_gradient_brightness(
            augmented,
            rng,
            (
                args.min_gradient_strength,
                args.max_gradient_strength,
            ),
        )

    # ---------------------------------------------------------------
    # Local brightness rectangles
    # ---------------------------------------------------------------

    if should_apply(
        rng,
        args.rectangle_probability,
    ):

        augmented = add_random_rectangles(
            augmented,
            rng,
            (
                args.min_rectangles,
                args.max_rectangles,
            ),
            (
                args.min_rectangle_brightness,
                args.max_rectangle_brightness,
            ),
        )

    # ---------------------------------------------------------------
    # Blur
    # ---------------------------------------------------------------

    if should_apply(
        rng,
        args.blur_probability,
    ):

        augmented = random_blur(
            augmented,
            rng,
            args.max_blur_kernel_size,
        )

    # ---------------------------------------------------------------
    # Salt & pepper noise
    # ---------------------------------------------------------------

    if should_apply(
        rng,
        args.noise_probability,
    ):

        augmented = add_salt_and_pepper_noise(
            augmented,
            rng,
            (
                args.min_noise_amount,
                args.max_noise_amount,
            ),
            args.salt_ratio,
        )

    return augmented


def output_path_for(
    image_path: Path,
    input_dir: Path,
    output_dir: Path,
    variant_index: int,
    recursive: bool,
) -> Path:

    relative_parent = (
        image_path.relative_to(input_dir).parent
        if recursive
        else Path()
    )

    filename = (
        f"{image_path.stem}"
        f"_aug{variant_index:03d}"
        f"{image_path.suffix}"
    )

    return (
        output_dir
        / relative_parent
        / filename
    )


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Create augmented copies of "
            "ground-truth seal images."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--variants-per-image",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--recursive",
        action="store_true",
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    # ---------------------------------------------------------------
    # Augmentation probabilities
    # ---------------------------------------------------------------

    parser.add_argument(
        "--geometry-probability",
        type=float,
        default=0.8,
        help="Probability of applying rotation/scale.",
    )

    parser.add_argument(
        "--gradient-probability",
        type=float,
        default=0.5,
        help="Probability of applying gradient brightness.",
    )

    parser.add_argument(
        "--rectangle-probability",
        type=float,
        default=0.4,
        help="Probability of applying local brightness rectangles.",
    )

    parser.add_argument(
        "--blur-probability",
        type=float,
        default=0.3,
        help="Probability of applying Gaussian blur.",
    )

    parser.add_argument(
        "--noise-probability",
        type=float,
        default=0.2,
        help="Probability of applying salt-and-pepper noise.",
    )

    # ---------------------------------------------------------------
    # Augmentation parameters
    # ---------------------------------------------------------------

    parser.add_argument(
        "--max-rotation-degrees",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--min-scale",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--max-scale",
        type=float,
        default=1.1,
    )

    parser.add_argument(
        "--min-noise-amount",
        type=float,
        default=0.002,
    )

    parser.add_argument(
        "--max-noise-amount",
        type=float,
        default=0.015,
    )

    parser.add_argument(
        "--salt-ratio",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--max-blur-kernel-size",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--min-rectangles",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--max-rectangles",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--min-rectangle-brightness",
        type=int,
        default=-80,
    )

    parser.add_argument(
        "--max-rectangle-brightness",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--min-gradient-strength",
        type=int,
        default=-60,
    )

    parser.add_argument(
        "--max-gradient-strength",
        type=int,
        default=60,
    )

    return parser.parse_args()


def validate_args(
    args: argparse.Namespace,
) -> None:

    if not args.input_dir.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: "
            f"{args.input_dir}"
        )

    if args.variants_per_image < 1:
        raise ValueError(
            "--variants-per-image must be at least 1"
        )

    if (
        args.min_scale <= 0
        or args.max_scale <= 0
        or args.min_scale > args.max_scale
    ):
        raise ValueError(
            "Scale bounds must be positive and "
            "--min-scale <= --max-scale"
        )

    if not (
        0
        <= args.min_noise_amount
        <= args.max_noise_amount
        <= 1
    ):
        raise ValueError(
            "Noise amount bounds must satisfy "
            "0 <= min <= max <= 1"
        )

    if not 0 <= args.salt_ratio <= 1:
        raise ValueError(
            "--salt-ratio must be between 0 and 1"
        )

    probabilities = (
        args.geometry_probability,
        args.gradient_probability,
        args.rectangle_probability,
        args.blur_probability,
        args.noise_probability,
    )

    if any(
        probability < 0 or probability > 1
        for probability in probabilities
    ):
        raise ValueError(
            "All augmentation probabilities "
            "must be between 0 and 1."
        )

    if args.max_blur_kernel_size % 2 == 0:
        raise ValueError(
            "--max-blur-kernel-size must be odd"
        )

    if (
        args.min_rectangles < 0
        or args.min_rectangles > args.max_rectangles
    ):
        raise ValueError(
            "Rectangle bounds must satisfy "
            "0 <= min <= max"
        )

    if (
        args.min_rectangle_brightness
        > args.max_rectangle_brightness
    ):
        raise ValueError(
            "Rectangle brightness bounds must "
            "satisfy min <= max"
        )

    if (
        args.min_gradient_strength
        > args.max_gradient_strength
    ):
        raise ValueError(
            "Gradient strength bounds must "
            "satisfy min <= max"
        )


def main() -> None:

    args = parse_args()

    validate_args(args)

    image_paths = collect_images(
        args.input_dir,
        args.recursive,
    )

    if not image_paths:
        raise ValueError(
            f"No image files found in: "
            f"{args.input_dir}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = np.random.default_rng(
        args.seed
    )

    written = 0
    skipped = 0

    for image_path in image_paths:

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_UNCHANGED,
        )

        if image is None:
            raise ValueError(
                f"Could not read image: "
                f"{image_path}"
            )

        if image.dtype != np.uint8:
            image = cv2.normalize(
                image,
                None,
                0,
                255,
                cv2.NORM_MINMAX,
            ).astype(np.uint8)

        for variant_index in range(
            1,
            args.variants_per_image + 1,
        ):

            output_path = output_path_for(
                image_path,
                args.input_dir,
                args.output_dir,
                variant_index,
                args.recursive,
            )

            if (
                output_path.exists()
                and not args.force
            ):
                skipped += 1
                continue

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            augmented = augment_image(
                image,
                rng,
                args,
            )

            if not cv2.imwrite(
                str(output_path),
                augmented,
            ):
                raise ValueError(
                    f"Could not write image: "
                    f"{output_path}"
                )

            written += 1

    print(
        f"Input images: {len(image_paths)}"
    )

    print(
        f"Variants per image: "
        f"{args.variants_per_image}"
    )

    print(
        f"Written: {written}"
    )

    print(
        f"Skipped existing: {skipped}"
    )

    print(
        f"Output directory: "
        f"{args.output_dir}"
    )

    print()
    print("Augmentation probabilities:")
    print(
        f"  geometry:   "
        f"{args.geometry_probability:.2f}"
    )
    print(
        f"  gradient:   "
        f"{args.gradient_probability:.2f}"
    )
    print(
        f"  rectangles: "
        f"{args.rectangle_probability:.2f}"
    )
    print(
        f"  blur:       "
        f"{args.blur_probability:.2f}"
    )
    print(
        f"  noise:      "
        f"{args.noise_probability:.2f}"
    )


if __name__ == "__main__":
    main()
