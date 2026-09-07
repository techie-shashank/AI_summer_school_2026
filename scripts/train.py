from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from nets import DigitCNN
from seal_digits_dataset import SealDigitsDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_MANIFEST = PROJECT_ROOT / "splits/ground_truth_chars_balanced/train.txt"


def parse_image_size(value: str) -> tuple[int, int]:
    parts = value.lower().replace(",", "x").split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected WIDTHxHEIGHT, for example 28x28")

    try:
        width, height = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Image size must contain integers") from exc

    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Image width and height must be positive")
    return width, height


def build_loader(
    manifest_path: str | Path,
    image_size: tuple[int, int],
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> torch.utils.data.DataLoader:
    dataset = SealDigitsDataset(
        manifest_path=manifest_path,
        project_root=PROJECT_ROOT,
        image_size=image_size,
        transform=transforms.ToTensor(),
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)

    running_loss = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if is_training:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_training:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * labels.size(0)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    mean_loss = running_loss / total
    accuracy = 100 * correct / total
    return mean_loss, accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CNN on seal numeral crops.")
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument(
        "--val-manifest",
        type=Path,
        default=None,
        help="Optional validation manifest. If provided, validation runs after each epoch.",
    )
    parser.add_argument(
        "--test-manifest",
        type=Path,
        default=None,
        help="Optional test manifest. If provided, testing runs once after training.",
    )
    parser.add_argument("--image-size", type=parse_image_size, default=(28, 28))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_loader = build_loader(
        manifest_path=args.train_manifest,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    val_loader = (
        build_loader(
            manifest_path=args.val_manifest,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )
        if args.val_manifest is not None
        else None
    )
    test_loader = (
        build_loader(
            manifest_path=args.test_manifest,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )
        if args.test_manifest is not None
        else None
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    model = DigitCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        train_loss, train_accuracy = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )

        message = (
            f"Epoch {epoch + 1:2d} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: {train_accuracy:.2f}%"
        )

        if val_loader is not None:
            val_loss, val_accuracy = run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
            )
            message += f" | Val loss: {val_loss:.4f} | Val accuracy: {val_accuracy:.2f}%"

        print(message)

    if test_loader is not None:
        test_loss, test_accuracy = run_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
        )
        print(f"Test loss: {test_loss:.4f} | Test accuracy: {test_accuracy:.2f}%")


if __name__ == "__main__":
    main()
