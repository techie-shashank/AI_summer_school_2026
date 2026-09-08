from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.json"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("Config file must contain a JSON object")
    return config


def config_path(value: str | Path, config_root: Path = ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else config_root / path


def resolve_device(value: str | None) -> torch.device:
    requested = (value or "gpu").lower()
    if requested not in {"cpu", "gpu"}:
        raise ValueError("device must be either 'cpu' or 'gpu'")
    if requested == "gpu":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("Warning: GPU requested but no CUDA/MPS device is available; using CPU")
    return torch.device("cpu")