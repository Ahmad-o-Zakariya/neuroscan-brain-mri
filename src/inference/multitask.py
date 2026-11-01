"""Preprocessing and safe loading shared by the multitask dashboard and tests."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from src.config import CLASSES, MEAN, STD
from src.models.multitask import MultiTaskModel


def preprocess_image(image: Image.Image, size: int = 224, mean=MEAN, std=STD) -> torch.Tensor:
    """Match notebook inference: EXIF transpose, RGB, square bilinear resize, ImageNet normalization."""
    prepared = ImageOps.exif_transpose(image).convert("RGB").resize(
        (int(size), int(size)), Image.Resampling.BILINEAR
    )
    array = np.asarray(prepared, dtype=np.float32).copy() / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean_tensor = torch.as_tensor(mean, dtype=tensor.dtype)[:, None, None]
    std_tensor = torch.as_tensor(std, dtype=tensor.dtype)[:, None, None]
    return (tensor - mean_tensor) / std_tensor


def load_multitask_checkpoint(checkpoint_path, device="cpu"):
    """Load a local checkpoint in weights-only mode with only TorchVersion allowlisted."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Multi-task checkpoint not found: {path}")

    safe_globals = []
    try:
        from torch.torch_version import TorchVersion

        safe_globals.append(TorchVersion)
    except ImportError:  # pragma: no cover - older supported torch builds
        pass

    with torch.serialization.safe_globals(safe_globals):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    metadata = checkpoint.get("metadata") or {}
    architecture = metadata.get("architecture")
    if architecture not in {"convnext_tiny", "swin_tiny"}:
        raise ValueError(f"Unsupported or missing checkpoint architecture: {architecture!r}")
    model = MultiTaskModel(architecture, int(metadata.get("input_size", 224)))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model, metadata


def classify_probabilities(probabilities, glioma_threshold=None):
    """Apply the saved validation-calibrated glioma override, if present."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.shape != (len(CLASSES),):
        raise ValueError(f"Expected {len(CLASSES)} class probabilities, got shape {values.shape}")
    if glioma_threshold is not None and values[0] >= float(glioma_threshold):
        return 0
    if glioma_threshold is not None:
        return int(1 + np.argmax(values[1:]))
    return int(np.argmax(values))
