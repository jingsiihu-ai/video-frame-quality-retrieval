"""Dependency-light image-quality signals for frame filtering."""

from __future__ import annotations

import numpy as np


def _grayscale(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim == 3 and array.shape[-1] >= 3:
        array = (
            0.299 * array[..., 0]
            + 0.587 * array[..., 1]
            + 0.114 * array[..., 2]
        )
    if array.ndim != 2:
        raise ValueError("frame must be a grayscale or RGB-like image")
    array = array.astype(np.float64)
    if array.size == 0:
        raise ValueError("frame must not be empty")
    if array.max() > 1.0:
        array = array / 255.0
    return np.clip(array, 0.0, 1.0)


def brightness_score(frame: np.ndarray, target: float = 0.5) -> float:
    """Score proximity to a target mean brightness in [0, 1]."""

    if not 0.0 < target < 1.0:
        raise ValueError("target must be strictly between 0 and 1")
    mean = float(_grayscale(frame).mean())
    scale = max(target, 1.0 - target)
    return float(np.clip(1.0 - abs(mean - target) / scale, 0.0, 1.0))


def contrast_score(frame: np.ndarray) -> float:
    """Map grayscale standard deviation to a bounded contrast score."""

    deviation = float(_grayscale(frame).std())
    return float(1.0 - np.exp(-5.0 * deviation))


def sharpness_score(frame: np.ndarray) -> float:
    """Bounded variance-of-Laplacian sharpness proxy."""

    image = _grayscale(frame)
    padded = np.pad(image, 1, mode="edge")
    laplacian = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4.0 * image
    )
    return float(np.tanh(4.0 * laplacian.var()))


def composite_quality_score(
    frame: np.ndarray,
    brightness_weight: float = 0.25,
    contrast_weight: float = 0.25,
    sharpness_weight: float = 0.50,
) -> float:
    """Weighted combination of independently inspectable quality signals."""

    weights = np.asarray(
        [brightness_weight, contrast_weight, sharpness_weight], dtype=np.float64
    )
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("quality weights must be non-negative and sum above zero")
    signals = np.asarray(
        [brightness_score(frame), contrast_score(frame), sharpness_score(frame)]
    )
    return float(np.dot(weights / weights.sum(), signals))

