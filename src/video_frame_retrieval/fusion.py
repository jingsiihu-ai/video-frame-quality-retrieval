"""Utilities for combining aligned visual, OCR, and ASR embeddings."""

from __future__ import annotations

import numpy as np


def l2_normalize(values: np.ndarray, axis: int = -1) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    norm = np.linalg.norm(array, axis=axis, keepdims=True)
    return array / np.clip(norm, 1e-12, None)


def fuse_modalities(
    visual: np.ndarray,
    ocr: np.ndarray | None = None,
    asr: np.ndarray | None = None,
    weights: tuple[float, float, float] = (0.60, 0.25, 0.15),
) -> np.ndarray:
    """Fuse available, shape-aligned modalities and normalize the result."""

    visual = np.asarray(visual, dtype=np.float64)
    if visual.ndim != 2:
        raise ValueError("visual embeddings must have shape [frames, dimensions]")
    arrays = [visual, ocr, asr]
    active: list[tuple[np.ndarray, float]] = []
    for values, weight in zip(arrays, weights):
        if weight < 0:
            raise ValueError("fusion weights must be non-negative")
        if values is None:
            continue
        array = np.asarray(values, dtype=np.float64)
        if array.shape != visual.shape:
            raise ValueError("all available modalities must match visual shape")
        if weight > 0:
            active.append((l2_normalize(array), weight))
    total = sum(weight for _, weight in active)
    if total <= 0:
        raise ValueError("at least one available modality must have positive weight")
    fused = sum(array * (weight / total) for array, weight in active)
    return l2_normalize(fused)

