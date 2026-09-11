"""Deterministic video sampling helpers."""

from __future__ import annotations

import numpy as np


def uniform_indices(frame_count: int, budget: int) -> np.ndarray:
    """Select up to `budget` unique indices while retaining both endpoints."""

    if frame_count < 0 or budget <= 0:
        raise ValueError("frame_count must be non-negative and budget positive")
    if frame_count == 0:
        return np.empty(0, dtype=np.int64)
    if budget >= frame_count:
        return np.arange(frame_count, dtype=np.int64)
    return np.rint(np.linspace(0, frame_count - 1, budget)).astype(np.int64)

