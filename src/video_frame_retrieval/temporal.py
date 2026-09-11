"""Temporal inconsistency scores over frame embeddings."""

from __future__ import annotations

import numpy as np

from .fusion import l2_normalize


def _validate(embeddings: np.ndarray, window: int) -> np.ndarray:
    array = np.asarray(embeddings, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError("embeddings must have shape [frames, dimensions]")
    if window <= 0:
        raise ValueError("window must be positive")
    return l2_normalize(array)


def local_isolation_scores(embeddings: np.ndarray, window: int = 2) -> np.ndarray:
    """Cosine distance from each frame to its local temporal context."""

    normalized = _validate(embeddings, window)
    frame_count = normalized.shape[0]
    scores = np.zeros(frame_count, dtype=np.float64)
    for index in range(frame_count):
        left = max(0, index - window)
        right = min(frame_count, index + window + 1)
        neighbors = np.concatenate(
            [normalized[left:index], normalized[index + 1 : right]], axis=0
        )
        if len(neighbors) == 0:
            continue
        context = l2_normalize(neighbors.mean(axis=0))
        scores[index] = np.clip((1.0 - normalized[index] @ context) / 2.0, 0.0, 1.0)
    return scores


def neighborhood_persistence_scores(
    isolation_scores: np.ndarray, radius: int = 1
) -> np.ndarray:
    """Smooth isolated anomalies to distinguish single glitches from short events."""

    scores = np.asarray(isolation_scores, dtype=np.float64)
    if scores.ndim != 1 or radius < 0:
        raise ValueError("scores must be one-dimensional and radius non-negative")
    output = np.zeros_like(scores)
    for index in range(len(scores)):
        left = max(0, index - radius)
        right = min(len(scores), index + radius + 1)
        output[index] = scores[left:right].mean()
    return output


def two_stage_temporal_scores(
    embeddings: np.ndarray,
    window: int = 2,
    isolation_weight: float = 0.75,
    persistence_weight: float = 0.25,
) -> np.ndarray:
    """Combine immediate isolation with neighborhood-level persistence."""

    if isolation_weight < 0 or persistence_weight < 0:
        raise ValueError("weights must be non-negative")
    total = isolation_weight + persistence_weight
    if total <= 0:
        raise ValueError("weights must sum above zero")
    isolation = local_isolation_scores(embeddings, window=window)
    persistence = neighborhood_persistence_scores(isolation)
    return (isolation_weight * isolation + persistence_weight * persistence) / total

