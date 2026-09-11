"""Quality-aware scoring and maximal marginal relevance reranking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fusion import l2_normalize


@dataclass(frozen=True)
class FrameSelection:
    index: int
    relevance: float
    quality: float
    anomaly: float
    utility: float


def mmr_select(
    embeddings: np.ndarray,
    relevance: np.ndarray,
    k: int,
    diversity_tradeoff: float = 0.65,
) -> list[int]:
    """Greedy maximal marginal relevance selection."""

    vectors = l2_normalize(np.asarray(embeddings, dtype=np.float64))
    relevance = np.asarray(relevance, dtype=np.float64)
    if vectors.ndim != 2 or relevance.shape != (vectors.shape[0],):
        raise ValueError("relevance must provide one score per embedding")
    if k <= 0 or not 0.0 <= diversity_tradeoff <= 1.0:
        raise ValueError("k must be positive and diversity_tradeoff in [0, 1]")
    if len(vectors) == 0:
        return []
    selected: list[int] = []
    remaining = set(range(len(vectors)))
    similarity = vectors @ vectors.T
    while remaining and len(selected) < min(k, len(vectors)):
        best_index = -1
        best_score = -np.inf
        for index in sorted(remaining):
            redundancy = max((similarity[index, other] for other in selected), default=0.0)
            score = (
                diversity_tradeoff * relevance[index]
                - (1.0 - diversity_tradeoff) * redundancy
            )
            if score > best_score:
                best_index, best_score = index, score
        selected.append(best_index)
        remaining.remove(best_index)
    return selected


def rank_frames(
    embeddings: np.ndarray,
    query_embedding: np.ndarray,
    quality_scores: np.ndarray,
    anomaly_scores: np.ndarray | None = None,
    k: int = 8,
    candidate_budget: int = 24,
    relevance_weight: float = 0.60,
    quality_weight: float = 0.30,
    anomaly_penalty: float = 0.10,
    diversity_tradeoff: float = 0.65,
) -> list[FrameSelection]:
    """Prefilter by utility, then rerank the candidate pool with MMR."""

    vectors = np.asarray(embeddings, dtype=np.float64)
    quality = np.asarray(quality_scores, dtype=np.float64)
    if vectors.ndim != 2 or quality.shape != (vectors.shape[0],):
        raise ValueError("quality_scores must provide one score per frame")
    anomaly = (
        np.zeros(len(vectors), dtype=np.float64)
        if anomaly_scores is None
        else np.asarray(anomaly_scores, dtype=np.float64)
    )
    if anomaly.shape != (len(vectors),):
        raise ValueError("anomaly_scores must provide one score per frame")
    if k <= 0 or candidate_budget <= 0:
        raise ValueError("k and candidate_budget must be positive")
    query = l2_normalize(np.asarray(query_embedding, dtype=np.float64))
    if query.shape != (vectors.shape[1],):
        raise ValueError("query dimension must match frame embeddings")

    normalized = l2_normalize(vectors)
    relevance = np.clip((normalized @ query + 1.0) / 2.0, 0.0, 1.0)
    utility = (
        relevance_weight * relevance
        + quality_weight * np.clip(quality, 0.0, 1.0)
        - anomaly_penalty * np.clip(anomaly, 0.0, 1.0)
    )
    pool_size = min(max(k, candidate_budget), len(vectors))
    pool = np.argsort(-utility, kind="stable")[:pool_size]
    local = mmr_select(
        normalized[pool], utility[pool], k=k, diversity_tradeoff=diversity_tradeoff
    )
    return [
        FrameSelection(
            index=int(pool[position]),
            relevance=float(relevance[pool[position]]),
            quality=float(quality[pool[position]]),
            anomaly=float(anomaly[pool[position]]),
            utility=float(utility[pool[position]]),
        )
        for position in local
    ]

