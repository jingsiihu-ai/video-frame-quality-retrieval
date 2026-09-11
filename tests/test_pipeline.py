import numpy as np

from video_frame_retrieval import (
    brightness_score,
    composite_quality_score,
    fuse_modalities,
    local_isolation_scores,
    mmr_select,
    rank_frames,
    sharpness_score,
    two_stage_temporal_scores,
    uniform_indices,
)


def test_uniform_sampling_retains_endpoints() -> None:
    indices = uniform_indices(frame_count=100, budget=8)
    assert len(indices) == 8
    assert indices[0] == 0 and indices[-1] == 99
    assert len(np.unique(indices)) == len(indices)


def test_quality_signals_are_bounded() -> None:
    checkerboard = (np.indices((16, 16)).sum(axis=0) % 2 * 255).astype(np.uint8)
    flat = np.full((16, 16), 128, dtype=np.uint8)
    assert 0.0 <= composite_quality_score(checkerboard) <= 1.0
    assert sharpness_score(checkerboard) > sharpness_score(flat)
    assert brightness_score(flat) > brightness_score(np.zeros((16, 16)))


def test_temporal_outlier_receives_highest_score() -> None:
    embeddings = np.tile(np.array([[1.0, 0.0, 0.0]]), (9, 1))
    embeddings[4] = [0.0, 1.0, 0.0]
    isolation = local_isolation_scores(embeddings, window=2)
    combined = two_stage_temporal_scores(embeddings, window=2)
    assert int(isolation.argmax()) == 4
    assert int(combined.argmax()) == 4


def test_fusion_returns_unit_vectors() -> None:
    rng = np.random.default_rng(2)
    visual = rng.normal(size=(5, 8))
    text = rng.normal(size=(5, 8))
    fused = fuse_modalities(visual, ocr=text)
    assert np.allclose(np.linalg.norm(fused, axis=1), 1.0)


def test_mmr_returns_unique_valid_indices() -> None:
    embeddings = np.eye(5)
    selected = mmr_select(embeddings, np.arange(5, dtype=float), k=3)
    assert len(selected) == len(set(selected)) == 3
    assert all(0 <= index < 5 for index in selected)


def test_rank_frames_prefers_relevant_high_quality_candidates() -> None:
    embeddings = np.array(
        [[1.0, 0.0], [0.95, 0.05], [-1.0, 0.0], [0.0, 1.0]]
    )
    quality = np.array([0.9, 0.1, 1.0, 0.8])
    selected = rank_frames(
        embeddings,
        np.array([1.0, 0.0]),
        quality,
        k=2,
        candidate_budget=4,
    )
    assert selected[0].index == 0
    assert len({item.index for item in selected}) == 2

