"""Retrieve diverse key frames from a deterministic synthetic video."""

from __future__ import annotations

import numpy as np

from video_frame_retrieval import (
    fuse_modalities,
    rank_frames,
    two_stage_temporal_scores,
)


def main() -> None:
    rng = np.random.default_rng(11)
    frame_count, dimensions = 48, 16
    scene_prototypes = rng.normal(size=(3, dimensions))
    scene_ids = np.repeat(np.arange(3), 16)

    visual = scene_prototypes[scene_ids] + rng.normal(0.0, 0.12, (frame_count, dimensions))
    ocr = scene_prototypes[scene_ids] + rng.normal(0.0, 0.25, (frame_count, dimensions))
    asr = scene_prototypes[scene_ids] + rng.normal(0.0, 0.30, (frame_count, dimensions))

    visual[[13, 14]] = rng.normal(size=(2, dimensions))
    quality = np.full(frame_count, 0.82)
    quality[[7, 13, 31]] = [0.18, 0.12, 0.25]

    fused = fuse_modalities(visual, ocr, asr)
    anomaly = two_stage_temporal_scores(fused)
    query = scene_prototypes[2]
    selected = rank_frames(
        fused,
        query,
        quality,
        anomaly_scores=anomaly,
        k=6,
        candidate_budget=18,
    )

    print("selected frames:", [item.index for item in selected])
    print("scene IDs:", [int(scene_ids[item.index]) for item in selected])
    print("utilities:", [round(item.utility, 3) for item in selected])


if __name__ == "__main__":
    main()

