"""Small, reproducible CPU benchmark for the retrieval pipeline."""

from __future__ import annotations

import platform
import statistics
import time

import numpy as np

from video_frame_retrieval import fuse_modalities, rank_frames, two_stage_temporal_scores


def main() -> None:
    rng = np.random.default_rng(23)
    frames, dimensions = 480, 128
    visual = rng.normal(size=(frames, dimensions))
    ocr = rng.normal(size=(frames, dimensions))
    asr = rng.normal(size=(frames, dimensions))
    query = rng.normal(size=dimensions)
    quality = rng.uniform(0.4, 1.0, size=frames)

    def run_once() -> None:
        fused = fuse_modalities(visual, ocr, asr)
        anomaly = two_stage_temporal_scores(fused)
        rank_frames(
            fused,
            query,
            quality,
            anomaly_scores=anomaly,
            k=8,
            candidate_budget=48,
        )

    for _ in range(3):
        run_once()

    timings_ms: list[float] = []
    for _ in range(20):
        start = time.perf_counter()
        run_once()
        timings_ms.append((time.perf_counter() - start) * 1_000)

    p95 = sorted(timings_ms)[int(0.95 * len(timings_ms)) - 1]
    input_mib = (visual.nbytes + ocr.nbytes + asr.nbytes + query.nbytes + quality.nbytes) / (1024**2)
    print(f"python: {platform.python_version()}")
    print(f"numpy: {np.__version__}")
    print(f"shape: frames={frames}, dimensions={dimensions}, candidates=48, selected=8")
    print(f"median_ms: {statistics.median(timings_ms):.3f}")
    print(f"p95_ms: {p95:.3f}")
    print(f"input_array_mib: {input_mib:.3f}")


if __name__ == "__main__":
    main()
