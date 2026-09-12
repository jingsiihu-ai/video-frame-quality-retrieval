# Reproducible results

These results exercise the complete public ranking pipeline with deterministic synthetic inputs. They are a reproducibility check, not a claim about production video data.

## Deterministic synthetic retrieval

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python examples/synthetic_demo.py
```

```text
selected frames: [36, 4, 35, 38, 42, 37]
scene IDs: [2, 0, 2, 2, 2, 2]
utilities: [0.845, 0.628, 0.842, 0.843, 0.843, 0.842]
```

The target query represents scene 2. Five of the six selected frames come from that scene; MMR also retains frame 4 from scene 0 as a diversity tradeoff. The fixed seed and fully synthetic arrays make the output directly reproducible.

## CPU microbenchmark

```bash
PYTHONPATH=src python benchmarks/benchmark.py
```

Reference run on Linux x86_64, Intel Xeon Platinum 8573C, Python 3.12.14, and NumPy 2.3.5:

| Workload | Median | p95 | Input arrays |
| --- | ---: | ---: | ---: |
| 480 frames × 128 dimensions; 48 candidates → 8 frames | 9.579 ms | 10.832 ms | 1.411 MiB |

The timed path includes three-modality fusion, two-stage temporal scoring, utility ranking, and MMR reranking. The script performs three warmups and reports 20 end-to-end measurements. `Input arrays` is the exact NumPy `nbytes` total for the benchmark inputs, not process peak memory. Runtime is a point measurement on shared cloud hardware and should be rerun on the target machine.

## Test coverage

The current tests cover bounded quality signals, deterministic sampling, temporal outlier detection, normalized fusion, MMR validity, and quality-aware relevance ranking. Run them after installing the development extra:

```bash
pip install -e '.[dev]'
pytest
```

## Next public-data experiment

A defensible next milestone is a small, licensed video set with documented queries and frame-level relevance labels. Report retrieval precision/recall, temporal coverage, runtime, and ablations for quality, anomaly, and MMR components. No public-data metric is claimed until that experiment is checked into the repository.
