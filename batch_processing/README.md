# Batch Processing Guide

This directory contains batch utilities for segmentation/centerline processing and the
legacy-vs-primary benchmark suite.

## Scripts

### Segmentation and Centerline Utilities

- `batch_segmentation.py`
  - Batch vessel segmentation from MRI `.nii.gz` inputs.
- `batch_extract_centerlines.py`
  - Batch centerline extraction from vessel segmentation `.npy` files.
- `batch_process_centerlines.py`
  - End-to-end centerline extraction + JSON conversion with spacing handling.
- `batch_convert_vtp_to_json.py`
  - Convert centerline `.vtp` files to JSON format.

### Legacy-vs-Primary Benchmark Suite

- `benchmark_legacy_vs_primary.py`
  - Benchmark runner that executes one compare-job per study and writes per-study records.
  - Supports manifest generation (`--manifest-only`) for Slurm head-node orchestration.
  - Works in Slurm arrays (`--manifest-file` + `--task-index`) so each array task handles one study.
- `reduce_legacy_vs_primary.py`
  - Reducer that aggregates per-study benchmark records into summary CSV/JSON artifacts.

## Legacy-vs-Primary Benchmark Usage

### 1. Build a study manifest from segmentation files

```bash
micromamba run -n vanguard python batch_processing/benchmark_legacy_vs_primary.py \
  --segmentation-dir /net/projects2/vanguard/vessel_segmentations \
  --output-dir /net/projects2/vanguard/benchmarks/legacy_vs_primary/run_001 \
  --manifest-only
```

This writes:
- `study_ids.txt`

### 2. Run benchmark locally (single process)

```bash
micromamba run -n vanguard python batch_processing/benchmark_legacy_vs_primary.py \
  --manifest-file /net/projects2/vanguard/benchmarks/legacy_vs_primary/run_001/study_ids.txt \
  --output-dir /net/projects2/vanguard/benchmarks/legacy_vs_primary/run_001 \
  --segmentation-dir /net/projects2/vanguard/vessel_segmentations \
  --compare-script graph_extraction/run_compare_legacy_pipeline_debug.py
```

Per-study outputs are written under:
- `studies/<study_id>/artifacts/` (includes rotating 3D MP4 and summary JSON from compare script)
- `studies/<study_id>/benchmark_record.json`

Top-level outputs (non-array runs):
- `benchmark_records.jsonl`
- `benchmark_records.csv`
- `runner_summary.json`

### 3. Reduce benchmark records

```bash
micromamba run -n vanguard python batch_processing/reduce_legacy_vs_primary.py \
  --benchmark-dir /net/projects2/vanguard/benchmarks/legacy_vs_primary/run_001
```

Reducer outputs:
- `benchmark_reduced_per_study.csv`
- `benchmark_reduced_summary.json`

## Slurm Usage

For cluster execution, use the scripts in `slurm_submit_scripts/`:
- `submit_legacy_vs_primary_benchmark.sh` (head-node orchestrator)
- `submit_legacy_vs_primary_benchmark_array.slurm` (array worker)
- `submit_legacy_vs_primary_benchmark_reduce.slurm` (reducer)

See `slurm_submit_scripts/README.md` for exact commands.
