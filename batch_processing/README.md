# Batch Processing Guide

This directory is for non-graph batch utilities (segmentation).

## Scripts

- `batch_segmentation.py`
  - Batch vessel segmentation from MRI `.nii.gz` inputs.

## Graph/TC4D Benchmark Scripts (Moved)

Graph-specific benchmark tooling was moved to `graph_extraction/`:

- `graph_extraction/benchmark_4d_vs_tc4d.py`
- `graph_extraction/reduce_4d_vs_tc4d.py`
- `graph_extraction/debug_compare_4d_vs_tc4d.py`

Use `slurm_submit_scripts/submit_4d_vs_tc4d_benchmark.sh` for full benchmark submission.
