# Modeling Slurm Scripts

These scripts submit pCR modeling experiments from the top-level training and ablation entrypoints.

## Files

- `submit_feature_ablation.slurm`
  - serial feature-block matrix run using `run_ablation_matrix.py`
- `submit_independent_signal_matrix_array.sh`
  - wrapper that submits the cached-table job, the arm/fold array job, and the merge job
- `submit_ablation_arm_fold_array.slurm`
  - per-task worker for one arm/fold pair

## Recommended Entry Point

```bash
cd slurm
./submit_independent_signal_matrix_array.sh
```

Check or override these before running:

- `CONFIG`
- `OUT_ROOT`
- `PARTITION`
- `CACHE_CPUS`, `CACHE_MEM`, `CACHE_TIME`
- `FOLD_CPUS`, `FOLD_MEM`, `FOLD_TIME`
- `MERGE_CPUS`, `MERGE_MEM`, `MERGE_TIME`

Typical use for the independent-signal matrix:

```bash
cd slurm
CONFIG=../configs/independent_signal.yaml \
OUT_ROOT=../experiments/independent_signal_q3_array_rerun1 \
./submit_independent_signal_matrix_array.sh
```

To test different arms:

1. edit `ablation_arms` in `../configs/independent_signal.yaml`
2. submit again to a fresh `OUT_ROOT`
