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

Check or override `CONFIG`, `OUT_ROOT`, and the Slurm resource settings before running.
