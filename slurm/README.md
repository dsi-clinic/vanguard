# Modeling Slurm Scripts

These scripts submit pCR modeling experiments from the top-level training and ablation entrypoints.

## Files

- `submit_feature_ablation.slurm`
  - serial feature-block matrix run using `run_ablation_matrix.py`
- `submit_independent_signal_matrix_array.sh`
  - wrapper that submits the cached-table job, the arm/fold array job, and the merge job
- `submit_ablation_arm_fold_array.slurm`
  - per-task worker for one arm/fold pair
- `submit_build_deepsets_dataset.slurm`
  - builds one tumor-local Deep Sets point set per case
- `submit_train_deepsets.slurm`
  - trains the baseline Deep Sets model from a saved manifest
- `submit_deepsets_pipeline.sh`
  - wrapper that submits Deep Sets dataset building first and training second

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

Config note:

- runtime defaults come from [`../config.py`](../config.py)
- the YAML named by `CONFIG` only needs to override values for the specific experiment

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

## Deep Sets Workflow

Typical Deep Sets run:

```bash
cd slurm
CONFIG=../configs/deepsets_ispy2.yaml \
OUT_ROOT=../experiments/deepsets_ispy2_test1 \
./submit_deepsets_pipeline.sh
```

Check or override these before running:

- `CONFIG`
- `OUT_ROOT`
- `PARTITION`
- `BUILD_CPUS`, `BUILD_MEM`, `BUILD_TIME`
- `TRAIN_CPUS`, `TRAIN_MEM`, `TRAIN_TIME`

What the wrapper does:

1. copies the base YAML to `OUT_ROOT/deepsets_runtime_config.yaml`
2. fills in `data_paths.deepsets_manifest_csv`
3. submits `submit_build_deepsets_dataset.slurm`
4. submits `submit_train_deepsets.slurm` with `afterok` dependency on the build job
