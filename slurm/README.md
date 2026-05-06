# Modeling Slurm Scripts

These scripts submit pCR modeling experiments from the top-level training and ablation entrypoints.

## Files

- `submit_feature_ablation.slurm`
  - serial feature-block matrix run using `run_ablation_matrix.py`
- `submit_model_family_matrix.slurm`
  - Issue #116 matrix: `model_families` × `ablation_arms` via `run_ablation_matrix.py`
- `submit_model_family_robustness.slurm`
  - Issue #117: top families × `split_mode_matrix` via `run_ablation_matrix.py`
- `submit_issue118_baseline_arms.slurm`
  - Issue #118: five baseline arms × (`lr`, `xgb`) in one `run_ablation_matrix.py` run
- `submit_independent_signal_matrix_array.sh`
  - wrapper that submits the cached-table job, the arm/fold array job, and the merge job
- `submit_ablation_arm_fold_array.slurm`
  - per-task worker for one arm/fold pair
- `submit_deepsets_pipeline.sh`
  - user-facing wrapper that submits the full Deep Sets workflow
- `submit_deepsets_build_merge.sh`
  - dataset build **only**: serial build (`BUILD_SHARDS=1`) **or** sharded build array plus manifest merge (**no training**); defaults `OUT_ROOT` to `results/deepsets` so it matches notebooks that expect `results/deepsets/deepsets_manifest.csv`
- `deepsets_job.slurm`
  - one parameterized job script used for Deep Sets build (`build`), single-process build (`build-single`), merge, and train stages

## Slurm usage (robust paths)

Submit from the **repository root** so `SLURM_SUBMIT_DIR` points at the repo (scripts `cd` there and write logs under `logs/`).

```bash
cd /path/to/vanguard
sbatch slurm/submit_model_family_matrix.slurm
```

Override config/output with env vars (each script echoes what it uses):

- `CONFIG` — path to YAML under the repo, e.g. `configs/model_family_matrix.yaml`
- `OUTDIR` — experiment output directory, e.g. `experiments/my_run`

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

## Model-Family Matrix (#116)

```bash
cd ~/vanguard
CONFIG=configs/model_family_matrix.yaml \
OUTDIR=experiments/model_family_matrix_ispy2 \
sbatch slurm/submit_model_family_matrix.slurm
```

Primary output:

- `OUTDIR/ablation_summary.csv` (includes `auc_std`, `ap_mean` / `ap_std` when folds support AP)
- `OUTDIR/ablation_matrix_meta.yaml`

## Model-Family Robustness (#117)

```bash
cd ~/vanguard
CONFIG=configs/model_family_robustness.yaml \
OUTDIR=experiments/model_family_robustness_ispy2 \
sbatch slurm/submit_model_family_robustness.slurm
```

Primary outputs:

- `OUTDIR/ablation_summary.csv`
- `OUTDIR/ablation_subtype_summary.csv` (when `export_subtype_summary: true` in config)

## Issue #118 — Baseline vessel arms (`lr` + `xgb`)

Single ablation run (both families in one table):

```bash
cd ~/vanguard
CONFIG=configs/issue118_baseline_arms.yaml \
OUTDIR=experiments/issue118_baseline_arms \
sbatch slurm/submit_issue118_baseline_arms.slurm
```

Output: `OUTDIR/ablation_summary.csv`

Details: `docs/issue118_baseline_comparison.md`

## Deep Sets Workflow

Typical Deep Sets run:

```bash
cd slurm
CONFIG=../configs/deepsets_ispy2.yaml \
OUT_ROOT=../experiments/deepsets_ispy2_test1 \
./submit_deepsets_pipeline.sh
```

### Dataset build only (no training; SLURM)

Use when you only need `deepsets_manifest.csv` and `.pt` case sets—for example notebooks that expect `results/deepsets/deepsets_manifest.csv`.

Submit from **repository root** (so paths and `logs/` behave like other wrappers):

```bash
cd ~/vanguard
./slurm/submit_deepsets_build_merge.sh
```

Defaults:

- `CONFIG=${REPO_ROOT}/configs/deepsets_ispy2.yaml`
- `OUT_ROOT=${REPO_ROOT}/results/deepsets` (absolute path after normalization)
- `BUILD_SHARDS=8` → Slurm job array shards + dependent merge step
- **`BUILD_SHARDS=1`** → one long `MODE=build-single` job (`build_deepsets_dataset.py` with default `--num-shards 1`, manifest written straight to `--output-dir`)

Override examples:

```bash
cd ~/my-clone-of-vanguard
PARTITION=general BUILD_TIME=48:00:00 BUILD_CPUS=8 BUILD_MEM=64G \
OUT_ROOT=/abs/path/results/deepsets \
./slurm/submit_deepsets_build_merge.sh
```

Low–medium parallelism instead of eight shards:

```bash
BUILD_SHARDS=4 ./slurm/submit_deepsets_build_merge.sh
```

Check or override these before running:

- `CONFIG`
- `OUT_ROOT`
- `PARTITION`
- `BUILD_CPUS`, `BUILD_MEM`, `BUILD_TIME`
- `TRAIN_CPUS`, `TRAIN_MEM`, `TRAIN_TIME`

Deep Sets inclusion-rule controls (in `model_params`) now include:

- `deepsets_inclusion_rule` (default `local_radius_with_fallback`)
- `deepsets_compare_inclusion_rules` (compact comparison set written to build stats)

Build-stage outputs include `OUT_ROOT/inclusion_rule_summary.csv` with:

- `cases_written`
- `cases_skipped`
- `fallback_fraction`
- `num_points_median`
- `num_points_range`

Default-selection notebook: `analysis/deepsets_issue121_notebook.ipynb`

What the wrapper does:

1. copies the base YAML to `OUT_ROOT/deepsets_runtime_config.yaml`
2. fills in `data_paths.deepsets_manifest_csv`
3. submits a build array through `deepsets_job.slurm` with `MODE=build`
4. submits a manifest-merge job through `deepsets_job.slurm` with `MODE=merge`
5. submits a training job through `deepsets_job.slurm` with `MODE=train`
