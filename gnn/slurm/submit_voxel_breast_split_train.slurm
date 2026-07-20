#!/usr/bin/env bash
#SBATCH --job-name=voxel-breast-split-train
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-8
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# Trains the three matched arms (mixed baseline / single breast / native
# unilateral only) of the harmonized single-breast dataset minimal
# experiment (Phase 5, LAB_NOTEBOOK.md 2026-07-14/15), 3 seeds each (9
# array tasks total), against the caches built by
# gnn/slurm/submit_voxel_breast_split_build.slurm. All 3 arms share the
# same frozen folds (split_mode: predefined); only model-init randomness
# varies across seeds (--random-state, see gnn/train.py).
#
# Array index -> (arm, seed): index = arm_idx * 3 + seed_idx.
#
# Usage:
#   sbatch gnn/slurm/submit_voxel_breast_split_train.slurm
#   SEEDS="42 142 242" sbatch ...   # override seeds

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
mkdir -p "${REPO_ROOT}/logs"
cd "${REPO_ROOT}"

set +u
source "${HOME}/.bashrc" || true
if command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook -s bash)"
  micromamba activate vanguard
fi
set -u

export PYTHONUNBUFFERED=1

CONFIGS="${CONFIGS:-configs/gnn_voxel_mixed_baseline.yaml configs/gnn_voxel_single_breast.yaml configs/gnn_voxel_unilateral_only.yaml}"
read -r -a CONFIG_ARR <<< "${CONFIGS}"
SEEDS="${SEEDS:-42 142 242}"
read -r -a SEED_ARR <<< "${SEEDS}"

NUM_SEEDS="${#SEED_ARR[@]}"
ARM_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_SEEDS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_SEEDS ))
CONFIG="${CONFIG_ARR[${ARM_IDX}]}"
SEED="${SEED_ARR[${SEED_IDX}]}"

ARM_NAME="$(basename "${CONFIG}" .yaml)"
# Explicit, collision-free outdir: resolve_run_output_dir's default naming
# is <name>_<second-granularity-timestamp>, and parallel array tasks for
# the same arm's different seeds can easily start within the same second,
# which would make two seeds silently overwrite the same output directory.
OUTDIR="experiments/harmonized_single_breast_v1/${ARM_NAME}/seed_${SEED}"

echo "Training arm config: ${CONFIG}, seed=${SEED} (array task ${SLURM_ARRAY_TASK_ID}) -> ${OUTDIR}"

python -u -m gnn.train --config "${CONFIG}" --random-state "${SEED}" --outdir "${OUTDIR}"
