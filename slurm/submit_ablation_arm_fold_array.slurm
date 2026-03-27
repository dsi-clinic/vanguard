#!/bin/bash
#SBATCH --job-name=ml-ablation-fold
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ml-ablation-fold-%A-%a.out
#SBATCH --error=logs/ml-ablation-fold-%A-%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
CONFIG="${CONFIG:?CONFIG is required}"
FEATURES_CSV="${FEATURES_CSV:?FEATURES_CSV is required}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT is required}"
N_SPLITS="${N_SPLITS:?N_SPLITS is required}"

TASK_ID="${SLURM_ARRAY_TASK_ID}"
ARM_INDEX=$(( TASK_ID / N_SPLITS ))
FOLD_INDEX=$(( TASK_ID % N_SPLITS ))

mkdir -p "${REPO_ROOT}/logs"
cd "${REPO_ROOT}"

source ~/.bashrc || true
if command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook -s bash)"
  micromamba activate vanguard
fi

python -m modeling.run_arm_fold \
  --config "${CONFIG}" \
  --features-csv "${FEATURES_CSV}" \
  --arm-index "${ARM_INDEX}" \
  --fold-index "${FOLD_INDEX}" \
  --out-root "${OUT_ROOT}"
