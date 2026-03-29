#!/bin/bash
#SBATCH --job-name=deepsets-build
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/deepsets-build-%A-%a.out
#SBATCH --error=logs/deepsets-build-%A-%a.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
CONFIG="${CONFIG:?CONFIG is required}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT is required}"
NUM_SHARDS="${NUM_SHARDS:?NUM_SHARDS is required}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

mkdir -p "${REPO_ROOT}/logs" "${OUT_ROOT}"
cd "${REPO_ROOT}"

source ~/.bashrc || true
if command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook -s bash)"
  micromamba activate vanguard
fi

export PYTHONUNBUFFERED=1

python -u build_deepsets_dataset.py \
  --config "${CONFIG}" \
  --output-dir "${OUT_ROOT}" \
  --num-shards "${NUM_SHARDS}" \
  --shard-index "${SHARD_INDEX}"
