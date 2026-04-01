#!/bin/bash
#SBATCH --job-name=deepsets-job
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/deepsets-%x-%j.out
#SBATCH --error=logs/deepsets-%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
MODE="${MODE:?MODE is required}"

mkdir -p "${REPO_ROOT}/logs"
cd "${REPO_ROOT}"

source ~/.bashrc || true
if command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook -s bash)"
  micromamba activate vanguard
fi

export PYTHONUNBUFFERED=1

case "${MODE}" in
  build)
    CONFIG="${CONFIG:?CONFIG is required for MODE=build}"
    OUT_ROOT="${OUT_ROOT:?OUT_ROOT is required for MODE=build}"
    NUM_SHARDS="${NUM_SHARDS:?NUM_SHARDS is required for MODE=build}"
    SHARD_INDEX="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required for MODE=build}"
    mkdir -p "${OUT_ROOT}"
    python -u build_deepsets_dataset.py \
      --config "${CONFIG}" \
      --output-dir "${OUT_ROOT}" \
      --num-shards "${NUM_SHARDS}" \
      --shard-index "${SHARD_INDEX}"
    ;;
  merge)
    OUT_ROOT="${OUT_ROOT:?OUT_ROOT is required for MODE=merge}"
    mkdir -p "${OUT_ROOT}"
    python -u merge_deepsets_manifest.py --output-dir "${OUT_ROOT}"
    ;;
  train)
    CONFIG="${CONFIG:?CONFIG is required for MODE=train}"
    OUTDIR="${OUTDIR:?OUTDIR is required for MODE=train}"
    mkdir -p "${OUTDIR}"
    python -u train_deepsets.py \
      --config "${CONFIG}" \
      --outdir "${OUTDIR}"
    ;;
  *)
    echo "Unsupported MODE: ${MODE}" >&2
    exit 2
    ;;
esac
