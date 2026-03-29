#!/bin/bash
#SBATCH --job-name=deepsets-train
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/deepsets-train-%j.out
#SBATCH --error=logs/deepsets-train-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
CONFIG="${CONFIG:?CONFIG is required}"
OUTDIR="${OUTDIR:?OUTDIR is required}"

mkdir -p "${REPO_ROOT}/logs" "${OUTDIR}"
cd "${REPO_ROOT}"

source ~/.bashrc || true
if command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook -s bash)"
  micromamba activate vanguard
fi

export PYTHONUNBUFFERED=1

python -u train_deepsets.py \
  --config "${CONFIG}" \
  --outdir "${OUTDIR}"
