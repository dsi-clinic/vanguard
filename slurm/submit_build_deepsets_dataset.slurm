#!/bin/bash
#SBATCH --job-name=deepsets-build
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/deepsets-build-%j.out
#SBATCH --error=logs/deepsets-build-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
CONFIG="${CONFIG:?CONFIG is required}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT is required}"

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
  --output-dir "${OUT_ROOT}"
