#!/bin/bash
#SBATCH --job-name=deepsets-merge
#SBATCH --partition=general
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/deepsets-merge-%j.out
#SBATCH --error=logs/deepsets-merge-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT is required}"

mkdir -p "${REPO_ROOT}/logs" "${OUT_ROOT}"
cd "${REPO_ROOT}"

source ~/.bashrc || true
if command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook -s bash)"
  micromamba activate vanguard
fi

export PYTHONUNBUFFERED=1

python -u merge_deepsets_manifest.py --output-dir "${OUT_ROOT}"
