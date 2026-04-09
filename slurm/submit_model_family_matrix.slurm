#!/bin/bash
#SBATCH --job-name=modelfam-matrix
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p "${REPO_ROOT}/logs"
cd "${REPO_ROOT}"

source "${HOME}/.bashrc" || true
if command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook -s bash)"
  micromamba activate vanguard
fi

CONFIG="${CONFIG:-configs/model_family_matrix.yaml}"
OUTDIR="${OUTDIR:-experiments/model_family_matrix_ispy2}"

echo "REPO_ROOT=${REPO_ROOT}"
echo "CONFIG=${CONFIG} OUTDIR=${OUTDIR}"

python run_ablation_matrix.py --config "${CONFIG}" --outdir "${OUTDIR}"
