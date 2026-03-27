#!/bin/bash
#SBATCH --job-name=ml-feature-ablation
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/ml-feature-ablation-%j.out
#SBATCH --error=logs/ml-feature-ablation-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$PWD}"
CONFIG="${CONFIG:-configs/feature_ablation_ispy2.yaml}"
OUTDIR="${OUTDIR:-experiments/feature_ablation_ispy2_q2}"

mkdir -p "${REPO_ROOT}/logs"
cd "${REPO_ROOT}"

source ~/.bashrc || true
if command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook -s bash)"
  micromamba activate vanguard
fi

python run_ablation_matrix.py \
  --config "${CONFIG}" \
  --outdir "${OUTDIR}"
