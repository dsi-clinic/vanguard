#!/bin/bash
#SBATCH --job-name=graph-ablation
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-${REPO_ROOT:-$PWD}}"
CONFIG="${CONFIG:-configs/clinical_graph_ablation.yaml}"
OUTDIR="${OUTDIR:-experiments/clinical_graph_ablation}"

mkdir -p "${REPO_ROOT}/logs"
cd "${REPO_ROOT}"

source ~/.bashrc || true
if command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook -s bash)"
  set +u  # conda/MKL activate scripts may reference unbound variables
  micromamba activate vanguard
  set -u
fi

echo "=== Graph feature ablation with clinical baseline ==="
echo "Config: ${CONFIG}"
echo "Output: ${OUTDIR}"
echo ""

python scripts/run_ablation_matrix.py \
  --config "${CONFIG}" \
  --outdir "${OUTDIR}"
