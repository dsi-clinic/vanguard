#!/bin/bash
#SBATCH --job-name=issue118-baseline
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
  # Some conda activate.d scripts read optional vars (e.g. MKL_INTERFACE_LAYER)
  # and fail under `set -u`; relax nounset only around activation.
  set +u
  eval "$(micromamba shell hook -s bash)"
  micromamba activate vanguard
  set -u
fi

CONFIG="${CONFIG:-configs/issue118_baseline_arms.yaml}"
OUTDIR="${OUTDIR:-experiments/issue118_baseline_arms}"

echo "REPO_ROOT=${REPO_ROOT}"
echo "CONFIG=${CONFIG} OUTDIR=${OUTDIR}"

python run_ablation_matrix.py --config "${CONFIG}" --outdir "${OUTDIR}"

echo "Summary: ${OUTDIR}/ablation_summary.csv"
