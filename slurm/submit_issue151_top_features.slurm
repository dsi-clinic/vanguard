#!/bin/bash
#SBATCH --job-name=issue151-topk
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
  set +u
  eval "$(micromamba shell hook -s bash)"
  micromamba activate vanguard
  set -u
fi

CONFIG="${CONFIG:-configs/top_features_eval.yaml}"
OUTDIR="${OUTDIR:-experiments/issue151_topk_run}"

echo "REPO_ROOT=${REPO_ROOT}"
echo "CONFIG=${CONFIG} OUTDIR=${OUTDIR}"

python modeling/run_top_features_eval.py --config "${CONFIG}" --outdir "${OUTDIR}"

echo "Ranking: ${OUTDIR}/feature_ranking_global.csv"
echo "Summary: ${OUTDIR}/ablation_summary.csv"
