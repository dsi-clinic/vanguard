#!/bin/bash
#SBATCH --job-name=her2-phase0-xgb
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

# Phase 0e: XGBoost vessel_all on the n=68 HER2 intersection cohort using
# the canonical fold map at data/fold_map_her2.csv. Apples-to-apples
# comparator for the HER2 Deep Sets work; gates whether Phase 2a attention
# pooling is the right next experiment (XGB ~0.67 on n=68) or whether the
# project pivots to a cohort-size story (XGB ~0.60 on n=68).
#
# Submit with:
#   sbatch slurm/submit_her2_phase0_xgb.slurm
#
# Override defaults with sbatch --export, e.g.:
#   sbatch --export=ALL,OUTDIR=experiments/her2_phase0/xgb_vessel_all_v2 \
#          slurm/submit_her2_phase0_xgb.slurm

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

FEATURES="${FEATURES:-experiments/independent_signal_q3_array/features_full_labeled.csv}"
FOLD_MAP="${FOLD_MAP:-data/fold_map_her2.csv}"
OUTDIR="${OUTDIR:-experiments/her2_phase0/xgb_vessel_all}"
LABEL_COL="${LABEL_COL:-pcr}"

echo "REPO_ROOT=${REPO_ROOT}"
echo "FEATURES=${FEATURES}"
echo "FOLD_MAP=${FOLD_MAP}"
echo "OUTDIR=${OUTDIR}"
echo "LABEL_COL=${LABEL_COL}"

if [[ ! -f "${FEATURES}" ]]; then
  echo "Features CSV not found: ${FEATURES}" >&2
  exit 2
fi
if [[ ! -f "${FOLD_MAP}" ]]; then
  echo "Fold map not found: ${FOLD_MAP}" >&2
  exit 2
fi

# XGBoost wheel from pypi (not the conda-forge build) — needed because the
# vanguard env's pinned conda channel did not ship xgboost. Skip silently if
# already installed.
if ! python -c "import xgboost" 2>/dev/null; then
  echo "Installing xgboost into the activated env..."
  pip install --quiet xgboost
fi

PYTHONPATH="${REPO_ROOT}" python scripts/her2_xgb_baseline.py \
    --features "${FEATURES}" \
    --fold-map "${FOLD_MAP}" \
    --label-col "${LABEL_COL}" \
    --outdir "${OUTDIR}"

echo "Predictions: ${OUTDIR}/predictions.csv"
echo "Metrics:     ${OUTDIR}/metrics.json"
