#!/usr/bin/env bash
#SBATCH --job-name=tabular-bar-uchicago-floored
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Phase 1.1 tabular bar: the number the UChicago vessel-GNN must beat.
# Fits logistic-regression + XGBoost on per-case vessel-summary features
# (mean/std/q10/q50/q90 of the 7 node kinetics + tte-no-arrival fraction +
# node/edge counts) read read-only from the floored voxel cache, under the
# frozen predefined folds. Reports POOLED OOF AUC over all 179 cases with a
# bootstrap 95% CI (the plan's headline metric, comparable to the GNN ~0.61).
#
# Usage (cache already built -- no build dependency):
#   sbatch gnn/slurm/submit_tabular_bar_uchicago_floored.slurm

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
mkdir -p "${REPO_ROOT}/logs"
cd "${REPO_ROOT}"

set +u
source "${HOME}/.bashrc" || true
if command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook -s bash)"
  micromamba activate vanguard
fi
set -u

export PYTHONUNBUFFERED=1

python -m tabular.gnn_feature_baseline \
  --config configs/uchicago_gnn_tier_baseline_floored.yaml \
  --out-dir experiments/uchicago_tabular_bar_floored
