#!/usr/bin/env bash
#
# Native-spacing run with LOCKED SPLIT — isolates the resampling effect.
#
# Runs the same 6-arm independent-signal ablation (ISPY2, nested-CV LR) as the
# resampled run, but uses:
#   - NATIVE-spacing centerlines (same as the old Q3 baseline)
#   - The exact same train/test fold assignment as the completed resampled run
#
# Because the split is locked, the only variable that differs between this run
# and the resampled run is the centerline source. Any AUC delta is attributable
# solely to resampling.
#
# PREREQUISITES (run in order before submitting this job)
# -------------------------------------------------------
# 1. Resampled run must be complete:
#      /ess/scratch/scratch1/t-9sbose/vanguard_experiments/
#      independent_signal_resampled_ispy2/runs/*/predictions.csv
#
# 2. Extract and save the locked split CSV (head node, ~seconds):
#      cd /ess/home/home1/t-9sbose/vanguard
#      micromamba run -n vanguard python scripts/extract_resampled_splits.py
#      # Output: /ess/scratch/scratch1/t-9sbose/vanguard_experiments/
#      #         locked_split_resampled_ispy2.csv
#
# USAGE
#   cd /ess/home/home1/t-9sbose/vanguard
#   bash slurm/submit_native_locked_split.sh
#
# OUTPUTS
#   /ess/scratch/scratch1/t-9sbose/vanguard_experiments/
#   independent_signal_native_locked_ispy2/ablation_summary.csv
#
# AFTER COMPLETION
#   micromamba run -n vanguard python scripts/compare_locked_split_auc.py
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

LOCKED_SPLIT_CSV="/ess/scratch/scratch1/t-9sbose/vanguard_experiments/locked_split_resampled_ispy2.csv"

if [[ ! -f "${LOCKED_SPLIT_CSV}" ]]; then
  echo "ERROR: locked split CSV not found at ${LOCKED_SPLIT_CSV}"
  echo "Run first: micromamba run -n vanguard python scripts/extract_resampled_splits.py"
  exit 1
fi

export CONFIG="${CONFIG:-${REPO_ROOT}/configs/native_locked_split_ispy2.yaml}"
export OUT_ROOT="${OUT_ROOT:-/ess/scratch/scratch1/t-9sbose/vanguard_experiments/independent_signal_native_locked_ispy2}"

echo "Native locked-split ablation: ISPY2, native centerlines, locked folds"
echo "  config          : ${CONFIG}"
echo "  out_root        : ${OUT_ROOT}"
echo "  locked split    : ${LOCKED_SPLIT_CSV}"
echo

exec bash "${SCRIPT_DIR}/submit_independent_signal_matrix_array.sh"
