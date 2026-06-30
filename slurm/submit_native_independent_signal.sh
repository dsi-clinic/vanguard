#!/usr/bin/env bash
#
# Control run for the resampling investigation (Hypothesis 1 test).
#
# Runs the same 6-arm independent-signal ablation as the resampled run but
# using the NATIVE (non-resampled) vessel centerlines that are available
# locally on this cluster. Produces Run B in the three-way comparison:
#
#   A. Old code  + native data → baseline (results/independent_signal_q3_summary.csv)
#   B. THIS RUN: current code + native data
#   C. Current code + resampled data → independent_signal_randi_resampled.yaml
#
# Interpreting the results:
#   B ≈ A → code version is not a confounder; C-vs-A delta is purely resampling
#   B ≠ A → code changes (feature set, selection logic) also shift AUC
#   C-vs-B → pure resampling effect, free of code-version confounding
#
# USAGE
#   PARTITION=tier1q bash slurm/submit_native_independent_signal.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export CONFIG="${CONFIG:-${REPO_ROOT}/configs/independent_signal_randi_native.yaml}"
export OUT_ROOT="${OUT_ROOT:-/ess/scratch/scratch1/t-9sbose/vanguard_experiments/independent_signal_native_ispy2}"

echo "Hypothesis 1 control: independent-signal matrix on NATIVE ISPY2 centerlines"
echo "  config   : ${CONFIG}"
echo "  out_root : ${OUT_ROOT}"
echo

exec bash "${SCRIPT_DIR}/submit_independent_signal_matrix_array.sh"
