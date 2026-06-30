#!/usr/bin/env bash
#
# Stage 1 of the "resampled segmentations -> pCR" comparison.
#
# Re-runs the previous group's independent-signal matrix (6 feature-block arms,
# 5-fold nested-CV logistic regression, ISPY2) using the RESAMPLED-segmentation
# centerlines, by handing configs/independent_signal_randi_resampled.yaml to the
# existing matrix launcher. All scheduling/array logic lives in
# submit_independent_signal_matrix_array.sh; this wrapper just fixes CONFIG and a
# scratch OUT_ROOT so the run is reproducible from one command.
#
# PREREQUISITE: Stage 0 must have populated *_tumor_graph_features.json in the
# resampled centerlines (graph + kinematic blocks). Run first:
#   bash graph_extraction/slurm/submit_resampled_tumor_graph_features.sh
#
# After it finishes, compare the resulting
#   ${OUT_ROOT}/ablation_summary.csv
# against the published baseline in
#   results/independent_signal_q3_summary.csv  (0.572 -> 0.596).
#
# USAGE
#   bash slurm/submit_resampled_independent_signal.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export CONFIG="${CONFIG:-${REPO_ROOT}/configs/independent_signal_randi_resampled.yaml}"
export OUT_ROOT="${OUT_ROOT:-/ess/scratch/scratch1/t-9sbose/vanguard_experiments/independent_signal_resampled_ispy2}"

echo "Stage 1: independent-signal matrix on RESAMPLED ISPY2 centerlines"
echo "  config   : ${CONFIG}"
echo "  out_root : ${OUT_ROOT}"
echo

exec bash "${SCRIPT_DIR}/submit_independent_signal_matrix_array.sh"
