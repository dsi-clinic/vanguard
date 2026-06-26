#!/usr/bin/env bash
# =============================================================================
# Reproducible entrypoint: run TC4D skeletonization over the resampled-dataset
# vessel segmentations, writing centerlines to the karczmar-lab GPFS workspace.
#
# This is the single documented command for that run. It calls the existing
# tc4d array launcher (graph_extraction/slurm/submit_tc4d_array.sh) with the
# correct GPFS input/output paths and concurrency settings.
#
# The skeletonization pipeline (run_skeleton_processing.py) is CPU-only and
# runs on tier1q. Each case takes roughly 5-20 minutes.
# Missing annotation paths (/net/projects2/vanguard/...) are handled
# gracefully — the pipeline logs them as context and continues.
#
# Outputs per case (under OUTPUT_ROOT/<COHORT>/<CASE>/):
#   <case>_skeleton_4d_exam_mask.npy
#   <case>_skeleton_4d_exam_support_mask.npy
#   <case>_center_manifold_4d_mask.npy
#   <case>_morphometry.json
#   <case>_vessel_coverage_mip.png
#   run_summary.json
#
# Usage:
#   bash scripts/run_resampled_centerlines.sh            # full run (1506 cases)
#   TEST=1 bash scripts/run_resampled_centerlines.sh     # first 5 cases only
#   DRY_RUN=1 bash scripts/run_resampled_centerlines.sh  # print, submit nothing
#
# Overridable inputs/outputs (env vars):
#   INPUT_ROOT  /gpfs/data/karczmar-lab/workspaces/saritbose/resampled-vessel-segmentations
#   OUTPUT_ROOT /gpfs/data/karczmar-lab/workspaces/saritbose/resamples_centerlines_tc4d
#   MAX_CONCURRENT  50
# Environment: micromamba env `vanguard` (activated inside the slurm scripts).
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

INPUT_ROOT="${INPUT_ROOT:-/gpfs/data/karczmar-lab/workspaces/saritbose/resampled-vessel-segmentations}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/gpfs/data/karczmar-lab/workspaces/saritbose/resamples_centerlines_tc4d}"
MAX_CONCURRENT="${MAX_CONCURRENT:-50}"
TEST="${TEST:-0}"
DRY_RUN="${DRY_RUN:-0}"

echo "=== Resampled centerlines (TC4D) run ==="
echo "Project root : ${PROJECT_ROOT}"
echo "Input root   : ${INPUT_ROOT}"
echo "Output root  : ${OUTPUT_ROOT}"
echo "Max concurrent: ${MAX_CONCURRENT}"
echo "Test mode    : ${TEST}"
echo ""

if [[ ! -d "${INPUT_ROOT}" ]]; then
  echo "ERROR: INPUT_ROOT does not exist: ${INPUT_ROOT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
mkdir -p "${PROJECT_ROOT}/logs"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[DRY RUN] Would call: bash graph_extraction/slurm/submit_tc4d_array.sh \\"
  echo "            ${INPUT_ROOT} \\"
  echo "            ${OUTPUT_ROOT} \\"
  echo "            $([ "${TEST}" == "1" ] && echo "--test" || echo "")"
  exit 0
fi

TEST_FLAG=""
if [[ "${TEST}" == "1" ]]; then
  TEST_FLAG="--test"
fi

MAX_CONCURRENT="${MAX_CONCURRENT}" \
  bash "${PROJECT_ROOT}/graph_extraction/slurm/submit_tc4d_array.sh" \
    "${INPUT_ROOT}" \
    "${OUTPUT_ROOT}" \
    "${TEST_FLAG}"
