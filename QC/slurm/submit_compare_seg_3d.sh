#!/usr/bin/env bash
# Submit the 3D segmentation comparison job to SLURM.
#
# Usage:
#   bash QC/slurm/submit_compare_seg_3d.sh
#
# Optional env var overrides (set before calling):
#   OUT_DIR=~/my_output bash QC/slurm/submit_compare_seg_3d.sh
#   SEED=7 STRIDE=4 bash QC/slurm/submit_compare_seg_3d.sh
#
# Output (after ~30 min):
#   {OUT_DIR}/{case_id}_3d_comparison.png   — 4-panel static PNG
#   {OUT_DIR}/{case_id}_3d_comparison.mp4   — rotating animation
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

mkdir -p "${REPO_ROOT}/logs"

JOB_ID=$(
  REPO_ROOT="${REPO_ROOT}" \
  OUT_DIR="${OUT_DIR:-${HOME}/vanguard_qc_pngs}" \
  SEED="${SEED:-42}" \
  STRIDE="${STRIDE:-3}" \
  N_PER_COHORT="${N_PER_COHORT:-2}" \
  N_FRAMES="${N_FRAMES:-60}" \
  sbatch --parsable "${SCRIPT_DIR}/compare_seg_3d.slurm"
)

echo "Submitted job: ${JOB_ID}"
echo "Monitor:  squeue -j ${JOB_ID}"
echo "Log:      ${REPO_ROOT}/logs/seg-3d-compare-${JOB_ID}.out"
echo "Output:   ${OUT_DIR:-${HOME}/vanguard_qc_pngs}"
