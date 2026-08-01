#!/bin/bash
#SBATCH --job-name=combined-tc4d
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=11:30:00
#SBATCH --output=logs/slurm/combined-tc4d-%j.out
#SBATCH --error=logs/slurm/combined-tc4d-%j.err

# Exploratory: run ONE unmodified TC4D call over a single combined HR+UFAST
# phase sequence (see analysis/skeletonize_combined_hr_ufast.py docstring for
# phase order and why this is NOT the supported route -- HR TC4D + skeleton
# mapping is). Same CPU/time budget as the separate-route TC4D stages
# (preprocessing/slurm/tc4d_map_case.slurm, analysis/submit_skeletonize_ufast_vessels.slurm):
# the combined stack is longer in time but TC4D's cost is dominated by
# per-frame candidate extraction, not stack length.
#
# Requires both routes' vessel predictions to already exist for each exam:
# HR under PREPROCESSING_ROOT/work/<exam-id>/hr_vessel_predictions/, UFAST-direct
# under OUTPUT_ROOT/<exam-id>/ufast_vessel_predictions/ (analysis/segment_ufast_vessels.py).
#
# Submit from the repo root, either a fixed exam list:
#   OUTPUT_ROOT=/gpfs/data/karczmar-lab/workspaces/saritbose/ufast-seg \
#   PREPROCESSING_ROOT=/gpfs/data/karczmar-lab/workspaces/saritbose/preprocessing_out \
#   EXAM_IDS="exam1 exam2" \
#     sbatch analysis/submit_skeletonize_combined_hr_ufast.slurm
# or the full cohort (every exam with both routes' vessel predictions):
#   OUTPUT_ROOT=... PREPROCESSING_ROOT=... FULL_COHORT=1 \
#     sbatch analysis/submit_skeletonize_combined_hr_ufast.slurm

set -eo pipefail

: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the UFAST segmentation output root}"
: "${PREPROCESSING_ROOT:?Set PREPROCESSING_ROOT to the paired preprocessing output root}"
if [[ "${FULL_COHORT:-0}" != "1" ]]; then
  : "${EXAM_IDS:?Set EXAM_IDS to a space-separated list of exam ids, or set FULL_COHORT=1}"
fi

REPO_ROOT="${REPO_ROOT:-$PWD}"
cd "${REPO_ROOT}"
mkdir -p logs/slurm

source ~/.bashrc || true
if command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook -s bash)"
  micromamba activate vanguard
fi

export PYTHONUNBUFFERED=1

EXAM_ARGS=()
if [[ "${FULL_COHORT:-0}" == "1" ]]; then
  EXAM_ARGS+=(--full-cohort)
else
  for exam_id in ${EXAM_IDS}; do
    EXAM_ARGS+=(--exam-id "${exam_id}")
  done
fi

python -u -m analysis.skeletonize_combined_hr_ufast \
  --preprocessing-root "${PREPROCESSING_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  "${EXAM_ARGS[@]}"
