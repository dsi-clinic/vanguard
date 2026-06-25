#!/usr/bin/env bash
# =============================================================================
# Submit wrapper for the resampling job array.
#
# It does three small, login-node-safe things:
#   1. counts how many cohort cases exist (a quick directory listing),
#   2. works out how many array tasks are needed at CASES_PER_TASK each,
#   3. calls sbatch with the right --array range and a concurrency cap.
#
# Usage:
#   bash scripts/slurm/submit_resample_array.sh            # real run
#   DRY_RUN=1 bash scripts/slurm/submit_resample_array.sh  # print the plan only
#
# Tunables (environment variables):
#   CASES_PER_TASK  cases handled per array task            (default 16)
#   CONCURRENCY     max array tasks running at once (the %)  (default 24)
#   DATA_ROOT       input image root                         (default: annawoodard scratch)
#   OUTPUT_ROOT     where resampled output goes              (default: your scratch)
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/ess/scratch/scratch1/annawoodard/MAMA-MIA-syn60868042/images}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/ess/scratch/scratch1/t-9sbose/resampled_segmentation}"
CASES_PER_TASK="${CASES_PER_TASK:-16}"
CONCURRENCY="${CONCURRENCY:-24}"

# Count case folders belonging to our four cohorts (lightweight `ls`, no data read).
NUM_CASES="$(ls -1 "${DATA_ROOT}" | grep -E '^(DUKE|ISPY1|ISPY2|NACT)_' | wc -l)"
# Number of array tasks = ceil(NUM_CASES / CASES_PER_TASK).
NUM_TASKS=$(( (NUM_CASES + CASES_PER_TASK - 1) / CASES_PER_TASK ))
LAST_INDEX=$(( NUM_TASKS - 1 ))

echo "Project root  : ${PROJECT_ROOT}"
echo "Data root     : ${DATA_ROOT}"
echo "Output root   : ${OUTPUT_ROOT}"
echo "Cases found   : ${NUM_CASES}"
echo "Cases/task    : ${CASES_PER_TASK}"
echo "Array tasks   : ${NUM_TASKS}  (indices 0-${LAST_INDEX})"
echo "Concurrency   : up to ${CONCURRENCY} tasks at once  (= ${CONCURRENCY} x 8 = $((CONCURRENCY*8)) cores)"
echo

SBATCH_CMD=(sbatch
  --array="0-${LAST_INDEX}%${CONCURRENCY}"
  --export="ALL,DATA_ROOT=${DATA_ROOT},OUTPUT_ROOT=${OUTPUT_ROOT},CASES_PER_TASK=${CASES_PER_TASK}"
  "${PROJECT_ROOT}/scripts/slurm/submit_resample_array.slurm"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[DRY RUN] Would submit:"
  printf '  %q ' "${SBATCH_CMD[@]}"; echo
  exit 0
fi

mkdir -p "${PROJECT_ROOT}/logs"
"${SBATCH_CMD[@]}"
