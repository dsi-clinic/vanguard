#!/bin/bash
#SBATCH --job-name=vanguard-map-full
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --array=0-179
#SBATCH --output=logs/slurm/vanguard-map-full-%A_%a.out
#SBATCH --error=logs/slurm/vanguard-map-full-%A_%a.err

# Runs map_case (HR TC4D + backfilled-UFAST TC4D merge) for the full cohort, now that
# analysis.backfill_ufast_tc4d has populated work/<exam>/ufast_tc4d/ from the standalone
# ufast-seg tree and the old centerlines/<dataset>/<exam>/ output has been archived to
# centerlines_pre_full_merge_backup_2026-07-28/ (map_case refuses to write into an
# existing output directory by design -- this archive is a precondition, not optional).
#
# Restartable: skips a case whose centerlines output already has a "merge" key in its
# provenance (i.e. a real merge already ran under this backfill), so a retried subset
# after a partial failure does not redo completed work or crash on an existing dir.
#
# Usage:
#   OUTPUT_ROOT=/gpfs/.../preprocessing_out \
#   CASE_LIST=/gpfs/.../case_list_dataset_examid.txt \
#   sbatch --array=0-179 preprocessing/slurm/map_case_array.slurm

set -euo pipefail

: "${OUTPUT_ROOT:?set OUTPUT_ROOT}"
: "${CASE_LIST:?set CASE_LIST, one dataset/exam_id per line}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"
mkdir -p logs/slurm

set +u
source "${HOME}/.bashrc" || true
if command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook -s bash)"
  micromamba activate vanguard
fi
set -u

export PYTHONUNBUFFERED=1

LINE="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${CASE_LIST}")"
DATASET="${LINE%%/*}"
EXAM_ID="${LINE#*/}"
echo "task ${SLURM_ARRAY_TASK_ID}: dataset=${DATASET} exam_id=${EXAM_ID}"

PROVENANCE="${OUTPUT_ROOT}/work/${EXAM_ID}/preprocessing_provenance.json"
ALREADY_MERGED="$(python3 -c "
import json
p = json.load(open('${PROVENANCE}'))
print('yes' if 'merge' in p else 'no')
")"

if [[ "${ALREADY_MERGED}" == "yes" ]]; then
  echo "skip: ${EXAM_ID} already has a real merge result"
  exit 0
fi

python -m preprocessing.pipeline map \
  --exam-id "${EXAM_ID}" \
  --output-root "${OUTPUT_ROOT}"

echo "done: ${EXAM_ID}"
