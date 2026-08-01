#!/bin/bash
#SBATCH --job-name=vanguard-complement-full
#SBATCH --account=karczmar-lab
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-179
#SBATCH --output=logs/slurm/vanguard-complement-full-%A_%a.out
#SBATCH --error=logs/slurm/vanguard-complement-full-%A_%a.err

# Runs the (latest) MATLAB SegVessel/Jerman complement stage over the full
# 180-case cohort already sitting in OUTPUT_ROOT (map_case already ran for
# all of them). Restartable/idempotent: skips a case whose provenance already
# has a "complement" key. Older complement runs that updated the centerline but
# not the support mask should use repair-complement-support (see
# repair_complement_support_array.slurm), not a silent re-run. To redo
# complement from scratch, restore the merge-only skeleton/support, remove the
# complement key from provenance, and delete the merge-only backup files first.
#
# Usage:
#   OUTPUT_ROOT=/gpfs/.../preprocessing_out \
#   CASE_LIST=/gpfs/.../case_list_dataset_examid.txt \
#   sbatch --array=0-179 preprocessing/slurm/complement_case_array.slurm

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
ALREADY_DONE="$(python3 -c "
import json
p = json.load(open('${PROVENANCE}'))
print('yes' if 'complement' in p else 'no')
")"

if [[ "${ALREADY_DONE}" == "yes" ]]; then
  echo "skip: ${EXAM_ID} already has a complement result"
  exit 0
fi

python -m preprocessing.pipeline complement \
  --exam-id "${EXAM_ID}" \
  --output-root "${OUTPUT_ROOT}"

echo "done: ${EXAM_ID}"
