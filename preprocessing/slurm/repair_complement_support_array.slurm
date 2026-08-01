#!/bin/bash
#SBATCH --job-name=vanguard-repair-complement-support
#SBATCH --account=karczmar-lab
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-179
#SBATCH --output=logs/slurm/vanguard-repair-complement-support-%A_%a.out
#SBATCH --error=logs/slurm/vanguard-repair-complement-support-%A_%a.err

# For cases that already ran complement before support propagation existed:
# re-derive SegVessel's vessel-width mask for the accepted branches and union it
# into the saved support mask. Does not rewrite the centerline. Skips cases that
# already have complement.support_propagated=true (including fresh complement
# runs from the fixed code).
#
# Usage:
#   OUTPUT_ROOT=/gpfs/.../preprocessing_out \
#   CASE_LIST=/gpfs/.../case_list_dataset_examid.txt \
#   sbatch --array=0-179 preprocessing/slurm/repair_complement_support_array.slurm

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
NEED_REPAIR="$(python3 -c "
import json
from pathlib import Path
p = Path('${PROVENANCE}')
if not p.exists():
    print('no_provenance')
else:
    data = json.loads(p.read_text())
    comp = data.get('complement')
    if not isinstance(comp, dict):
        print('no_complement')
    elif comp.get('support_propagated'):
        print('already_done')
    else:
        print('yes')
")"

if [[ "${NEED_REPAIR}" != "yes" ]]; then
  echo "skip: ${EXAM_ID} (${NEED_REPAIR})"
  exit 0
fi

python -m preprocessing.pipeline repair-complement-support \
  --exam-id "${EXAM_ID}" \
  --output-root "${OUTPUT_ROOT}"

echo "done: ${EXAM_ID}"
