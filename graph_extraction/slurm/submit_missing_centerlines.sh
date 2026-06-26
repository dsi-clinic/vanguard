#!/usr/bin/env bash
# Submit skeletonization jobs for studies that have segmentation but no centerlines.
# Uses inventory.py to find missing cases, then submits via submit_tc4d_array.slurm.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

INPUT_ROOT="${1:-/scratch/t-9sbose/vessel_segmentations}"
OUTPUT_ROOT="${2:-/scratch/t-9sbose/centerlines_tc4d/studies}"
PARTITION="${PARTITION:-tier1q}"
MAX_CONCURRENT="${MAX_CONCURRENT:-50}"

mkdir -p "${REPO_ROOT}/logs" "${OUTPUT_ROOT}"

# Build study list: seg=yes, centerlines=no
STUDY_LIST="$(mktemp "${REPO_ROOT}/logs/missing-centerlines.XXXXXX.txt")"

python3 - <<PY > "${STUDY_LIST}"
import csv, sys
from io import StringIO
from pathlib import Path

sys.path.insert(0, "${REPO_ROOT}/scripts")
from inventory import discover_studies, build_record

input_root = Path("${INPUT_ROOT}")
output_root = Path("${OUTPUT_ROOT}")

pairs = discover_studies(input_root)
for cohort, case_id in pairs:
    r = build_record(cohort, case_id, input_root, output_root)
    if r.segmentation and not r.centerlines:
        print(f"{cohort}/{case_id}")
PY

if [[ ! -s "${STUDY_LIST}" ]]; then
  echo "No studies need centerlines. Nothing to submit." >&2
  exit 0
fi

TASK_COUNT="$(wc -l < "${STUDY_LIST}")"
ARRAY_SPEC="0-$((TASK_COUNT - 1))%${MAX_CONCURRENT}"

JOB_ID="$(sbatch --parsable \
  --partition="${PARTITION}" \
  --array="${ARRAY_SPEC}" \
  --export=ALL,REPO_ROOT="${REPO_ROOT}",INPUT_ROOT="${INPUT_ROOT}",OUTPUT_ROOT="${OUTPUT_ROOT}",STUDY_LIST="${STUDY_LIST}" \
  "${SCRIPT_DIR}/submit_tc4d_array.slurm")"

cat <<MSG
Submitted missing-centerlines array job:
  input_root  : ${INPUT_ROOT}
  output_root : ${OUTPUT_ROOT}
  study_list  : ${STUDY_LIST}
  task_count  : ${TASK_COUNT}
  job_id      : ${JOB_ID}

Monitor:
  squeue -j ${JOB_ID}
  sacct -j ${JOB_ID} --format=JobIDRaw,State,Elapsed,ExitCode -n -P
MSG
