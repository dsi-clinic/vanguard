#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
INPUT_ROOT="${1:-/net/projects2/vanguard/vessel_segmentations}"
OUTPUT_ROOT="${2:-/net/projects2/vanguard/centerlines_tc4d/studies}"
TEST_MODE="${3:-}"
PARTITION="${PARTITION:-tier1q}"
SEG_ONLY="${SEG_ONLY:-0}"
MAX_CONCURRENT="${MAX_CONCURRENT:-50}"

mkdir -p "${REPO_ROOT}/logs" "${OUTPUT_ROOT}"
STUDY_LIST="$(mktemp "${REPO_ROOT}/logs/tc4d-study-list.XXXXXX.txt")"

SEG_ONLY_PY="$( [[ "${SEG_ONLY}" == "1" ]] && echo True || echo False )"

python3 - <<PY > "${STUDY_LIST}"
from pathlib import Path
input_root = Path(${INPUT_ROOT@Q})
seg_only = ${SEG_ONLY_PY}
rows = []
for site_dir in sorted(input_root.iterdir()):
    if not site_dir.is_dir():
        continue
    for study_dir in sorted(site_dir.iterdir()):
        if not study_dir.is_dir():
            continue
        if seg_only:
            images_dir = study_dir / "images"
            if not images_dir.is_dir():
                continue
            if not any(f.name.endswith("_vessel_segmentation.npz")
                       for f in images_dir.iterdir()):
                continue
        rows.append(f"{site_dir.name}/{study_dir.name}")
print("\n".join(rows))
PY

if [[ ! -s "${STUDY_LIST}" ]]; then
  echo "No studies found under ${INPUT_ROOT}" >&2
  exit 1
fi

if [[ "${TEST_MODE}" == "--test" ]]; then
  head -n 5 "${STUDY_LIST}" > "${STUDY_LIST}.tmp"
  mv "${STUDY_LIST}.tmp" "${STUDY_LIST}"
fi

TASK_COUNT="$(wc -l < "${STUDY_LIST}")"
ARRAY_SPEC="0-$((TASK_COUNT - 1))%${MAX_CONCURRENT}"

JOB_ID="$({
  sbatch --parsable \
    --partition="${PARTITION}" \
    --array="${ARRAY_SPEC}" \
    --export=ALL,REPO_ROOT="${REPO_ROOT}",INPUT_ROOT="${INPUT_ROOT}",OUTPUT_ROOT="${OUTPUT_ROOT}",STUDY_LIST="${STUDY_LIST}" \
    "${SCRIPT_DIR}/submit_tc4d_array.slurm"
} )"

cat <<MSG
Submitted tc4d array job:
  input_root : ${INPUT_ROOT}
  output_root: ${OUTPUT_ROOT}
  seg_only   : ${SEG_ONLY}
  study_list : ${STUDY_LIST}
  task_count : ${TASK_COUNT}
  job_id     : ${JOB_ID}

Monitor:
  squeue -j ${JOB_ID}
  sacct -j ${JOB_ID} --format=JobIDRaw,State,Elapsed,ExitCode -n -P
MSG
