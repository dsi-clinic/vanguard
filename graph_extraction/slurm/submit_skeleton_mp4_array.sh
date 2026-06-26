#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CENTERLINES_DIR="${CENTERLINES_DIR:-/ess/scratch/scratch1/t-9sbose/centerlines_tc4d}"
MAX_CONCURRENT="${MAX_CONCURRENT:-32}"
LIST_FILE="${REPO_ROOT}/logs/skeleton_mp4_list.txt"

mkdir -p "${REPO_ROOT}/logs"

# Build the file list (skip already-rendered files)
find "${CENTERLINES_DIR}" -name "*_skeleton_4d_exam_mask.npy" | sort > /tmp/skeleton_all.txt

SKIP=0
> "${LIST_FILE}"
while IFS= read -r path; do
  case_dir="$(dirname "${path}")"
  case_id="$(basename "${case_dir}")"
  mp4="${case_dir}/${case_id}_skeleton_rotating.mp4"
  if [[ -f "${mp4}" ]]; then
    SKIP=$((SKIP + 1))
  else
    echo "${path}" >> "${LIST_FILE}"
  fi
done < /tmp/skeleton_all.txt

TOTAL=$(wc -l < "${LIST_FILE}")
echo "Total skeletons found : $(wc -l < /tmp/skeleton_all.txt)"
echo "Already rendered      : ${SKIP}"
echo "To render             : ${TOTAL}"

if [[ "${TOTAL}" -eq 0 ]]; then
  echo "Nothing to do."
  exit 0
fi

ARRAY_MAX=$((TOTAL - 1))
echo "Submitting array 0-${ARRAY_MAX}%${MAX_CONCURRENT} ..."

REPO_ROOT="${REPO_ROOT}" \
SKELETON_LIST="${LIST_FILE}" \
sbatch --parsable \
  --array="0-${ARRAY_MAX}%${MAX_CONCURRENT}" \
  "${REPO_ROOT}/graph_extraction/slurm/submit_skeleton_mp4_array.slurm"
