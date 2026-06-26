#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGES_DIR="${IMAGES_DIR:-/net/projects2/vanguard/MAMA-MIA-syn60868042/images}"
OUTPUT_DIR="${OUTPUT_DIR:-/net/projects2/vanguard/vessel_segmentations}"
BREAST_MODEL="${BREAST_MODEL:-${PROJECT_ROOT}/vanguard-blood-vessel-segmentation/trained_models/breast_model.pth}"
VESSEL_MODEL="${VESSEL_MODEL:-${PROJECT_ROOT}/vanguard-blood-vessel-segmentation/trained_models/dv_model.pth}"
FILES_PER_TASK="${FILES_PER_TASK:-20}"
CHUNK_SIZE="${CHUNK_SIZE:-100}"
# Raise default concurrency; set lower (e.g. 4) if cluster is heavily loaded.
MAX_CONCURRENT_TASKS="${MAX_CONCURRENT_TASKS:-16}"
PARTITION="${PARTITION:-gpuq}"
START_INDEX="${START_INDEX:-}"
END_INDEX="${END_INDEX:-}"

if [[ -n "${START_INDEX}" && -n "${END_INDEX}" ]]; then
  ARRAY_SPEC="${START_INDEX}-${END_INDEX}"

  echo "Submitting single array range: ${ARRAY_SPEC}%${MAX_CONCURRENT_TASKS}"
  IMAGES_DIR="${IMAGES_DIR}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  BREAST_MODEL="${BREAST_MODEL}" \
  VESSEL_MODEL="${VESSEL_MODEL}" \
  FILES_PER_TASK="${FILES_PER_TASK}" \
  sbatch --partition="${PARTITION}" --array=${ARRAY_SPEC}%${MAX_CONCURRENT_TASKS} "${PROJECT_ROOT}/segmentation/slurm/submit_batch_segmentation_array.slurm"
  exit 0
fi

PYTHON="${PYTHON:-python}"
COUNT=$("${PYTHON}" - <<'PY'
from pathlib import Path
import os
images_dir = os.environ.get("IMAGES_DIR", "/net/projects2/vanguard/MAMA-MIA-syn60868042/images")
count = sum(1 for _ in Path(images_dir).glob("*/*.nii.gz"))
print(count)
PY
)

if [[ "${COUNT}" -le 0 ]]; then
  echo "No .nii.gz files found under: ${IMAGES_DIR}"
  exit 1
fi

TASK_COUNT=$(( (COUNT + FILES_PER_TASK - 1) / FILES_PER_TASK ))
ARRAY_MAX=$((TASK_COUNT - 1))

echo "Submitting array jobs for ${COUNT} files"
echo "  Tasks: ${TASK_COUNT} (0-${ARRAY_MAX}), ${FILES_PER_TASK} files/task"
echo "  Max concurrent: ${MAX_CONCURRENT_TASKS}"

START=0
while [[ ${START} -le ${ARRAY_MAX} ]]; do
  END=$((START + CHUNK_SIZE - 1))
  if [[ ${END} -gt ${ARRAY_MAX} ]]; then
    END=${ARRAY_MAX}
  fi

  ARRAY_SPEC="${START}-${END}"

  echo "Submitting array range: ${ARRAY_SPEC}%${MAX_CONCURRENT_TASKS}"
  IMAGES_DIR="${IMAGES_DIR}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  BREAST_MODEL="${BREAST_MODEL}" \
  VESSEL_MODEL="${VESSEL_MODEL}" \
  FILES_PER_TASK="${FILES_PER_TASK}" \
  sbatch --partition="${PARTITION}" --array=${ARRAY_SPEC}%${MAX_CONCURRENT_TASKS} "${PROJECT_ROOT}/segmentation/slurm/submit_batch_segmentation_array.slurm"

  START=$((END + 1))
done
