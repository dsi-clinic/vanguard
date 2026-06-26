#!/usr/bin/env bash
# Submit the fast segmentation pipeline as a SLURM array over all images.
# Usage: bash submit_fast_array.sh
#   Override any env var before calling, e.g.:
#     FILES_PER_TASK=30 MAX_CONCURRENT_TASKS=8 bash submit_fast_array.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGES_DIR="${IMAGES_DIR:-/ess/scratch/scratch1/annawoodard/MAMA-MIA-syn60868042/images}"
OUTPUT_DIR="${OUTPUT_DIR:-/ess/scratch/scratch1/t-9sbose/vessel_segmentations_fast_smoke}"
BREAST_MODEL="${BREAST_MODEL:-${PROJECT_ROOT}/vanguard-blood-vessel-segmentation/trained_models/breast_model.pth}"
VESSEL_MODEL="${VESSEL_MODEL:-${PROJECT_ROOT}/vanguard-blood-vessel-segmentation/trained_models/dv_model.pth}"
FILES_PER_TASK="${FILES_PER_TASK:-20}"
MAX_CONCURRENT_TASKS="${MAX_CONCURRENT_TASKS:-16}"
PARTITION="${PARTITION:-gpuq}"

PYTHON="${PYTHON:-python3}"
COUNT=$("${PYTHON}" - <<'PY'
from pathlib import Path
import os
images_dir = os.environ.get("IMAGES_DIR", "/ess/scratch/scratch1/annawoodard/MAMA-MIA-syn60868042/images")
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

echo "Images dir:       ${IMAGES_DIR}"
echo "Output dir:       ${OUTPUT_DIR}"
echo "Files found:      ${COUNT}"
echo "Files per task:   ${FILES_PER_TASK}"
echo "Tasks:            ${TASK_COUNT} (0-${ARRAY_MAX})"
echo "Max concurrent:   ${MAX_CONCURRENT_TASKS}"
echo "Partition:        ${PARTITION}"
echo ""

mkdir -p /ess/home/home1/t-9sbose/vanguard/logs

IMAGES_DIR="${IMAGES_DIR}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
BREAST_MODEL="${BREAST_MODEL}" \
VESSEL_MODEL="${VESSEL_MODEL}" \
FILES_PER_TASK="${FILES_PER_TASK}" \
sbatch \
  --partition="${PARTITION}" \
  --array=0-${ARRAY_MAX}%${MAX_CONCURRENT_TASKS} \
  "${PROJECT_ROOT}/faster-segmentation-test/submit_fast_array.slurm"

echo ""
echo "Submitted ${TASK_COUNT} tasks (up to ${MAX_CONCURRENT_TASKS} running at once)."
echo "Logs: /ess/home/home1/t-9sbose/vanguard/logs/fast-seg-array-<JOBID>-<TASKID>.{out,err}"
