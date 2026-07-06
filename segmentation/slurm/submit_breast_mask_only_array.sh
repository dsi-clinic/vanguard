#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGES_DIR="${IMAGES_DIR:-/scratch/annawoodard/MAMA-MIA-syn60868042/images}"
OUTPUT_DIR="${OUTPUT_DIR:-/gpfs/data/karczmar-lab/workspaces/saritbose/breast_masks}"
BREAST_MODEL="${BREAST_MODEL:-${PROJECT_ROOT}/vanguard-blood-vessel-segmentation/trained_models/breast_model.pth}"
FILES_PER_TASK="${FILES_PER_TASK:-40}"
CHUNK_SIZE="${CHUNK_SIZE:-100}"
START_INDEX="${START_INDEX:-}"
END_INDEX="${END_INDEX:-}"

COUNT=$(python3 - <<'PY'
from pathlib import Path
import os
images_dir = os.environ.get("IMAGES_DIR", "/scratch/annawoodard/MAMA-MIA-syn60868042/images")
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

if [[ -n "${START_INDEX}" && -n "${END_INDEX}" ]]; then
  ARRAY_SPEC="${START_INDEX}-${END_INDEX}"

  echo "Submitting single array range: ${ARRAY_SPEC}"
  IMAGES_DIR="${IMAGES_DIR}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  BREAST_MODEL="${BREAST_MODEL}" \
  FILES_PER_TASK="${FILES_PER_TASK}" \
  sbatch --array=${ARRAY_SPEC} "${PROJECT_ROOT}/segmentation/slurm/submit_breast_mask_only_array.slurm"
  exit 0
fi

echo "Submitting array jobs for ${COUNT} files (${TASK_COUNT} tasks: 0-${ARRAY_MAX}) in chunks of ${CHUNK_SIZE}"

START=0
while [[ ${START} -le ${ARRAY_MAX} ]]; do
  END=$((START + CHUNK_SIZE - 1))
  if [[ ${END} -gt ${ARRAY_MAX} ]]; then
    END=${ARRAY_MAX}
  fi

  ARRAY_SPEC="${START}-${END}"

  echo "Submitting array range: ${ARRAY_SPEC}"
  IMAGES_DIR="${IMAGES_DIR}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  BREAST_MODEL="${BREAST_MODEL}" \
  FILES_PER_TASK="${FILES_PER_TASK}" \
  sbatch --array=${ARRAY_SPEC} "${PROJECT_ROOT}/segmentation/slurm/submit_breast_mask_only_array.slurm"

  START=$((END + 1))
done
