#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGES_DIR="${IMAGES_DIR:-/net/projects2/vanguard/MAMA-MIA-syn60868042/images}"
OUTPUT_DIR="${OUTPUT_DIR:-/net/projects2/vanguard/vessel_segmentations}"
BREAST_MODEL="${BREAST_MODEL:-${PROJECT_ROOT}/vanguard-blood-vessel-segmentation/trained_models/breast_model.pth}"
VESSEL_MODEL="${VESSEL_MODEL:-${PROJECT_ROOT}/vanguard-blood-vessel-segmentation/trained_models/dv_model.pth}"
CHUNK_SIZE="${CHUNK_SIZE:-100}"
ARRAY_THROTTLE="${ARRAY_THROTTLE:-}"
START_INDEX="${START_INDEX:-}"
END_INDEX="${END_INDEX:-}"

COUNT=$(python - <<'PY'
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

ARRAY_MAX=$((COUNT - 1))

if [[ -n "${START_INDEX}" && -n "${END_INDEX}" ]]; then
  if [[ -n "${ARRAY_THROTTLE}" ]]; then
    ARRAY_SPEC="${START_INDEX}-${END_INDEX}%${ARRAY_THROTTLE}"
  else
    ARRAY_SPEC="${START_INDEX}-${END_INDEX}"
  fi

  echo "Submitting single array range: ${ARRAY_SPEC}"
  IMAGES_DIR="${IMAGES_DIR}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  BREAST_MODEL="${BREAST_MODEL}" \
  VESSEL_MODEL="${VESSEL_MODEL}" \
  sbatch --array=${ARRAY_SPEC} "${PROJECT_ROOT}/slurm_submit_scripts/submit_batch_segmentation_array.slurm"
  exit 0
fi

echo "Submitting array jobs for ${COUNT} files (0-${ARRAY_MAX}) in chunks of ${CHUNK_SIZE}"

START=0
while [[ ${START} -le ${ARRAY_MAX} ]]; do
  END=$((START + CHUNK_SIZE - 1))
  if [[ ${END} -gt ${ARRAY_MAX} ]]; then
    END=${ARRAY_MAX}
  fi

  if [[ -n "${ARRAY_THROTTLE}" ]]; then
    ARRAY_SPEC="${START}-${END}%${ARRAY_THROTTLE}"
  else
    ARRAY_SPEC="${START}-${END}"
  fi

  echo "Submitting array range: ${ARRAY_SPEC}"
  IMAGES_DIR="${IMAGES_DIR}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  BREAST_MODEL="${BREAST_MODEL}" \
  VESSEL_MODEL="${VESSEL_MODEL}" \
  sbatch --array=${ARRAY_SPEC} "${PROJECT_ROOT}/slurm_submit_scripts/submit_batch_segmentation_array.slurm"

  START=$((END + 1))
done
