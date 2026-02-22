#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_DIR="${INPUT_DIR:-/net/projects2/vanguard/vessel_segmentations}"
OUTPUT_DIR="${OUTPUT_DIR:-/net/projects2/vanguard/centerlines_4d}"
PATTERN="${PATTERN:-*_vessel_segmentation.npz}"
THRESHOLD_LOW="${THRESHOLD_LOW:-0.5}"
THRESHOLD_HIGH="${THRESHOLD_HIGH:-0.85}"
NPY_CHANNEL="${NPY_CHANNEL:-1}"
STUDIES_PER_TASK="${STUDIES_PER_TASK:-1}"
CHUNK_SIZE="${CHUNK_SIZE:-100}"
ARRAY_THROTTLE="${ARRAY_THROTTLE:-}"
START_INDEX="${START_INDEX:-}"
END_INDEX="${END_INDEX:-}"

COUNT=$(python - <<'PY'
from pathlib import Path
import os

input_dir = Path(os.environ.get("INPUT_DIR", "/net/projects2/vanguard/vessel_segmentations"))
pattern = os.environ.get("PATTERN", "*_vessel_segmentation.npz")

# Extract unique study IDs from filenames
files = list(input_dir.rglob(pattern))
study_ids = set()
for f in files:
    # Filenames like: ISPY2_202539_T0_vessel_segmentation.npz
    # Extract ISPY2_202539 as study ID
    parts = f.stem.split("_")
    if len(parts) >= 3:
        study_id = "_".join(parts[:2])  # ISPY2_202539
        study_ids.add(study_id)

count = len(study_ids)
print(count)
PY
)

if [[ "${COUNT}" -le 0 ]]; then
  echo "No study IDs found under: ${INPUT_DIR} (pattern: ${PATTERN})"
  exit 1
fi

TASK_COUNT=$(( (COUNT + STUDIES_PER_TASK - 1) / STUDIES_PER_TASK ))
ARRAY_MAX=$((TASK_COUNT - 1))

if [[ -n "${START_INDEX}" && -n "${END_INDEX}" ]]; then
  if [[ -n "${ARRAY_THROTTLE}" ]]; then
    ARRAY_SPEC="${START_INDEX}-${END_INDEX}%${ARRAY_THROTTLE}"
  else
    ARRAY_SPEC="${START_INDEX}-${END_INDEX}"
  fi

  echo "Submitting single array range: ${ARRAY_SPEC}"
  INPUT_DIR="${INPUT_DIR}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  PATTERN="${PATTERN}" \
  THRESHOLD_LOW="${THRESHOLD_LOW}" \
  THRESHOLD_HIGH="${THRESHOLD_HIGH}" \
  NPY_CHANNEL="${NPY_CHANNEL}" \
  STUDIES_PER_TASK="${STUDIES_PER_TASK}" \
  sbatch --array=${ARRAY_SPEC} "${PROJECT_ROOT}/slurm_submit_scripts/submit_graph_pruning_array.slurm"
  exit 0
fi

echo "Submitting array jobs for ${COUNT} study IDs (${TASK_COUNT} tasks: 0-${ARRAY_MAX}) in chunks of ${CHUNK_SIZE}"

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
  INPUT_DIR="${INPUT_DIR}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  PATTERN="${PATTERN}" \
  THRESHOLD_LOW="${THRESHOLD_LOW}" \
  THRESHOLD_HIGH="${THRESHOLD_HIGH}" \
  NPY_CHANNEL="${NPY_CHANNEL}" \
  STUDIES_PER_TASK="${STUDIES_PER_TASK}" \
  sbatch --array=${ARRAY_SPEC} "${PROJECT_ROOT}/slurm_submit_scripts/submit_graph_pruning_array.slurm"

  START=$((END + 1))
done
