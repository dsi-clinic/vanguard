#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_DIR="${INPUT_DIR:-/net/projects2/vanguard/vessel_segmentations}"
OUTPUT_DIR="${OUTPUT_DIR:-/net/projects2/vanguard/graph_pruning_outdir}"
PATTERN="${PATTERN:-*_vessel_segmentation.npy}"
THRESHOLD="${THRESHOLD:-0.5}"
RECURSIVE="${RECURSIVE:-0}"
FILES_PER_TASK="${FILES_PER_TASK:-40}"
CHUNK_SIZE="${CHUNK_SIZE:-100}"
ARRAY_THROTTLE="${ARRAY_THROTTLE:-}"
START_INDEX="${START_INDEX:-}"
END_INDEX="${END_INDEX:-}"

COUNT=$(python - <<'PY'
from pathlib import Path
import math
import os

input_dir = Path(os.environ.get("INPUT_DIR", "/net/projects2/vanguard/vessel_segmentations"))
pattern = os.environ.get("PATTERN", "*_vessel_segmentation.npy")
recursive = os.environ.get("RECURSIVE", "0") == "1"

if recursive:
    files = list(input_dir.rglob(pattern))
else:
    files = list(input_dir.glob(pattern))

count = sum(1 for path in files if path.is_file())
print(count)
PY
)

if [[ "${COUNT}" -le 0 ]]; then
  echo "No .npy files found under: ${INPUT_DIR} (pattern: ${PATTERN})"
  exit 1
fi

TASK_COUNT=$(( (COUNT + FILES_PER_TASK - 1) / FILES_PER_TASK ))
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
  THRESHOLD="${THRESHOLD}" \
  RECURSIVE="${RECURSIVE}" \
  FILES_PER_TASK="${FILES_PER_TASK}" \
  sbatch --array=${ARRAY_SPEC} "${PROJECT_ROOT}/slurm_submit_scripts/submit_graph_pruning_array.slurm"
  exit 0
fi

echo "Submitting array jobs for ${COUNT} files (${TASK_COUNT} tasks: 0-${ARRAY_MAX}) in chunks of ${CHUNK_SIZE}"

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
  THRESHOLD="${THRESHOLD}" \
  RECURSIVE="${RECURSIVE}" \
  FILES_PER_TASK="${FILES_PER_TASK}" \
  sbatch --array=${ARRAY_SPEC} "${PROJECT_ROOT}/slurm_submit_scripts/submit_graph_pruning_array.slurm"

  START=$((END + 1))
done
