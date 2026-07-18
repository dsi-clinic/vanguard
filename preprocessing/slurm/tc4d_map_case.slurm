#!/bin/bash
#SBATCH --job-name=vanguard-tc4d
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/vanguard-tc4d-%j.out

set -euo pipefail

: "${EXAM_ID:?set EXAM_ID}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT}"
: "${VANGUARD_ENV_PREFIX:=/ess/scratch/scratch1/aakrithiram/micromamba/envs/vanguard}"
: "${TC4D_PYTHON:=${VANGUARD_ENV_PREFIX}/bin/python}"
: "${MAPPING_PYTHON:=${VANGUARD_ENV_PREFIX}/bin/python}"

TC4D_ROOT="${OUTPUT_ROOT}/work/${EXAM_ID}/hr_tc4d"
TC4D_EXPECTED=(
  "${TC4D_ROOT}/exam_skeleton_zyx.npy"
  "${TC4D_ROOT}/exam_support_zyx.npy"
  "${TC4D_ROOT}/center_manifold_4d_yxz.npy"
)
if [[ -d "$TC4D_ROOT" ]]; then
  for path in "${TC4D_EXPECTED[@]}"; do
    [[ -f "$path" ]] || { echo "incomplete existing TC4D output: $path" >&2; exit 1; }
  done
  echo "reusing complete TC4D output: $TC4D_ROOT"
else
  "$TC4D_PYTHON" -m preprocessing.pipeline tc4d \
    --exam-id "$EXAM_ID" \
    --output-root "$OUTPUT_ROOT"
fi
DATASET="$($MAPPING_PYTHON -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["case"]["dataset"])' \
  "${OUTPUT_ROOT}/work/${EXAM_ID}/preprocessing_provenance.json")"
CENTERLINE_ROOT="${OUTPUT_ROOT}/centerlines/${DATASET}/${EXAM_ID}"
if [[ -f "${CENTERLINE_ROOT}/${EXAM_ID}_skeleton_4d_exam_mask.npy" \
   && -f "${CENTERLINE_ROOT}/${EXAM_ID}_skeleton_4d_exam_support_mask.npy" \
   && -f "${CENTERLINE_ROOT}/run_summary.json" ]]; then
  echo "reusing complete mapped centerline output: $CENTERLINE_ROOT"
else
  "$MAPPING_PYTHON" -m preprocessing.pipeline map \
    --exam-id "$EXAM_ID" \
    --output-root "$OUTPUT_ROOT"
fi
if [[ -f "${CENTERLINE_ROOT}/mapping_qc.png" ]]; then
  echo "reusing mapping QC: ${CENTERLINE_ROOT}/mapping_qc.png"
else
  "$MAPPING_PYTHON" -m preprocessing.pipeline qc \
    --exam-id "$EXAM_ID" \
    --output-root "$OUTPUT_ROOT"
fi
