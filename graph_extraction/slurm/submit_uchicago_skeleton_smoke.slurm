#!/usr/bin/env bash
#SBATCH --job-name=uc-skel-smoke
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/uc-skel-smoke-%A-%a.out
#SBATCH --error=logs/uc-skel-smoke-%A-%a.err

set -euo pipefail

cd "${REPO_ROOT}"
source ~/.bashrc || true
if command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook -s bash)"
  micromamba activate vanguard
fi

LINE_NO=$((SLURM_ARRAY_TASK_ID + 1))
LINE="$(sed -n "${LINE_NO}p" "${CASE_LIST}")"
CASE_ID="${LINE%%,*}"
SUB_SOURCE="${LINE##*,}"
INPUT_DIR="${SEG_OUTPUT_DIR}/${SUB_SOURCE}"
OUTPUT_DIR="${OUTPUT_ROOT}/${SUB_SOURCE}/${CASE_ID}"

echo "Case: ${CASE_ID}  sub_source: ${SUB_SOURCE}"
echo "Input dir: ${INPUT_DIR}"
echo "Output dir: ${OUTPUT_DIR}"

python graph_extraction/run_skeleton_processing.py \
  --case-id "${CASE_ID}" \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --dataset-name uchicago \
  --dataset-root "${UCHICAGO_ROOT}"
