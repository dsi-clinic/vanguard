#!/usr/bin/env bash
#SBATCH --job-name=uc-skel-smoke
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/uc-skel-smoke-%A-%a.out
#SBATCH --error=logs/uc-skel-smoke-%A-%a.err

# `general` does not exist on this cluster (see gnn/slurm/submit_gnn_build.slurm);
# `tier1q` is the confirmed replacement. `-u` (nounset) is deliberately not set
# (matching segmentation/slurm/submit_uchicago_segmentation_smoke.slurm's
# `set -eo pipefail`) -- it tripped on an unset variable inside the system
# /etc/bashrc during `source ~/.bashrc || true`, aborting every array task
# before it reached any of this script's own work (all 5 tasks of job 12844891
# failed in ~2s with only a bashrc stderr line and no other output).
set -eo pipefail

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
