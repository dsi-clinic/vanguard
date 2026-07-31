#!/usr/bin/env bash
#SBATCH --job-name=vanguard-paired-infer
#SBATCH --partition=gpuq
#SBATCH --qos=priority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/vanguard-paired-infer-%A_%a.out
#SBATCH --error=logs/slurm/vanguard-paired-infer-%A_%a.err
# infer now runs breast+vessel model inference twice per case (HR route,
# then the UFAST-direct route -- UFAST typically has several times more
# temporal phases than HR, e.g. 24 vs 5 for a representative exam), so the
# original 4h HR-only budget is doubled here for headroom.

set -euo pipefail
umask 0007

: "${REPO_ROOT:?Set REPO_ROOT to the Vanguard checkout}"
: "${PAIRED_INVENTORY:?Set PAIRED_INVENTORY to the shared DICOM inventory}"
: "${CASE_MANIFEST:?Set CASE_MANIFEST to the reviewed runnable cases}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a derived-output directory}"

VANGUARD_PYTHON="${VANGUARD_PYTHON:-${CONDA_PREFIX:?activate the vanguard env or set VANGUARD_PYTHON}/bin/python}"
BATCH_SIZE="${BATCH_SIZE:-8}"

cd "${REPO_ROOT}"
echo "[job] id=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} host=$(hostname)"
echo "[python] ${VANGUARD_PYTHON}"
"${VANGUARD_PYTHON}" preprocessing/run_cohort.py infer \
  --inventory "${PAIRED_INVENTORY}" \
  --case-manifest "${CASE_MANIFEST}" \
  --output-root "${OUTPUT_ROOT}" \
  --array-index "${SLURM_ARRAY_TASK_ID}" \
  --batch-size "${BATCH_SIZE}"
