#!/usr/bin/env bash
#SBATCH --job-name=stage-hr-dicom
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/stage-hr-dicom-%A_%a.out
#SBATCH --error=logs/slurm/stage-hr-dicom-%A_%a.err

set -euo pipefail
umask 0007

: "${REPO_ROOT:?Set REPO_ROOT to the Vanguard checkout}"
: "${SELECTION_MANIFEST:?Set SELECTION_MANIFEST to the reviewed series manifest}"
: "${DESTINATION_ROOT:?Set DESTINATION_ROOT to the HR staging directory}"

VANGUARD_PYTHON="${VANGUARD_PYTHON:-/ess/scratch/scratch1/aakrithiram/micromamba/envs/vanguard/bin/python}"

cd "${REPO_ROOT}"
echo "[job] id=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} host=$(hostname)"
echo "[python] ${VANGUARD_PYTHON}"
"${VANGUARD_PYTHON}" preprocessing/stage_high_resolution_dicom.py stage \
  --selection "${SELECTION_MANIFEST}" \
  --destination "${DESTINATION_ROOT}" \
  --index "${SLURM_ARRAY_TASK_ID}"
