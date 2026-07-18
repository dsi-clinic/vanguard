#!/usr/bin/env bash
#SBATCH --job-name=finalize-hr-dicom
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/finalize-hr-dicom-%j.out
#SBATCH --error=logs/slurm/finalize-hr-dicom-%j.err

set -euo pipefail
umask 0007

: "${REPO_ROOT:?Set REPO_ROOT to the Vanguard checkout}"
: "${SELECTION_MANIFEST:?Set SELECTION_MANIFEST to the reviewed series manifest}"
: "${DESTINATION_ROOT:?Set DESTINATION_ROOT to the HR staging directory}"
: "${UFAST_MANIFEST:?Set UFAST_MANIFEST to the existing cohort manifest}"

VANGUARD_PYTHON="${VANGUARD_PYTHON:-/ess/scratch/scratch1/aakrithiram/micromamba/envs/vanguard/bin/python}"

cd "${REPO_ROOT}"
echo "[job] id=${SLURM_JOB_ID} host=$(hostname)"
echo "[python] ${VANGUARD_PYTHON}"
"${VANGUARD_PYTHON}" preprocessing/stage_high_resolution_dicom.py finalize \
  --selection "${SELECTION_MANIFEST}" \
  --destination "${DESTINATION_ROOT}" \
  --ufast-manifest "${UFAST_MANIFEST}"

"${VANGUARD_PYTHON}" preprocessing/stage_high_resolution_dicom.py geometry \
  --selection "${SELECTION_MANIFEST}" \
  --destination "${DESTINATION_ROOT}"
