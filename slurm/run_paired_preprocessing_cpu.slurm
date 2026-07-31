#!/usr/bin/env bash
#SBATCH --job-name=vanguard-paired-cpu
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/vanguard-paired-%x-%A_%a.out
#SBATCH --error=logs/slurm/vanguard-paired-%x-%A_%a.err
# postprocess runs tc4d+map+qc. tc4d also runs unmodified TC4D over the
# UFAST-direct route (CPU-bound) before map merges it in. Matches the 11.5h
# budget analysis/submit_skeletonize_ufast_vessels.slurm already uses for that
# computation, plus headroom for the (fast) HR TC4D + map + qc stages that
# still run in this same job. prepare shares this script and is lighter.

set -euo pipefail
umask 0007

: "${REPO_ROOT:?Set REPO_ROOT to the Vanguard checkout}"
: "${PAIRED_INVENTORY:?Set PAIRED_INVENTORY to the shared DICOM inventory}"
: "${CASE_MANIFEST:?Set CASE_MANIFEST to the reviewed runnable cases}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a derived-output directory}"
: "${PIPELINE_STAGE:?Set PIPELINE_STAGE to prepare or postprocess}"

VANGUARD_PYTHON="${VANGUARD_PYTHON:-${CONDA_PREFIX:?activate the vanguard env or set VANGUARD_PYTHON}/bin/python}"

cd "${REPO_ROOT}"
echo "[job] id=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} host=$(hostname)"
echo "[stage] ${PIPELINE_STAGE}"
echo "[python] ${VANGUARD_PYTHON}"
"${VANGUARD_PYTHON}" preprocessing/run_cohort.py "${PIPELINE_STAGE}" \
  --inventory "${PAIRED_INVENTORY}" \
  --case-manifest "${CASE_MANIFEST}" \
  --output-root "${OUTPUT_ROOT}" \
  --array-index "${SLURM_ARRAY_TASK_ID}"
