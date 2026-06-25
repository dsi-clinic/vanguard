#!/usr/bin/env bash
# =============================================================================
# Slurm job array: resample MAMA-MIA volumes into the acceptable spacing range.
#
# This is CPU-ONLY work (SimpleITK has no GPU path), so we request NO GPU and run
# on the general-purpose CPU partition `tier1q`. Speed comes from parallelism:
#   - the ARRAY spreads cases across many nodes (one task = a chunk of cases),
#   - each task runs a multiprocessing Pool across its `--cpus-per-task` cores.
#
# Do NOT run this on the login node. Submit it with scripts/slurm/submit_resample_array.sh
# =============================================================================
#SBATCH --job-name=resample-array
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/resample-array-%A-%a.out
#SBATCH --error=logs/resample-array-%A-%a.err
set -eo pipefail

# --- Activate the project environment (same pattern as the segmentation job) ---
source ~/.bashrc || true
eval "$(micromamba shell hook -s bash)"
micromamba activate vanguard

# Prevent the libxml2 deactivate hook from failing under `set -u`
export xml_catalog_files_libxml2=""

# --- Keep each math library single-threaded; our parallelism is the Pool --------
# (If ITK/OpenMP also grabbed all 8 cores per process, the 8 worker processes would
#  fight over the same cores and the whole thing would run SLOWER, not faster.)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
export PYTHONUNBUFFERED=1

mkdir -p logs

# --- Configuration (override any of these via environment variables) -----------
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-/ess/scratch/scratch1/annawoodard/MAMA-MIA-syn60868042/images}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/ess/scratch/scratch1/t-9sbose/resampled_segmentation}"
CASES_PER_TASK="${CASES_PER_TASK:-16}"

cd "${PROJECT_ROOT}"

# Which slice of the sorted case list does THIS array task own?
TASK_ID=${SLURM_ARRAY_TASK_ID}
CASE_START=$((TASK_ID * CASES_PER_TASK))
CASE_END=$((CASE_START + CASES_PER_TASK - 1))   # inclusive; the script clamps to the real end

echo "========================================="
echo "Resample Array Task Started"
echo "  Array Job ID : ${SLURM_ARRAY_JOB_ID}"
echo "  Task ID      : ${SLURM_ARRAY_TASK_ID}"
echo "  Node         : $(hostname)"
echo "  CPUs         : ${SLURM_CPUS_PER_TASK}"
echo "  Cases        : ${CASE_START}..${CASE_END}  (${CASES_PER_TASK} per task)"
echo "  Data root    : ${DATA_ROOT}"
echo "  Output root  : ${OUTPUT_ROOT}"
echo "  Date         : $(date)"
echo "========================================="

python -u "${PROJECT_ROOT}/scripts/resample_to_range.py" \
  --data-root   "${DATA_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --case-start  "${CASE_START}" \
  --case-end    "${CASE_END}" \
  --workers     "${SLURM_CPUS_PER_TASK}" \
  --resume

echo ""
echo "========================================="
echo "Resample Array Task ${SLURM_ARRAY_TASK_ID} finished at $(date)"
echo "========================================="
