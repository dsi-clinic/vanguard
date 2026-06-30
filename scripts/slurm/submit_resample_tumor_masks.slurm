#!/usr/bin/env bash
# =============================================================================
# Slurm job: resample expert tumor masks onto the RESAMPLED image grid so they
# line up with the resampled vessel segmentations / skeletons.
#
# CPU-ONLY work (SimpleITK has no GPU path) -> no GPU, partition tier1q. Each mask
# is one cheap nearest-neighbour resample, so a single multi-worker job is plenty
# (no array needed). Speed comes from the multiprocessing Pool across the task's
# cores.
#
# Do NOT run on the login node. Submit with:
#   sbatch scripts/slurm/submit_resample_tumor_masks.slurm
#   COHORT=NACT sbatch scripts/slurm/submit_resample_tumor_masks.slurm   # other cohort
# =============================================================================
#SBATCH --job-name=resample-tumor-masks
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/resample-tumor-masks-%j.out
#SBATCH --error=logs/resample-tumor-masks-%j.err
set -eo pipefail

# --- Activate the project environment (same pattern as the resample-array job) --
source ~/.bashrc || true
eval "$(micromamba shell hook -s bash)"
micromamba activate vanguard
export xml_catalog_files_libxml2=""

# --- Keep each math library single-threaded; our parallelism is the Pool --------
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
export PYTHONUNBUFFERED=1

mkdir -p logs

# --- Configuration (override any of these via environment variables) -----------
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MASK_DIR="${MASK_DIR:-/ess/scratch/scratch1/annawoodard/MAMA-MIA-syn60868042/segmentations/expert}"
REFERENCE_ROOT="${REFERENCE_ROOT:-/ess/scratch/scratch1/t-9sbose/resampled_segmentation}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/gpfs/data/karczmar-lab/workspaces/saritbose/resampled-tumor-masks}"
# Default to ISPY2: that is the previous group's baseline cohort. Set COHORT=""
# (empty) to process every cohort.
COHORT="${COHORT-ISPY2}"

cd "${PROJECT_ROOT}"

echo "========================================="
echo "Resample tumor masks"
echo "  Job ID         : ${SLURM_JOB_ID}"
echo "  Node           : $(hostname)"
echo "  CPUs           : ${SLURM_CPUS_PER_TASK}"
echo "  Mask dir       : ${MASK_DIR}"
echo "  Reference root : ${REFERENCE_ROOT}"
echo "  Output root    : ${OUTPUT_ROOT}"
echo "  Cohort         : ${COHORT:-(all)}"
echo "  Date           : $(date)"
echo "========================================="

COHORT_ARG=()
if [[ -n "${COHORT}" ]]; then
  COHORT_ARG=(--cohort "${COHORT}")
fi

python -u "${PROJECT_ROOT}/scripts/resample_tumor_masks.py" \
  --mask-dir       "${MASK_DIR}" \
  --reference-root "${REFERENCE_ROOT}" \
  --output-root    "${OUTPUT_ROOT}" \
  --workers        "${SLURM_CPUS_PER_TASK}" \
  "${COHORT_ARG[@]}" \
  --resume

echo ""
echo "Resample tumor masks finished at $(date)"
