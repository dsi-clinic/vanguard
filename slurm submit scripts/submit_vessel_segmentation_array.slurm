#!/usr/bin/env bash
#SBATCH --job-name=vessel-seg-array
#SBATCH --partition=general          # Use general partition (has GPUs available)
#SBATCH --gres=gpu:1                 # Request 1 GPU per array task
#SBATCH --cpus-per-task=1            # Only need 1 CPU per task (processing single file)
#SBATCH --mem=16G                    # Reduced memory per task (single file processing)
#SBATCH --time=12:00:00              # 12 hours per task
#SBATCH --array=0-7925               # Array size = number of files (update if needed)
#SBATCH --output=logs/vessel-seg-array-%A-%a.out
#SBATCH --error=logs/vessel-seg-array-%A-%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ruochun@duke.edu

set -euo pipefail

# --- Activate your environment ---
source ~/.bashrc || true
eval "$(micromamba shell hook -s bash)"
micromamba activate vanguard-gpu

# --- Make logs/temp unique per job ---
mkdir -p logs
# Use temp directory on /net/projects2/vanguard which has more space
export TMPDIR="/net/projects2/vanguard/tmp/batch_vessel_seg_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${TMPDIR}"

# Cleanup function to remove temp directory on exit
cleanup() {
    echo "Cleaning up temporary directory: ${TMPDIR}"
    rm -rf "${TMPDIR}" || true
}
trap cleanup EXIT

# --- Optimize thread/process settings ---
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# --- Configuration ---
# Get project root directory (assuming script is in slurm submit scripts/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGES_DIR="/net/projects2/vanguard/MAMA-MIA-syn60868042/images"
OUTPUT_DIR="/net/projects2/vanguard/vessel_segmentations"
BREAST_MODEL="${PROJECT_ROOT}/3D-Breast-FGT-and-Blood-Vessel-Segmentation/trained_models/breast_model.pth"
VESSEL_MODEL="${PROJECT_ROOT}/3D-Breast-FGT-and-Blood-Vessel-Segmentation/trained_models/dv_model.pth"

# Each array task processes a single file based on SLURM_ARRAY_TASK_ID
TASK_ID=${SLURM_ARRAY_TASK_ID}

echo "========================================="
echo "Vessel Segmentation Array Job Started"
echo "========================================="
echo "Array Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "Date: $(date)"
echo "========================================="

# --- Run the vessel segmentation pipeline for single file ---
# Each array task processes exactly one file based on its task ID
python -u "${PROJECT_ROOT}/batch_processing/batch_segmentation.py" \
  --images-dir      "${IMAGES_DIR}" \
  --output-dir     "${OUTPUT_DIR}" \
  --temp-dir       "${TMPDIR}" \
  --breast-model-path  "${BREAST_MODEL}" \
  --vessel-model-path  "${VESSEL_MODEL}" \
  --file-index     ${TASK_ID} \
  --resume \
  --cleanup

echo ""
echo "========================================="
echo "Array Task ${SLURM_ARRAY_TASK_ID} Completed"
echo "Date: $(date)"
echo "========================================="

