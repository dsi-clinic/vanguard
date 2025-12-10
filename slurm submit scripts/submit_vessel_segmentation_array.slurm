#!/usr/bin/env bash
#SBATCH --job-name=vessel-seg-array
#SBATCH --partition=general          # Use general partition (has GPUs available)
#SBATCH --gres=gpu:1                 # Request 1 GPU per array task
#SBATCH --cpus-per-task=16           # CPUs per task
#SBATCH --mem=128G                   # Memory per task
#SBATCH --time=12:00:00              # 12 hours per task
#SBATCH --array=0-3                  # Run 4 jobs in parallel (adjust based on available GPUs)
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
MAX_WORKERS=8                         # Workers per array task

# Calculate which files this array task should process
# Each task processes a subset of files based on array index
TOTAL_TASKS=${SLURM_ARRAY_TASK_COUNT:-4}
TASK_ID=${SLURM_ARRAY_TASK_ID}

echo "========================================="
echo "Vessel Segmentation Array Job Started"
echo "========================================="
echo "Array Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Total Tasks: ${TOTAL_TASKS}"
echo "Node: $(hostname)"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "Workers: ${MAX_WORKERS}"
echo "Date: $(date)"
echo "========================================="

# --- Run the vessel segmentation pipeline with resume ---
# Each array task processes all files but with different workers
# The --resume flag ensures no duplicate processing
python -u "${PROJECT_ROOT}/batch_processing/batch_segmentation.py" \
  --images-dir      "${IMAGES_DIR}" \
  --output-dir     "${OUTPUT_DIR}" \
  --temp-dir       "${TMPDIR}" \
  --breast-model-path  "${BREAST_MODEL}" \
  --vessel-model-path  "${VESSEL_MODEL}" \
  --max-workers    ${MAX_WORKERS} \
  --resume \
  --cleanup

echo ""
echo "========================================="
echo "Array Task ${SLURM_ARRAY_TASK_ID} Completed"
echo "Date: $(date)"
echo "========================================="

