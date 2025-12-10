#!/usr/bin/env bash
#SBATCH --job-name=vessel-seg-opt
#SBATCH --partition=general          # Use general partition (has GPUs available)
#SBATCH --gres=gpu:1                 # Request 1 GPU for segmentation
#SBATCH --cpus-per-task=32           # Increased CPUs for more workers (was 16)
#SBATCH --mem=256G                   # Increased memory for more parallel workers (was 128G)
#SBATCH --time=12:00:00              # 12 hours (max for general partition)
#SBATCH --output=logs/vessel-seg-opt-%j.out
#SBATCH --error=logs/vessel-seg-opt-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ruochun@duke.edu

set -euo pipefail

# --- Activate your environment ---
source ~/.bashrc || true
eval "$(micromamba shell hook -s bash)"
micromamba activate vanguard-gpu

# --- Make logs/temp unique per job ---
mkdir -p logs
# Use temp directory on /net/projects2/vanguard which has more space (601G available)
export TMPDIR="/net/projects2/vanguard/tmp/batch_vessel_seg_${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}"

# Cleanup function to remove temp directory on exit
cleanup() {
    echo "Cleaning up temporary directory: ${TMPDIR}"
    rm -rf "${TMPDIR}" || true
}
trap cleanup EXIT

# --- Optimize thread/process settings ---
# More aggressive parallelization
export OMP_NUM_THREADS=4              # Reduced per-process threads (was full CPU count)
export MKL_NUM_THREADS=4              # Reduced per-process threads
export CUDA_VISIBLE_DEVICES=0         # Explicitly set GPU
export PYTHONUNBUFFERED=1

# --- Configuration ---
# Get project root directory (assuming script is in slurm_submit_scripts/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGES_DIR="/net/projects2/vanguard/MAMA-MIA-syn60868042/images"
OUTPUT_DIR="/net/projects2/vanguard/vessel_segmentations"
BREAST_MODEL="${PROJECT_ROOT}/3D-Breast-FGT-and-Blood-Vessel-Segmentation/trained_models/breast_model.pth"
VESSEL_MODEL="${PROJECT_ROOT}/3D-Breast-FGT-and-Blood-Vessel-Segmentation/trained_models/dv_model.pth"
MAX_WORKERS=16                        # Increased workers (was 8) - 2 CPUs per worker with 32 total

echo "========================================="
echo "Vessel Segmentation Job Started (OPTIMIZED)"
echo "========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "Workers: ${MAX_WORKERS}"
echo "Time limit: ${SLURM_TIME_LIMIT:-unknown}"
echo "Date: $(date)"
echo "========================================="

# --- Run the vessel segmentation pipeline with resume ---
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
echo "If job timed out, resubmit to continue processing"
echo "========================================="

echo "========================================="
echo "Job Completed Successfully"
echo "Date: $(date)"
echo "========================================="

