#!/usr/bin/env bash
#SBATCH --job-name=vessel-segmentation
#SBATCH --partition=general          # Use general partition (has GPUs available)
#SBATCH --gres=gpu:1                 # Request 1 GPU for segmentation
#SBATCH --cpus-per-task=16           # Increased CPUs for more workers
#SBATCH --mem=128G                   # Increased memory for more parallel workers
#SBATCH --time=12:00:00              # 12 hours (max for general partition)
#SBATCH --output=logs/vessel-seg-%j.out
#SBATCH --error=logs/vessel-seg-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ruochun@duke.edu

set -euo pipefail

# --- Activate your environment ---
source ~/.bashrc || true
eval "$(micromamba shell hook -s bash)"
micromamba activate vanguard-gpu

# --- Make logs/temp unique per job ---
mkdir -p logs
# Use local /tmp on compute node (avoids filling shared storage when jobs crash)
export TMPDIR="/tmp/batch_vessel_seg_${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}"

# Cleanup function to remove temp directory on exit
cleanup() {
    echo "Cleaning up temporary directory: ${TMPDIR}"
    rm -rf "${TMPDIR}" || true
}
trap cleanup EXIT

# --- Optional: be nice to BLAS/OpenMP threads ---
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1

# --- Configuration ---
# Get project root directory (assuming script is in slurm_submit_scripts/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGES_DIR="/net/projects2/vanguard/MAMA-MIA-syn60868042/images"
OUTPUT_DIR="/net/projects2/vanguard/vessel_segmentations"
BREAST_MODEL="${PROJECT_ROOT}/3D-Breast-FGT-and-Blood-Vessel-Segmentation/trained_models/breast_model.pth"
VESSEL_MODEL="${PROJECT_ROOT}/3D-Breast-FGT-and-Blood-Vessel-Segmentation/trained_models/dv_model.pth"
MAX_WORKERS=8                         # Increased workers for faster processing (8 workers with 16 CPUs)

echo "========================================="
echo "Vessel Segmentation Job Started"
echo "========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
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

