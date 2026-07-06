#!/usr/bin/env bash
#SBATCH --job-name=breast-mask-array
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/breast-mask-array-%A-%a.out
#SBATCH --error=logs/breast-mask-array-%A-%a.err
set -eo pipefail

# --- Activate your environment ---
source ~/.bashrc || true
eval "$(micromamba shell hook -s bash)"
micromamba activate vanguard

# Prevent libxml2 deactivate hook from failing under `set -u`
export xml_catalog_files_libxml2=""

# --- Make logs/temp unique per job ---
mkdir -p logs
export TMPDIR="/tmp/breast_mask_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${TMPDIR}"

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
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
IMAGES_DIR="${IMAGES_DIR:-/scratch/annawoodard/MAMA-MIA-syn60868042/images}"
OUTPUT_DIR="${OUTPUT_DIR:-/gpfs/data/karczmar-lab/workspaces/saritbose/breast_masks}"
BREAST_MODEL="${BREAST_MODEL:-${PROJECT_ROOT}/vanguard-blood-vessel-segmentation/trained_models/breast_model.pth}"

cd "${PROJECT_ROOT}"

TASK_ID=${SLURM_ARRAY_TASK_ID}
FILES_PER_TASK=${FILES_PER_TASK:-40}
FILE_START=$((TASK_ID * FILES_PER_TASK))
FILE_END=$((FILE_START + FILES_PER_TASK - 1))

echo "========================================="
echo "Breast Mask (STEP-2 only) Array Job Started"
echo "========================================="
echo "Array Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-none}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "Images dir: ${IMAGES_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Date: $(date)"
echo "========================================="

python -u "${PROJECT_ROOT}/segmentation/batch_segmentation.py" \
  --images-dir         "${IMAGES_DIR}" \
  --output-dir         "${OUTPUT_DIR}" \
  --temp-dir           "${TMPDIR}" \
  --breast-model-path  "${BREAST_MODEL}" \
  --skip-vessel \
  --file-start         ${FILE_START} \
  --file-end           ${FILE_END} \
  --resume \
  --cleanup

echo ""
echo "========================================="
echo "Array Task ${SLURM_ARRAY_TASK_ID} Completed"
echo "Date: $(date)"
echo "========================================="
