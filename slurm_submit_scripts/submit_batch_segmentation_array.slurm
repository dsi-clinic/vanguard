#!/usr/bin/env bash
#SBATCH --job-name=batch-seg-array
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/batch-seg-array-%A-%a.out
#SBATCH --error=logs/batch-seg-array-%A-%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ruochun@duke.edu

set -eo pipefail

# --- Activate your environment ---
source ~/.bashrc || true
eval "$(micromamba shell hook -s bash)"
micromamba activate vanguard

# Prevent libxml2 deactivate hook from failing under `set -u`
export xml_catalog_files_libxml2=""

# --- Make logs/temp unique per job ---
mkdir -p logs
export TMPDIR="/net/projects2/vanguard/tmp/batch_seg_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
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
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
IMAGES_DIR="${IMAGES_DIR:-/net/projects2/vanguard/MAMA-MIA-syn60868042/images}"
OUTPUT_DIR="${OUTPUT_DIR:-/net/projects2/vanguard/vessel_segmentations}"
BREAST_MODEL="${BREAST_MODEL:-${PROJECT_ROOT}/vanguard-blood-vessel-segmentation/trained_models/breast_model.pth}"
VESSEL_MODEL="${VESSEL_MODEL:-${PROJECT_ROOT}/vanguard-blood-vessel-segmentation/trained_models/dv_model.pth}"

cd "${PROJECT_ROOT}"

TASK_ID=${SLURM_ARRAY_TASK_ID}

echo "========================================="
echo "Batch Segmentation Array Job Started"
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

python -u "${PROJECT_ROOT}/batch_processing/batch_segmentation.py" \
  --images-dir         "${IMAGES_DIR}" \
  --output-dir         "${OUTPUT_DIR}" \
  --temp-dir           "${TMPDIR}" \
  --breast-model-path  "${BREAST_MODEL}" \
  --vessel-model-path  "${VESSEL_MODEL}" \
  --file-index         ${TASK_ID} \
  --resume \
  --cleanup

echo ""
echo "========================================="
echo "Array Task ${SLURM_ARRAY_TASK_ID} Completed"
echo "Date: $(date)"
echo "========================================="
