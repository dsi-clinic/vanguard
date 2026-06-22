#!/usr/bin/env bash
#SBATCH --job-name=fast-seg-array
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/ess/home/home1/t-9sbose/vanguard/logs/fast-seg-array-%A-%a.out
#SBATCH --error=/ess/home/home1/t-9sbose/vanguard/logs/fast-seg-array-%A-%a.err
set -eo pipefail

source ~/.bashrc || true
eval "$(micromamba shell hook -s bash)"
micromamba activate vanguard
export xml_catalog_files_libxml2=""

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p /ess/home/home1/t-9sbose/vanguard/logs

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

IMAGES_DIR="${IMAGES_DIR:-/ess/scratch/scratch1/annawoodard/MAMA-MIA-syn60868042/images}"
OUTPUT_DIR="${OUTPUT_DIR:-/ess/scratch/scratch1/t-9sbose/vessel_segmentations_fast_smoke}"
BREAST_MODEL="${BREAST_MODEL:-${PROJECT_ROOT}/vanguard-blood-vessel-segmentation/trained_models/breast_model.pth}"
VESSEL_MODEL="${VESSEL_MODEL:-${PROJECT_ROOT}/vanguard-blood-vessel-segmentation/trained_models/dv_model.pth}"
FILES_PER_TASK="${FILES_PER_TASK:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"

TASK_ID=${SLURM_ARRAY_TASK_ID}
FILE_START=$((TASK_ID * FILES_PER_TASK))
FILE_END=$((FILE_START + FILES_PER_TASK - 1))

TMPDIR="/tmp/fast_seg_array_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${TMPDIR}"
trap 'rm -rf "${TMPDIR}" || true' EXIT

echo "========================================="
echo "Fast Segmentation Array Job"
echo "========================================="
echo "Array Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-?}  Date: $(date)"
echo "Files: ${FILE_START}-${FILE_END}  batch_size=${BATCH_SIZE}"
echo "Images dir: ${IMAGES_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "========================================="

python -u "${PROJECT_ROOT}/faster-segmentation-test/batch_segmentation_fast.py" \
  --images-dir          "${IMAGES_DIR}" \
  --output-dir          "${OUTPUT_DIR}" \
  --temp-dir            "${TMPDIR}" \
  --breast-model-path   "${BREAST_MODEL}" \
  --vessel-model-path   "${VESSEL_MODEL}" \
  --file-start          "${FILE_START}" \
  --file-end            "${FILE_END}" \
  --batch-size          "${BATCH_SIZE}" \
  --num-workers         3 \
  --preprocess-workers  4 \
  --resume \
  --cleanup

echo "========================================="
echo "Task ${SLURM_ARRAY_TASK_ID} completed: $(date)"
echo "========================================="
