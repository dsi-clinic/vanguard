#!/usr/bin/env bash
#SBATCH --job-name=fast-seg-task91-rerun
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/ess/home/home1/t-9sbose/vanguard/logs/fast-seg-task91-rerun-%j.out
#SBATCH --error=/ess/home/home1/t-9sbose/vanguard/logs/fast-seg-task91-rerun-%j.err
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

IMAGES_DIR="${IMAGES_DIR:-/ess/scratch/scratch1/t-9sbose/resampled_segmentation_flat}"
OUTPUT_DIR="${OUTPUT_DIR:-/gpfs/data/karczmar-lab/workspaces/saritbose/resampled-vessel-segmentations}"
BREAST_MODEL="${BREAST_MODEL:-${PROJECT_ROOT}/vanguard-blood-vessel-segmentation/trained_models/breast_model.pth}"
VESSEL_MODEL="${VESSEL_MODEL:-${PROJECT_ROOT}/vanguard-blood-vessel-segmentation/trained_models/dv_model.pth}"
FILE_START="${FILE_START:-1820}"
FILE_END="${FILE_END:-1839}"
Z_DIVISION="${Z_DIVISION:-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"

TMPDIR="/tmp/fast_seg_task91_rerun_${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}"
trap 'rm -rf "${TMPDIR}" || true' EXIT

echo "========================================="
echo "Fast Segmentation Task-91 Rerun"
echo "========================================="
echo "Job ID  : ${SLURM_JOB_ID}"
echo "Node    : $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-?}  Date: $(date)"
echo "Files   : ${FILE_START}-${FILE_END}  z_division=${Z_DIVISION}  batch_size=${BATCH_SIZE}"
echo "Images  : ${IMAGES_DIR}"
echo "Output  : ${OUTPUT_DIR}"
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
  --z-division          "${Z_DIVISION}" \
  --resume \
  --cleanup

echo "========================================="
echo "Task-91 rerun completed: $(date)"
echo "========================================="
