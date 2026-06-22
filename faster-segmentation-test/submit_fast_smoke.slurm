#!/usr/bin/env bash
#SBATCH --job-name=fast-seg-smoke
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/fast-seg-smoke-%j.out
#SBATCH --error=logs/fast-seg-smoke-%j.err
set -eo pipefail

# Small GPU smoke test for the fast pipeline. Processes a few files (that already
# have ground truth) into a SEPARATE output dir, so it cannot disturb the live
# segmentation run. Validate afterwards with validate_outputs.py.

source ~/.bashrc || true
eval "$(micromamba shell hook -s bash)"
micromamba activate vanguard
export xml_catalog_files_libxml2=""

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p logs

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1

IMAGES_DIR="${IMAGES_DIR:-/ess/scratch/scratch1/annawoodard/MAMA-MIA-syn60868042/images}"
OUTPUT_DIR="${OUTPUT_DIR:-/ess/scratch/scratch1/t-9sbose/vessel_segmentations_fast_smoke}"
BREAST_MODEL="${BREAST_MODEL:-${PROJECT_ROOT}/vanguard-blood-vessel-segmentation/trained_models/breast_model.pth}"
VESSEL_MODEL="${VESSEL_MODEL:-${PROJECT_ROOT}/vanguard-blood-vessel-segmentation/trained_models/dv_model.pth}"
FILE_START="${FILE_START:-0}"
FILE_END="${FILE_END:-1}"
BATCH_SIZE="${BATCH_SIZE:-16}"

TMPDIR="/tmp/fast_seg_smoke_${SLURM_JOB_ID:-local}"
mkdir -p "${TMPDIR}"
trap 'rm -rf "${TMPDIR}" || true' EXIT

echo "=== Fast Segmentation Smoke Test ==="
echo "Node: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-?}  Date: $(date)"
echo "Files: ${FILE_START}-${FILE_END}  batch_size=${BATCH_SIZE}"
echo "Output: ${OUTPUT_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u "${PROJECT_ROOT}/faster-segmentation-test/batch_segmentation_fast.py" \
  --images-dir        "${IMAGES_DIR}" \
  --output-dir        "${OUTPUT_DIR}" \
  --temp-dir          "${TMPDIR}" \
  --breast-model-path "${BREAST_MODEL}" \
  --vessel-model-path "${VESSEL_MODEL}" \
  --file-start        "${FILE_START}" \
  --file-end          "${FILE_END}" \
  --batch-size        "${BATCH_SIZE}" \
  --num-workers       3 \
  --preprocess-workers 4 \
  --cleanup

echo "=== Smoke test finished: $(date) ==="
