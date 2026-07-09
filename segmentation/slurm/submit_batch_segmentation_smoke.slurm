#!/usr/bin/env bash
#SBATCH --job-name=batch-seg-smoke
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/batch-seg-smoke-%j.out
#SBATCH --error=logs/batch-seg-smoke-%j.err
set -eo pipefail

# Small GPU smoke test for the batch segmentation pipeline. Processes a few
# files (that already have ground truth) into a SEPARATE output dir, so it
# cannot disturb the live segmentation run. Validate afterwards with
# validate_outputs.py.
#
# No .sh wrapper for this one -- submit with PROJECT_ROOT set explicitly, e.g.
# from the repo root:
#   PROJECT_ROOT="$(pwd)" sbatch segmentation/slurm/submit_batch_segmentation_smoke.slurm

source ~/.bashrc || true
eval "$(micromamba shell hook -s bash)"
micromamba activate vanguard
export xml_catalog_files_libxml2=""

# Not derived from $SLURM_SUBMIT_DIR (caller's cwd at submission, not this
# script's location) or ${BASH_SOURCE[0]} (sbatch may run a spooled copy) --
# see submit_batch_segmentation_array.slurm for the full rationale.
PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT must be set, e.g. PROJECT_ROOT=\$(pwd) sbatch ... from the repo root}"
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

echo "=== Batch Segmentation Smoke Test ==="
echo "Node: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-?}  Date: $(date)"
echo "Files: ${FILE_START}-${FILE_END}  batch_size=${BATCH_SIZE}"
echo "Output: ${OUTPUT_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u "${PROJECT_ROOT}/segmentation/batch_segmentation.py" \
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
