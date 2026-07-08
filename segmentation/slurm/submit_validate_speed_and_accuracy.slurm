#!/usr/bin/env bash
#SBATCH --job-name=validate-speed-acc
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/validate-speed-acc-%j.out
#SBATCH --error=logs/validate-speed-acc-%j.err
set -eo pipefail

# FROZEN / historical: validate_speed_and_accuracy.py compared the old
# subprocess-based pipeline against the new in-process pipeline. The old
# pipeline's CLI was removed in the segmentation-folder consolidation cleanup,
# so this script can no longer run an actual old-vs-fast comparison -- kept
# only as a record of how validation_results.md was produced.

source ~/.bashrc || true
eval "$(micromamba shell hook -s bash)"
micromamba activate vanguard
export xml_catalog_files_libxml2=""

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
mkdir -p logs

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1

IMAGES_DIR="${IMAGES_DIR:-/ess/scratch/scratch1/annawoodard/MAMA-MIA-syn60868042/images}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/ess/scratch/scratch1/t-9sbose/validate_speed_and_accuracy}"
FILE_START="${FILE_START:-0}"
FILE_END="${FILE_END:-6}"

echo "=== Validate Speed and Accuracy ==="
echo "Node: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-?}  Date: $(date)"
echo "Files: ${FILE_START}-${FILE_END}"
echo "Output root: ${OUTPUT_ROOT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u "${PROJECT_ROOT}/segmentation/validate_speed_and_accuracy.py" \
  --images-dir   "${IMAGES_DIR}" \
  --file-start   "${FILE_START}" \
  --file-end     "${FILE_END}" \
  --output-root  "${OUTPUT_ROOT}"

echo "=== Validation finished: $(date) ==="
