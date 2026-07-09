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
#
# No .sh wrapper for this one -- submit with PROJECT_ROOT set explicitly, e.g.
# from the repo root:
#   PROJECT_ROOT="$(pwd)" sbatch segmentation/slurm/submit_validate_speed_and_accuracy.slurm

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

MAMAMIA_ROOT="${MAMAMIA_ROOT:-/gpfs/data/karczmar-lab/MAMA-MIA-syn60868042}"
IMAGES_DIR="${IMAGES_DIR:-${MAMAMIA_ROOT}/images}"
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
