#!/usr/bin/env bash
#SBATCH --job-name=batch-seg-array
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/batch-seg-array-%A-%a.out
#SBATCH --error=logs/batch-seg-array-%A-%a.err
set -eo pipefail

# --- Activate your environment ---
source ~/.bashrc || true
eval "$(micromamba shell hook -s bash)"
micromamba activate vanguard

# Prevent libxml2 deactivate hook from failing under `set -u`
export xml_catalog_files_libxml2=""

# --- Make logs/temp unique per job ---
mkdir -p logs
export TMPDIR="/tmp/batch_seg_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
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
# PROJECT_ROOT must be passed in explicitly (submit_batch_segmentation_array.sh
# does this). It is NOT derived from $SLURM_SUBMIT_DIR (that's the caller's cwd
# at submission time, not this script's location -- wrong if you `cd
# segmentation/slurm` first, per the documented entry point) or from
# ${BASH_SOURCE[0]} (sbatch may run a spooled copy of this script, so its path
# is not reliably this repo checkout).
PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT must be set (submit via submit_batch_segmentation_array.sh, or pass PROJECT_ROOT=/path/to/vanguard explicitly)}"
MAMAMIA_ROOT="${MAMAMIA_ROOT:-/gpfs/data/karczmar-lab/MAMA-MIA-syn60868042}"
IMAGES_DIR="${IMAGES_DIR:-${MAMAMIA_ROOT}/images}"
OUTPUT_DIR="${OUTPUT_DIR:-/net/projects2/vanguard/vessel_segmentations}"
BREAST_MODEL="${BREAST_MODEL:-${PROJECT_ROOT}/vanguard-blood-vessel-segmentation/trained_models/breast_model.pth}"
VESSEL_MODEL="${VESSEL_MODEL:-${PROJECT_ROOT}/vanguard-blood-vessel-segmentation/trained_models/dv_model.pth}"

cd "${PROJECT_ROOT}"

TASK_ID=${SLURM_ARRAY_TASK_ID}
FILES_PER_TASK=${FILES_PER_TASK:-40}
FILE_START=$((TASK_ID * FILES_PER_TASK))
FILE_END=$((FILE_START + FILES_PER_TASK - 1))

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

python -u "${PROJECT_ROOT}/segmentation/batch_segmentation.py" \
  --images-dir         "${IMAGES_DIR}" \
  --output-dir         "${OUTPUT_DIR}" \
  --temp-dir           "${TMPDIR}" \
  --breast-model-path  "${BREAST_MODEL}" \
  --vessel-model-path  "${VESSEL_MODEL}" \
  --file-start         ${FILE_START} \
  --file-end           ${FILE_END} \
  --resume \
  --cleanup

echo ""
echo "========================================="
echo "Array Task ${SLURM_ARRAY_TASK_ID} Completed"
echo "Date: $(date)"
echo "========================================="
