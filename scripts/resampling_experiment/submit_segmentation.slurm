#!/usr/bin/env bash
#SBATCH --job-name=duke001-seg
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/seg-%A-%a.out
#SBATCH --error=logs/seg-%A-%a.err
set -eo pipefail

# STEP 2: run vessel segmentation on each of the four versions.
# This is a 4-task array (one task per version).  Each task runs the existing
# batch_segmentation.py, which expects   <images-dir>/<case_id>/*.nii.gz   and
# writes <output-dir>/<SOURCE>/<case_id>/images/<...>_vessel_segmentation.npz.

source ~/.bashrc || true
eval "$(micromamba shell hook -s bash)"
micromamba activate vanguard
export xml_catalog_files_libxml2=""

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

mkdir -p logs

# Map the array index to a version folder.
VERSIONS=(version1_z2.0mm version2_z1.4mm version3_z1.0mm version4_original)
VERSION="${VERSIONS[$SLURM_ARRAY_TASK_ID]}"

PROJECT_ROOT="$HOME/vanguard"
EXP_ROOT="$HOME/vanguard_qc_pngs/resampling_experiment"
IMAGES_DIR="${EXP_ROOT}/volumes/${VERSION}"
OUTPUT_DIR="${EXP_ROOT}/segmentations/${VERSION}"
BREAST_MODEL="${PROJECT_ROOT}/vanguard-blood-vessel-segmentation/trained_models/breast_model.pth"
VESSEL_MODEL="${PROJECT_ROOT}/vanguard-blood-vessel-segmentation/trained_models/dv_model.pth"

# Per-task scratch for the intermediate preprocessing/inference files.
export TMPDIR="/tmp/duke001_seg_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${TMPDIR}"
cleanup() { rm -rf "${TMPDIR}" || true; }
trap cleanup EXIT

echo "========================================="
echo "Task ${SLURM_ARRAY_TASK_ID}  ->  version: ${VERSION}"
echo "Node: $(hostname)   GPU: ${CUDA_VISIBLE_DEVICES}   Date: $(date)"
echo "Images dir: ${IMAGES_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "========================================="

python -u "${PROJECT_ROOT}/segmentation/batch_segmentation.py" \
    --images-dir        "${IMAGES_DIR}" \
    --output-dir        "${OUTPUT_DIR}" \
    --temp-dir          "${TMPDIR}" \
    --breast-model-path "${BREAST_MODEL}" \
    --vessel-model-path "${VESSEL_MODEL}" \
    --cleanup

echo "Task ${SLURM_ARRAY_TASK_ID} (${VERSION}) complete: $(date)"
