#!/usr/bin/env bash
#SBATCH --job-name=missing33-seg
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-32%16
#SBATCH --output=logs/missing33-seg-%A-%a.out
#SBATCH --error=logs/missing33-seg-%A-%a.err
set -eo pipefail

source ~/.bashrc || true
eval "$(micromamba shell hook -s bash)"
micromamba activate vanguard
export xml_catalog_files_libxml2=""

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
IMAGES_DIR="${IMAGES_DIR:-/ess/scratch/scratch1/annawoodard/MAMA-MIA-syn60868042/images}"
OUTPUT_DIR="${OUTPUT_DIR:-/ess/scratch/scratch1/t-9sbose/vessel_segmentations}"
BREAST_MODEL="${BREAST_MODEL:-${PROJECT_ROOT}/vanguard-blood-vessel-segmentation/trained_models/breast_model.pth}"
VESSEL_MODEL="${VESSEL_MODEL:-${PROJECT_ROOT}/vanguard-blood-vessel-segmentation/trained_models/dv_model.pth}"

export TMPDIR="/tmp/missing33_seg_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${TMPDIR}"
trap 'rm -rf "${TMPDIR}" || true' EXIT

mkdir -p logs

# One entry per case: "CASE_ID FILE_START FILE_END"
# Indices are positions in the sorted file list from images_dir (computed 2026-06-22).
CASES=(
    "DUKE_012 21 24"
    "DUKE_019 25 28"
    "DUKE_021 29 32"
    "DUKE_022 33 37"
    "DUKE_045 72 75"
    "DUKE_046 76 79"
    "DUKE_069 115 119"
    "DUKE_101 155 159"
    "DUKE_119 194 198"
    "DUKE_142 236 239"
    "DUKE_168 276 279"
    "DUKE_233 351 354"
    "DUKE_234 355 358"
    "DUKE_258 396 398"
    "DUKE_307 474 477"
    "DUKE_378 554 558"
    "DUKE_400 595 599"
    "DUKE_489 710 713"
    "DUKE_491 714 717"
    "ISPY2_239061 2873 2878"
    "ISPY2_255388 2987 2992"
    "ISPY2_255535 2993 2998"
    "ISPY2_275626 3101 3106"
    "ISPY2_277848 3107 3112"
    "ISPY2_277888 3113 3118"
    "ISPY2_287300 3185 3190"
    "ISPY2_287961 3191 3196"
    "ISPY2_299840 3269 3274"
    "ISPY2_311316 3341 3346"
    "ISPY2_311455 3347 3352"
    "ISPY2_313243 3353 3358"
    "ISPY2_317641 3383 3388"
    "ISPY2_318293 3389 3394"
)

read -r CASE_ID FILE_START FILE_END <<< "${CASES[$SLURM_ARRAY_TASK_ID]}"

echo "========================================="
echo "Missing-33 Segmentation Job"
echo "Array Job ID : ${SLURM_ARRAY_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Case         : ${CASE_ID}  (files ${FILE_START}-${FILE_END})"
echo "Node         : $(hostname)"
echo "Date         : $(date)"
echo "========================================="

cd "${PROJECT_ROOT}"

python -u "${PROJECT_ROOT}/segmentation/batch_segmentation.py" \
  --images-dir        "${IMAGES_DIR}" \
  --output-dir        "${OUTPUT_DIR}" \
  --temp-dir          "${TMPDIR}" \
  --breast-model-path "${BREAST_MODEL}" \
  --vessel-model-path "${VESSEL_MODEL}" \
  --file-start        "${FILE_START}" \
  --file-end          "${FILE_END}" \
  --resume \
  --cleanup

echo "========================================="
echo "Task ${SLURM_ARRAY_TASK_ID} (${CASE_ID}) completed: $(date)"
echo "========================================="
