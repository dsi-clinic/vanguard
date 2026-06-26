#!/usr/bin/env bash
# =============================================================================
# One-off rerun for Slurm array task 91 (files 1820-1839) from job 12284980.
#
# WHY THIS EXISTS
# ---------------
# Task 91 of the main segmentation array (job 12284980) failed with:
#   AssertionError: ISPY1_1228_0000 has regions that weren't analyzed;
#   configure x_y_divisions or z_division
# Root cause: ISPY1_1228 has z=307 slices after resampling. With z_division=3
# and input_dim=96, the tile step = (307-96)/(3-1) = 105 > 96, so middle slices
# are never covered. The assertion fires and kills the entire batch of 20 files.
# Fix: z_division=4 → step = (307-96)/(4-1) = 70 ≤ 96, all slices covered.
#
# This script reruns exactly those 20 files (1820-1839) with z_division=4.
# The --resume flag inside the slurm job means if any .npz already landed
# in OUTPUT_DIR, they are skipped (idempotent).
#
# Usage (from project root, on the login node):
#   bash faster-segmentation-test/submit_task91_rerun.sh
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

IMAGES_DIR="${IMAGES_DIR:-/ess/scratch/scratch1/t-9sbose/resampled_segmentation_flat}"
OUTPUT_DIR="${OUTPUT_DIR:-/gpfs/data/karczmar-lab/workspaces/saritbose/resampled-vessel-segmentations}"
BREAST_MODEL="${BREAST_MODEL:-${PROJECT_ROOT}/vanguard-blood-vessel-segmentation/trained_models/breast_model.pth}"
VESSEL_MODEL="${VESSEL_MODEL:-${PROJECT_ROOT}/vanguard-blood-vessel-segmentation/trained_models/dv_model.pth}"

# Task 91 range from the original FILES_PER_TASK=20 array
FILE_START=1820
FILE_END=1839
Z_DIVISION=4

echo "=== Task-91 rerun (z_division=${Z_DIVISION}) ==="
echo "Images dir : ${IMAGES_DIR}"
echo "Output dir : ${OUTPUT_DIR}"
echo "File range : ${FILE_START}-${FILE_END} (20 files)"
echo "Z division : ${Z_DIVISION}"
echo ""

mkdir -p /ess/home/home1/t-9sbose/vanguard/logs

JOB_ID=$(
  IMAGES_DIR="${IMAGES_DIR}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  BREAST_MODEL="${BREAST_MODEL}" \
  VESSEL_MODEL="${VESSEL_MODEL}" \
  FILE_START="${FILE_START}" \
  FILE_END="${FILE_END}" \
  Z_DIVISION="${Z_DIVISION}" \
  sbatch --parsable \
    --job-name=fast-seg-task91-rerun \
    --partition=gpuq \
    --gres=gpu:1 \
    --cpus-per-task=4 \
    --mem=32G \
    --time=02:00:00 \
    --output=/ess/home/home1/t-9sbose/vanguard/logs/fast-seg-task91-rerun-%j.out \
    --error=/ess/home/home1/t-9sbose/vanguard/logs/fast-seg-task91-rerun-%j.err \
    --export=ALL \
    "${PROJECT_ROOT}/faster-segmentation-test/submit_task91_rerun.slurm"
)

echo "Submitted job ${JOB_ID}"
echo "Logs: /ess/home/home1/t-9sbose/vanguard/logs/fast-seg-task91-rerun-${JOB_ID}.{out,err}"
