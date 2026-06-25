#!/usr/bin/env bash
# =============================================================================
# Reproducible entrypoint: run the FAST vessel-segmentation pipeline over the
# RESAMPLED dataset, writing results to the karczmar-lab GPFS workspace.
#
# This is the single documented command for that run. It wires together steps
# that must otherwise be typed by hand (and therefore would not be reproducible):
#   1. (re)build the flat symlink farm over the resampled data,
#   2. ensure the GPFS output directory exists,
#   3. submit the work to Slurm via the existing fast-segmentation launcher.
#
# It NEVER runs GPU inference itself — all compute goes through Slurm.
#
# Modes:
#   bash scripts/run_resampled_vessel_segmentation.sh            # full array (all 7926 files)
#   MODE=smoke bash scripts/run_resampled_vessel_segmentation.sh # 2-file GPU smoke test
#   DRY_RUN=1 bash scripts/run_resampled_vessel_segmentation.sh  # print, submit nothing
#
# Overridable inputs/outputs (env vars, with the defaults used for the real run):
#   RESAMPLED_ROOT  /ess/scratch/scratch1/t-9sbose/resampled_segmentation
#   FLAT_FARM       /ess/scratch/scratch1/t-9sbose/resampled_segmentation_flat
#   OUTPUT_DIR      /gpfs/data/karczmar-lab/workspaces/saritbose/resampled-vessel-segmentations
# Environment: micromamba env `vanguard` (activated inside the slurm scripts).
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESAMPLED_ROOT="${RESAMPLED_ROOT:-/ess/scratch/scratch1/t-9sbose/resampled_segmentation}"
FLAT_FARM="${FLAT_FARM:-/ess/scratch/scratch1/t-9sbose/resampled_segmentation_flat}"
OUTPUT_DIR="${OUTPUT_DIR:-/gpfs/data/karczmar-lab/workspaces/saritbose/resampled-vessel-segmentations}"
MODE="${MODE:-full}"      # full | smoke
DRY_RUN="${DRY_RUN:-0}"

echo "=== Resampled vessel-segmentation run ==="
echo "Project root  : ${PROJECT_ROOT}"
echo "Resampled root: ${RESAMPLED_ROOT}"
echo "Flat farm     : ${FLAT_FARM}"
echo "Output dir    : ${OUTPUT_DIR}"
echo "Mode          : ${MODE}"
echo

# --- Step 1: (re)build the flat symlink farm (idempotent) --------------------
echo "[1/3] Building flat symlink farm..."
RESAMPLED_ROOT="${RESAMPLED_ROOT}" FLAT_FARM="${FLAT_FARM}" \
  bash "${PROJECT_ROOT}/scripts/build_resampled_flat_farm.sh"
echo

# --- Step 2: ensure the GPFS output directory exists -------------------------
echo "[2/3] Ensuring output directory exists..."
mkdir -p "${OUTPUT_DIR}"
echo "  ${OUTPUT_DIR}"
echo

# --- Step 3: submit to Slurm -------------------------------------------------
echo "[3/3] Submitting to Slurm (mode=${MODE})..."
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[DRY RUN] No job submitted. Would run mode='${MODE}' with the above paths."
  exit 0
fi

if [[ "${MODE}" == "smoke" ]]; then
  # Single 2-file GPU job (files 0-1) to validate the chain end-to-end.
  IMAGES_DIR="${FLAT_FARM}" OUTPUT_DIR="${OUTPUT_DIR}" FILE_START=0 FILE_END=1 \
    sbatch --parsable "${PROJECT_ROOT}/faster-segmentation-test/submit_fast_smoke.slurm"
elif [[ "${MODE}" == "full" ]]; then
  # Full array over every resampled .nii.gz (the launcher computes the task count).
  IMAGES_DIR="${FLAT_FARM}" OUTPUT_DIR="${OUTPUT_DIR}" \
    bash "${PROJECT_ROOT}/faster-segmentation-test/submit_fast_array.sh"
else
  echo "ERROR: unknown MODE='${MODE}' (use 'full' or 'smoke')" >&2
  exit 1
fi
