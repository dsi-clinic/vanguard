#!/usr/bin/env bash
#SBATCH --job-name=uc-seg-smoke
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/uc-seg-smoke-%j.out
#SBATCH --error=logs/uc-seg-smoke-%j.err
set -eo pipefail

# Step 4d smoke test: batch vessel segmentation on 5 real UChicago patients,
# via the newly-wired --dataset-name uchicago adapter path (see Step 4d commit
# message). Writes to a SEPARATE scratch output dir; cannot disturb any
# existing MAMA-MIA segmentation output.
#
# --mem=128G (vs. 32G in the analogous MAMA-MIA smoke script) is not a guess:
# a first run at 32G was OOM-killed during inference. UChicago has ~24 phases
# per patient at larger volume dimensions vs. MAMA-MIA's ~4-6, and
# run_inference_in_process loads every volume for the current stage into
# memory at once (Dataset3DSimple/Dataset3DDivided) -- 5 patients x 24 phases
# = 120 full volumes resident at once.
#
# ORIENTATION (was a known risk; now checked): UChicagoDataset.preprocess() used
# to be a pass-through, assuming the manifest's images already arrived correctly
# oriented for this model. That was wrong -- it fed mis-oriented volumes in and
# crashed inference on a tiling-coverage assertion. preprocess() now flips x/z
# (RAS -> LAI) before the base reorientation; see cohorts/README.md. One residual
# unknown: a pure left-right mirror can't be ruled out from near-symmetric breast
# anatomy, so treat left/right-specific conclusions from this run as provisional.
#
# Submit from the repo root:
#   PROJECT_ROOT="$(pwd)" sbatch segmentation/slurm/submit_uchicago_segmentation_smoke.slurm

source ~/.bashrc || true
eval "$(micromamba shell hook -s bash)"
micromamba activate vanguard
export xml_catalog_files_libxml2=""

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT must be set, e.g. PROJECT_ROOT=\$(pwd) sbatch ... from the repo root}"
mkdir -p logs

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1

UCHICAGO_ROOT="${UCHICAGO_ROOT:-/gpfs/data/karczmar-lab/vanguard/dce2d_internal_ultrafast_manifest}"
OUTPUT_DIR="${OUTPUT_DIR:-/ess/scratch/scratch1/t-9sbose/uchicago_vessel_segmentations_smoke}"
BREAST_MODEL="${BREAST_MODEL:-${PROJECT_ROOT}/vanguard-blood-vessel-segmentation/trained_models/breast_model.pth}"
VESSEL_MODEL="${VESSEL_MODEL:-${PROJECT_ROOT}/vanguard-blood-vessel-segmentation/trained_models/dv_model.pth}"
PATIENT_LIMIT="${PATIENT_LIMIT:-5}"
BATCH_SIZE="${BATCH_SIZE:-16}"

TMPDIR="/tmp/uc_seg_smoke_${SLURM_JOB_ID:-local}"
mkdir -p "${TMPDIR}"
trap 'rm -rf "${TMPDIR}" || true' EXIT

echo "=== UChicago Segmentation Smoke Test (${PATIENT_LIMIT} patients) ==="
echo "Node: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-?}  Date: $(date)"
echo "Manifest root: ${UCHICAGO_ROOT}"
echo "Output: ${OUTPUT_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u "${PROJECT_ROOT}/segmentation/batch_segmentation.py" \
  --dataset-name       uchicago \
  --dataset-root       "${UCHICAGO_ROOT}" \
  --output-dir         "${OUTPUT_DIR}" \
  --temp-dir           "${TMPDIR}" \
  --breast-model-path  "${BREAST_MODEL}" \
  --vessel-model-path  "${VESSEL_MODEL}" \
  --patient-limit      "${PATIENT_LIMIT}" \
  --batch-size         "${BATCH_SIZE}" \
  --num-workers        3 \
  --preprocess-workers 4 \
  --cleanup

echo "=== Smoke test finished: $(date) ==="
echo "Outputs under: ${OUTPUT_DIR}"
