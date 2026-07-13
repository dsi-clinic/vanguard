#!/usr/bin/env bash
#SBATCH --job-name=uc-seg-smoke-ext
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/uc-seg-smoke-ext-%j.out
#SBATCH --error=logs/uc-seg-smoke-ext-%j.err
set -eo pipefail

# External-breast-mask counterpart to submit_uchicago_segmentation_smoke.slurm.
# Same 5 UChicago patients, same vessel model, but STEP-2 breast-model
# inference is REPLACED with the ground-truth BreastDivider whole-breast masks
# shipped in the manifest TAR (--use-external-breast-masks). This isolates
# whether real breast masks change the vessel segmentation outcome vs. the
# model-predicted-breast-mask baseline.
#
# BASELINE TO COMPARE AGAINST:
#   /ess/scratch/scratch1/t-9sbose/uchicago_vessel_segmentations_smoke_v2
#
# CAVEAT (per mentor + data dictionary): there is ONE whole-breast mask per
# exam, generated from phase 0 only, and this preprocessing build did NOT
# include motion correction. The same phase-0 mask is therefore reused across
# all ~24 dynamic phases of an exam -- a documented approximation (assumes the
# breast does not move across phases), not a bug.
#
# Distinct OUTPUT_DIR and TMPDIR (with an _ext_ prefix) so this can never
# collide with or race against the baseline run -- a concurrency bug this
# session already hit once (see LAB_NOTEBOOK.md, 2026-07-13). RUN THIS SOLO:
# confirm `squeue -u $USER` shows no other uc-seg-smoke job first.
#
# Submit from the repo root:
#   PROJECT_ROOT="$(pwd)" sbatch segmentation/slurm/submit_uchicago_segmentation_smoke_external_breast.slurm

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
OUTPUT_DIR="${OUTPUT_DIR:-/ess/scratch/scratch1/t-9sbose/uchicago_vessel_segmentations_smoke_external_breast}"
VESSEL_MODEL="${VESSEL_MODEL:-${PROJECT_ROOT}/vanguard-blood-vessel-segmentation/trained_models/dv_model.pth}"
BREAST_MODEL="${BREAST_MODEL:-${PROJECT_ROOT}/vanguard-blood-vessel-segmentation/trained_models/breast_model.pth}"
# Persist the converted external masks outside the (cleaned) temp dir so the
# comparison script can read exactly what STEP-3 consumed.
PERSIST_DIR="${PERSIST_DIR:-/ess/scratch/scratch1/t-9sbose/uchicago_breast_masks_external}"
PATIENT_LIMIT="${PATIENT_LIMIT:-5}"
BATCH_SIZE="${BATCH_SIZE:-16}"

TMPDIR="/tmp/uc_seg_smoke_ext_${SLURM_JOB_ID:-local}"
mkdir -p "${TMPDIR}" "${PERSIST_DIR}"
trap 'rm -rf "${TMPDIR}" || true' EXIT

echo "=== UChicago Segmentation Smoke Test -- EXTERNAL breast masks (${PATIENT_LIMIT} patients) ==="
echo "Node: $(hostname)  GPU: ${CUDA_VISIBLE_DEVICES:-?}  Date: $(date)"
echo "Manifest root: ${UCHICAGO_ROOT}"
echo "Output: ${OUTPUT_DIR}"
echo "Persisted external masks: ${PERSIST_DIR}"
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
  --use-external-breast-masks \
  --external-breast-mask-persist-dir "${PERSIST_DIR}" \
  --cleanup

echo "=== Smoke test (external breast masks) finished: $(date) ==="
echo "Outputs under: ${OUTPUT_DIR}"
echo "Compare against baseline: /ess/scratch/scratch1/t-9sbose/uchicago_vessel_segmentations_smoke_v2"
