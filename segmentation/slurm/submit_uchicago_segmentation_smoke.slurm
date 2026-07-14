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
# ORIENTATION: HFDP already canonicalized the anatomical axis directions.
# SimpleITK returns each phase as (z, y, x), so UChicagoDataset.preprocess()
# converts array order only to the model's (y, x, z). A pass-through put the thin
# slice axis in the wrong model dimension; header-derived flips overcorrected the
# data. See cohorts/README.md for the validated source-to-model contract.
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

# CASE_IDS (optional): a file of case ids (one per line) or a comma-separated
# list, selecting exactly which exams to run. Mutually exclusive with
# PATIENT_LIMIT, which can only ever reach the FIRST N cases in discovery order
# -- that is why the original smoke runs were all her2_naclike. Example:
#   CASE_IDS=segmentation/slurm/uchicago_qc_cases_simbiosys_uch_nac.txt
CASE_IDS="${CASE_IDS:-}"
if [[ -n "${CASE_IDS}" ]]; then
  SELECT_ARGS=(--case-ids "${CASE_IDS}")
  SELECT_DESC="case ids from ${CASE_IDS}"
else
  SELECT_ARGS=(--patient-limit "${PATIENT_LIMIT}")
  SELECT_DESC="first ${PATIENT_LIMIT} discovered patients"
fi

TMPDIR="/tmp/uc_seg_smoke_${SLURM_JOB_ID:-local}"
mkdir -p "${TMPDIR}"
trap 'rm -rf "${TMPDIR}" || true' EXIT

echo "=== UChicago Segmentation Smoke Test (${SELECT_DESC}) ==="
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
  "${SELECT_ARGS[@]}" \
  --batch-size         "${BATCH_SIZE}" \
  --num-workers        3 \
  --preprocess-workers 4 \
  --cleanup

echo "=== Smoke test finished: $(date) ==="
echo "Outputs under: ${OUTPUT_DIR}"
