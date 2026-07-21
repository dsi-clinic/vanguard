#!/usr/bin/env bash
#SBATCH --job-name=tc4d-array
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/tc4d-array-%A-%a.out
#SBATCH --error=logs/tc4d-array-%A-%a.err

set -euo pipefail

cd "${REPO_ROOT}"
source ~/.bashrc || true
if command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook -s bash)"
  micromamba activate vanguard
fi

LINE_NO=$((SLURM_ARRAY_TASK_ID + 1))
REL_PATH="$(sed -n "${LINE_NO}p" "${STUDY_LIST}")"
SITE="${REL_PATH%%/*}"
STUDY_ID="${REL_PATH##*/}"
INPUT_DIR="${INPUT_ROOT}/${SITE}"
OUTPUT_DIR="${OUTPUT_ROOT}/${SITE}/${STUDY_ID}"

# Dataset adapter (multi-dataset migration Step 5: required on every run).
# DATASET_COHORT left unset by default -> all four MAMA-MIA cohorts combined,
# matching this job's historical scope of processing sites across all cohorts.
MAMAMIA_ROOT="${MAMAMIA_ROOT:-/gpfs/data/karczmar-lab/MAMA-MIA-syn60868042}"
DATASET_COHORT="${DATASET_COHORT:-}"

ARGS=(
  python graph_extraction/run_skeleton_processing.py
  --case-id "${STUDY_ID}"
  --input-dir "${INPUT_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --dataset-name mamamia
  --dataset-root "${MAMAMIA_ROOT}"
)
if [[ -n "${DATASET_COHORT}" ]]; then
  ARGS+=(--dataset-cohort "${DATASET_COHORT}")
fi

if [[ "${FEATURES_ONLY:-0}" == "1" ]]; then
  ARGS+=(--features-only)
fi
if [[ "${FORCE_FEATURES:-0}" == "1" ]]; then
  ARGS+=(--force-features)
fi
if [[ "${FORCE_SKELETON:-0}" == "1" ]]; then
  ARGS+=(--force-skeleton)
fi
if [[ "${STRICT_QC:-0}" == "1" ]]; then
  ARGS+=(--strict-qc)
fi
if [[ "${NO_RENDER_MIP:-0}" == "1" ]]; then
  ARGS+=(--no-render-mip)
fi

"${ARGS[@]}"
