#!/usr/bin/env bash
#SBATCH --job-name=uchic-tumor-finalize
#SBATCH --account=karczmar-lab
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

export PYTHONNOUSERSITE=1

eval "$(micromamba shell hook -s bash)"
micromamba activate vanguard
set -euo pipefail

: "${REPO_ROOT:?set REPO_ROOT}"
: "${RUN_ROOT:?set RUN_ROOT}"
: "${INVENTORY:?set INVENTORY}"
: "${CASE_MANIFEST:?set CASE_MANIFEST}"
: "${COHORT_MANIFEST:?set COHORT_MANIFEST}"
: "${MODEL_DIR:?set MODEL_DIR}"

cd "${REPO_ROOT}"

ARGS=(python -m preprocessing.tumor finalize-cohort \
  --output-root "${RUN_ROOT}" \
  --inventory "${INVENTORY}" \
  --case-manifest "${CASE_MANIFEST}" \
  --cohort-manifest "${COHORT_MANIFEST}" \
  --model-dir "${MODEL_DIR}")
if [[ -n "${CENTERLINE_ROOT:-}" ]]; then
  ARGS+=(--centerline-root "${CENTERLINE_ROOT}")
fi
"${ARGS[@]}"
