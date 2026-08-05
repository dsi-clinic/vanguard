#!/usr/bin/env bash
#SBATCH --job-name=uchic-tumor-index
#SBATCH --account=karczmar-lab
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
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
: "${MODEL_DIR:?set MODEL_DIR}"

cd "${REPO_ROOT}"

python -m preprocessing.tumor index-cohort \
  --output-root "${RUN_ROOT}" \
  --inventory "${INVENTORY}" \
  --case-manifest "${CASE_MANIFEST}" \
  --model-dir "${MODEL_DIR}"
