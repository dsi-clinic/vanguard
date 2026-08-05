#!/usr/bin/env bash
#SBATCH --job-name=uchic-tumor-prep
#SBATCH --account=karczmar-lab
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x-%A-%a.out
#SBATCH --error=logs/slurm/%x-%A-%a.err

export PYTHONNOUSERSITE=1

eval "$(micromamba shell hook -s bash)"
micromamba activate vanguard
set -euo pipefail

: "${REPO_ROOT:?set REPO_ROOT}"
: "${RUN_ROOT:?set RUN_ROOT}"
: "${INVENTORY:?set INVENTORY}"
: "${CASE_MANIFEST:?set CASE_MANIFEST}"

cd "${REPO_ROOT}"

python -m preprocessing.tumor prepare-case \
  --output-root "${RUN_ROOT}" \
  --inventory "${INVENTORY}" \
  --case-manifest "${CASE_MANIFEST}" \
  --array-index "${SLURM_ARRAY_TASK_ID}"
