#!/bin/bash
#SBATCH --job-name=uchic-tumor-prep30
#SBATCH --account=karczmar-lab
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
eval "$(micromamba shell hook -s bash)"
micromamba activate vanguard
set -euo pipefail

mkdir -p logs/slurm

: "${RUN_ROOT:?set RUN_ROOT to the prepared tumor output directory}"

echo "host=$(hostname)"
echo "date=$(date -Is)"
echo "python=$(which python)"
echo "run_root=${RUN_ROOT}"

python -m preprocessing.tumor prepare-cohort \
  --output-root "${RUN_ROOT}"
