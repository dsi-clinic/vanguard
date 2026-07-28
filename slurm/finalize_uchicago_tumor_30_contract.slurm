#!/bin/bash
#SBATCH --job-name=uchic-tumor-final30
#SBATCH --account=karczmar-lab
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -eo pipefail

cd /ess/home/home1/aakrithiram/vanguard

eval "$(micromamba shell hook -s bash)"
micromamba activate vanguard
set -u

mkdir -p logs/slurm

RUN_ROOT="${RUN_ROOT:-/ess/scratch/scratch1/aakrithiram/uchic_tumor_masks_contract_30}"

echo "host=$(hostname)"
echo "date=$(date -Is)"
echo "python=$(which python)"
echo "run_root=${RUN_ROOT}"

python -m preprocessing.tumor finalize-cohort \
  --output-root "${RUN_ROOT}"
