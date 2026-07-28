#!/bin/bash
#SBATCH --job-name=uchic-tumor-qc30
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

export MPLCONFIGDIR=/ess/scratch/scratch1/aakrithiram/tmp/matplotlib
mkdir -p "${MPLCONFIGDIR}" logs/slurm

RUN_ROOT="${RUN_ROOT:-/ess/scratch/scratch1/aakrithiram/uchic_tumor_masks_contract_30}"
OUT_DIR="${OUT_DIR:-/ess/home/home1/aakrithiram/vanguard/qc/uchic_tumor_masks_contract_30}"

echo "host=$(hostname)"
echo "date=$(date -Is)"
echo "python=$(which python)"
echo "run_root=${RUN_ROOT}"
echo "out_dir=${OUT_DIR}"

python scripts/qc_uchicago_tumor_contract_30.py \
  --run-root "${RUN_ROOT}" \
  --out-dir "${OUT_DIR}"
