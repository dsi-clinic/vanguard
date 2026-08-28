#!/bin/bash
#SBATCH --job-name=uchic-tumor-qc30
#SBATCH --account=karczmar-lab
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "${REPO_ROOT}"
eval "$(micromamba shell hook -s bash)"
micromamba activate vanguard
set -euo pipefail

: "${RUN_ROOT:?set RUN_ROOT to the prepared tumor output directory}"
OUT_DIR="${OUT_DIR:-${RUN_ROOT}/qc}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${RUN_ROOT}/.matplotlib}"
mkdir -p "${MPLCONFIGDIR}" logs/slurm

echo "host=$(hostname)"
echo "date=$(date -Is)"
echo "python=$(which python)"
echo "run_root=${RUN_ROOT}"
echo "out_dir=${OUT_DIR}"

python scripts/qc_uchicago_tumor_contract_30.py \
  --run-root "${RUN_ROOT}" \
  --out-dir "${OUT_DIR}"
