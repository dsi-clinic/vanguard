#!/bin/bash
#SBATCH --job-name=ispy2_nvr
#SBATCH --partition=express
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/gpfs/data/karczmar-lab/workspaces/saritbose/logs/ispy2_nvr_%j.log

set -euo pipefail

PY=/home/t-9sbose/micromamba/envs/vanguard/bin/python3
REPO="$HOME/vanguard"
NATIVE=/gpfs/data/karczmar-lab/workspaces/saritbose/tabular/all_datasets_run/features_full_labeled.csv
RESAMPLED=/ess/scratch/scratch1/t-9sbose/vanguard_experiments/independent_signal_resampled_ispy2/features_full_labeled.csv
OUT=/ess/scratch/scratch1/t-9sbose/ispy2_native_vs_resampled_pcr

"$PY" "$REPO/scripts/compare_native_resampled_pcr_ispy2.py" \
    --native "$NATIVE" \
    --resampled "$RESAMPLED" \
    --outdir "$OUT"

echo "DONE -> $OUT"
