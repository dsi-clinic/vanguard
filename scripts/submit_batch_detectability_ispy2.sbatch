#!/bin/bash
#SBATCH --job-name=ispy2_batchdet
#SBATCH --partition=express
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/gpfs/data/karczmar-lab/vanguard/batch_effect_inputs/logs/ispy2_batchdet_%j.log

set -euo pipefail

PY=/home/t-9sbose/micromamba/envs/vanguard/bin/python3
REPO="$HOME/vanguard"
DATA=/gpfs/data/karczmar-lab/vanguard/batch_effect_inputs
NATIVE="$DATA/features_full_labeled_native.csv"
RESAMPLED="$DATA/features_full_labeled_resampled_808.csv"
CLINICAL=/gpfs/data/karczmar-lab/MAMA-MIA-syn60868042/clinical_and_imaging_info.xlsx
OUT=/ess/scratch/scratch1/t-9sbose/ispy2_batch_detectability

"$PY" "$REPO/scripts/batch_detectability_ispy2.py" \
    --native "$NATIVE" \
    --resampled "$RESAMPLED" \
    --clinical "$CLINICAL" \
    --outdir "$OUT"

echo "DONE -> $OUT"
