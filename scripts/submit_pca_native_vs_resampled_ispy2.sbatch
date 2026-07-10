#!/bin/bash
#SBATCH --job-name=ispy2_pca_nvr
#SBATCH --partition=express
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/gpfs/data/karczmar-lab/vanguard/batch_effect_inputs/logs/ispy2_pca_nvr_%j.log

set -euo pipefail

PY=/home/t-9sbose/micromamba/envs/vanguard/bin/python3
REPO="$HOME/vanguard"
DATA=/gpfs/data/karczmar-lab/vanguard/batch_effect_inputs
NATIVE="$DATA/features_full_labeled_native.csv"
RESAMPLED="$DATA/features_full_labeled_resampled_808.csv"
SPACING="$DATA/spacing_by_hospital.csv"
CLINICAL=/gpfs/data/karczmar-lab/MAMA-MIA-syn60868042/clinical_and_imaging_info.xlsx
OUT=/ess/scratch/scratch1/t-9sbose/ispy2_pca_native_vs_resampled

"$PY" "$REPO/scripts/pca_native_vs_resampled_ispy2.py" \
    --native "$NATIVE" \
    --resampled "$RESAMPLED" \
    --spacing "$SPACING" \
    --clinical "$CLINICAL" \
    --outdir "$OUT"

echo "DONE -> $OUT"
