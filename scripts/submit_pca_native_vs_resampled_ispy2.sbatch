#!/bin/bash
#SBATCH --job-name=ispy2_pca_nvr
#SBATCH --partition=express
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/gpfs/data/karczmar-lab/workspaces/saritbose/logs/ispy2_pca_nvr_%j.log

set -euo pipefail

PY=/home/t-9sbose/micromamba/envs/vanguard/bin/python3
REPO="$HOME/vanguard"
NATIVE=/gpfs/data/karczmar-lab/workspaces/saritbose/tabular/all_datasets_run/features_full_labeled.csv
RESAMPLED=/ess/scratch/scratch1/t-9sbose/vanguard_experiments/independent_signal_resampled_ispy2/features_full_labeled.csv
SPACING=/gpfs/data/karczmar-lab/workspaces/saritbose/data_viz/spacing_by_hospital.csv
CLINICAL=/gpfs/data/karczmar-lab/MAMA-MIA-syn60868042/clinical_and_imaging_info.xlsx
OUT=/ess/scratch/scratch1/t-9sbose/ispy2_pca_native_vs_resampled

"$PY" "$REPO/scripts/pca_native_vs_resampled_ispy2.py" \
    --native "$NATIVE" \
    --resampled "$RESAMPLED" \
    --spacing "$SPACING" \
    --clinical "$CLINICAL" \
    --outdir "$OUT"

echo "DONE -> $OUT"
