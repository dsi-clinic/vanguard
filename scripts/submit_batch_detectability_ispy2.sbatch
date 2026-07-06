#!/bin/bash
#SBATCH --job-name=ispy2_batchdet
#SBATCH --partition=express
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/gpfs/data/karczmar-lab/workspaces/saritbose/logs/ispy2_batchdet_%j.log

set -euo pipefail

PY=/home/t-9sbose/micromamba/envs/vanguard/bin/python3
REPO="$HOME/vanguard"
NATIVE=/gpfs/data/karczmar-lab/workspaces/saritbose/tabular/all_datasets_run/features_full_labeled.csv
RESAMPLED=/ess/scratch/scratch1/t-9sbose/vanguard_experiments/independent_signal_resampled_ispy2/features_full_labeled.csv
CLINICAL=/gpfs/data/karczmar-lab/MAMA-MIA-syn60868042/clinical_and_imaging_info.xlsx
OUT=/ess/scratch/scratch1/t-9sbose/ispy2_batch_detectability

"$PY" "$REPO/scripts/batch_detectability_ispy2.py" \
    --native "$NATIVE" \
    --resampled "$RESAMPLED" \
    --clinical "$CLINICAL" \
    --outdir "$OUT"

echo "DONE -> $OUT"
