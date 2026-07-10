#!/bin/bash
#SBATCH --job-name=ispy2_pca
#SBATCH --partition=express
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/gpfs/data/karczmar-lab/workspaces/saritbose/logs/ispy2_pca_%j.log

set -euo pipefail

# Compute-node python (per CLAUDE.md: the vanguard micromamba env, hardcoded).
PY=/home/t-9sbose/micromamba/envs/vanguard/bin/python3
REPO="$HOME/vanguard"
DATA=/gpfs/data/karczmar-lab/workspaces/saritbose
OUT=/ess/scratch/scratch1/t-9sbose/ispy2_pca_batch_effect

"$PY" "$REPO/scripts/pca_ispy2_batch_effect.py" \
    --features "$DATA/tabular/all_datasets_run/features_full_labeled.csv" \
    --spacing  "$DATA/data_viz/spacing_by_hospital.csv" \
    --outdir   "$OUT"

echo "DONE -> $OUT"
