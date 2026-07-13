#!/bin/bash
#SBATCH --job-name=ispy2_pca
#SBATCH --partition=express
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/gpfs/data/karczmar-lab/vanguard/batch_effect_inputs/logs/ispy2_pca_%j.log

set -euo pipefail

# Activate the project environment (see AGENTS.md).
set +u
eval "$(micromamba shell hook -s bash)"
micromamba activate vanguard
set -u

REPO="$HOME/vanguard"
DATA=/gpfs/data/karczmar-lab/vanguard/batch_effect_inputs
OUT=/ess/scratch/scratch1/t-9sbose/ispy2_pca_batch_effect

python3 "$REPO/scripts/batch_effect/pca_ispy2_batch_effect.py" \
    --features "$DATA/features_full_labeled_native.csv" \
    --spacing  "$DATA/spacing_by_hospital.csv" \
    --outdir   "$OUT"

echo "DONE -> $OUT"
