#!/bin/bash
#SBATCH --job-name=ispy2_arms
#SBATCH --partition=express
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/gpfs/data/karczmar-lab/vanguard/batch_effect_inputs/logs/ispy2_arms_%j.log

set -euo pipefail

PY=/home/t-9sbose/micromamba/envs/vanguard/bin/python3
REPO="$HOME/vanguard"
DATA=/gpfs/data/karczmar-lab/vanguard/batch_effect_inputs
NATIVE="$DATA/features_full_labeled_native.csv"
RESAMPLED="$DATA/features_full_labeled_resampled_808.csv"
CLINICAL=/gpfs/data/karczmar-lab/MAMA-MIA-syn60868042/clinical_and_imaging_info.xlsx
BASE=/ess/scratch/scratch1/t-9sbose

# arm label -> comma-separated feature prefixes
declare -A ARMS=(
    [morph]="morph_"
    [kinematic]="kinematic_"
    [vessel_all]="morph_,graph_,kinematic_"
)

for arm in "${!ARMS[@]}"; do
    prefixes="${ARMS[$arm]}"
    echo "===== batch detectability: $arm ($prefixes) ====="
    "$PY" "$REPO/scripts/batch_detectability_ispy2.py" \
        --native "$NATIVE" --resampled "$RESAMPLED" --clinical "$CLINICAL" \
        --feature-prefixes "$prefixes" \
        --outdir "$BASE/ispy2_batch_detectability_${arm}"

    echo "===== signal preservation: $arm ($prefixes) ====="
    "$PY" "$REPO/scripts/signal_preservation_ispy2.py" \
        --native "$NATIVE" --resampled "$RESAMPLED" \
        --feature-prefixes "$prefixes" \
        --outdir "$BASE/ispy2_signal_preservation_${arm}"
done

echo "DONE"
