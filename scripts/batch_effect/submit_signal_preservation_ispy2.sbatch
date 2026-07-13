#!/bin/bash
#SBATCH --job-name=ispy2_sigpres
#SBATCH --partition=express
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/gpfs/data/karczmar-lab/vanguard/batch_effect_inputs/logs/ispy2_sigpres_%j.log

set -euo pipefail

# Activate the project environment (see AGENTS.md).
set +u
eval "$(micromamba shell hook -s bash)"
micromamba activate vanguard
set -u

REPO="$HOME/vanguard"
DATA=/gpfs/data/karczmar-lab/vanguard/batch_effect_inputs
NATIVE="$DATA/features_full_labeled_native.csv"
RESAMPLED="$DATA/features_full_labeled_resampled_808.csv"
OUT=/ess/scratch/scratch1/t-9sbose/ispy2_signal_preservation

python3 "$REPO/scripts/batch_effect/signal_preservation_ispy2.py" \
    --native "$NATIVE" \
    --resampled "$RESAMPLED" \
    --outdir "$OUT"

echo "DONE -> $OUT"
