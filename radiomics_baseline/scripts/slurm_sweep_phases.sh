#!/bin/bash
#SBATCH --job-name=rad_phases
#SBATCH --output=logs/sweep_phases_%j.out
#SBATCH --error=logs/sweep_phases_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=general
# ---------------------------------------------------------------------------
# Sweep: peri radius × image phases × classifier
#
# 4 radii × 2 phase configs × 3 classifiers = 24 configs
# 8 unique extractions + 16 training-only runs
# ---------------------------------------------------------------------------

set -euo pipefail

PROJ_DIR="$HOME/vanguard/radiomics_baseline"
SCRIPTS_DIR="${PROJ_DIR}/scripts"

eval "$(micromamba shell hook --shell bash)"
micromamba activate vanguard

cd "${PROJ_DIR}"
mkdir -p logs

echo "[$(date)] Starting phase sweep"

python "${SCRIPTS_DIR}/run_ablations.py" \
    "${PROJ_DIR}/configs/sweep_phases.yaml" \
    --generated-dir "${PROJ_DIR}/configs/generated"

echo "[$(date)] Phase sweep complete"
