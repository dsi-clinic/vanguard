#!/bin/bash
#SBATCH --job-name=rad_concat_agg
#SBATCH --output=logs/sweep_concat_agg_%j.out
#SBATCH --error=logs/sweep_concat_agg_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=general
# ---------------------------------------------------------------------------
# Sweep: concat vs aggregate non-scalar handling
#
# 2 configs, 1 shared extraction + 2 training runs
# ---------------------------------------------------------------------------

set -euo pipefail

PROJ_DIR="$HOME/vanguard/radiomics_baseline"
SCRIPTS_DIR="${PROJ_DIR}/scripts"

eval "$(micromamba shell hook --shell bash)"
micromamba activate vanguard

cd "${PROJ_DIR}"
mkdir -p logs

echo "[$(date)] Starting concat vs aggregate sweep"

python "${SCRIPTS_DIR}/run_ablations.py" \
    "${PROJ_DIR}/configs/sweep_concat_vs_aggregate.yaml" \
    --generated-dir "${PROJ_DIR}/configs/generated"

echo "[$(date)] Concat vs aggregate sweep complete"
