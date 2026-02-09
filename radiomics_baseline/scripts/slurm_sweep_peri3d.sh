#!/bin/bash
#SBATCH --job-name=rad_peri3d
#SBATCH --output=logs/sweep_peri3d_%j.out
#SBATCH --error=logs/sweep_peri3d_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=cpu
# ---------------------------------------------------------------------------
# Sweep B: 3D isotropic peritumor shell (original behaviour)
# ---------------------------------------------------------------------------

set -euo pipefail

PROJ_DIR="$HOME/vanguard/radiomics_baseline"
SCRIPTS_DIR="${PROJ_DIR}/scripts"

eval "$(micromamba shell hook --shell bash)"
micromamba activate vanguard
mkdir -p logs

echo "[$(date)] Starting 3D peritumor sweep"

python "${SCRIPTS_DIR}/run_ablations.py" \
    "${PROJ_DIR}/configs/sweep_peri3d_only.yaml" \
    --generated-dir "${PROJ_DIR}/configs/generated"

echo "[$(date)] 3D sweep complete"