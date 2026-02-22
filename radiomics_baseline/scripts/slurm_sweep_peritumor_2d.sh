#!/bin/bash
#SBATCH --job-name=rad_peri_2d
#SBATCH --output=logs/sweep_peritumor_2d_%j.out
#SBATCH --error=logs/sweep_peritumor_2d_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --partition=general
# ---------------------------------------------------------------------------
# Sweep: peritumor configuration — 2D extraction only (6 configs)
#
#   peri_mode=2d × force_2d={false,true} × peri_radius={0, 2.5, 5} mm
#
# 2D extractions take ~18 min each → ~2 h total; fits within 3 h wall time.
# Submit alongside slurm_sweep_peritumor_3d.sh (they run independently).
#
# Submit:
#   sbatch scripts/slurm_sweep_peritumor_2d.sh
# ---------------------------------------------------------------------------

set -euo pipefail

PROJ_DIR="$HOME/vanguard/radiomics_baseline"
SCRIPTS_DIR="${PROJ_DIR}/scripts"

eval "$(micromamba shell hook --shell bash)"
micromamba activate vanguard

cd "${PROJ_DIR}"
mkdir -p logs

echo "[$(date)] Starting peritumor 2D sweep"
echo "  Config: configs/sweep_peritumor_2d.yaml"
echo "  Configs: 6  |  Extractions: 6  (~18 min each)"

python "${SCRIPTS_DIR}/run_ablations.py" \
    "${PROJ_DIR}/configs/sweep_peritumor_2d.yaml" \
    --generated-dir "${PROJ_DIR}/configs/generated"

echo "[$(date)] Peritumor 2D sweep complete"
