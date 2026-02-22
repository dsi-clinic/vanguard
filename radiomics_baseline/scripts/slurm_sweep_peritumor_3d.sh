#!/bin/bash
#SBATCH --job-name=rad_peri_3d
#SBATCH --output=logs/sweep_peritumor_3d_%j.out
#SBATCH --error=logs/sweep_peritumor_3d_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --partition=general
# ---------------------------------------------------------------------------
# Sweep: peritumor configuration — 3D extraction only (6 configs)
#
#   peri_mode=3d × force_2d={false,true} × peri_radius={0, 2.5, 5} mm
#
# 3D extractions take ~55 min each → ~5.5 h total; fits within 6 h wall time.
# Submit alongside slurm_sweep_peritumor_2d.sh (they run independently).
#
# Submit:
#   sbatch scripts/slurm_sweep_peritumor_3d.sh
# ---------------------------------------------------------------------------

set -euo pipefail

PROJ_DIR="$HOME/vanguard/radiomics_baseline"
SCRIPTS_DIR="${PROJ_DIR}/scripts"

eval "$(micromamba shell hook --shell bash)"
micromamba activate vanguard

cd "${PROJ_DIR}"
mkdir -p logs

echo "[$(date)] Starting peritumor 3D sweep"
echo "  Config: configs/sweep_peritumor_3d.yaml"
echo "  Configs: 6  |  Extractions: 6  (~55 min each)"

python "${SCRIPTS_DIR}/run_ablations.py" \
    "${PROJ_DIR}/configs/sweep_peritumor_3d.yaml" \
    --generated-dir "${PROJ_DIR}/configs/generated"

echo "[$(date)] Peritumor 3D sweep complete"
