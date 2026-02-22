#!/bin/bash
#SBATCH --job-name=rad_peritumor
#SBATCH --output=logs/sweep_peritumor_%j.out
#SBATCH --error=logs/sweep_peritumor_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --partition=general
# ---------------------------------------------------------------------------
# Sweep: peritumor configuration — 12 configs, 12 unique extractions
#
#   peri_mode    × {3d, 2d}
#   force_2d     × {false, true}
#   peri_radius  × {0, 2.5, 5} mm
#
# All three parameters affect extraction so every combination needs its own
# extraction run — 12 in total.  The 24-hour wall time is conservative;
# each extraction takes ~45-60 min at 8 CPUs for ~1000 patients.
#
# Submit:
#   sbatch scripts/slurm_sweep_peritumor.sh
# ---------------------------------------------------------------------------

set -euo pipefail

PROJ_DIR="$HOME/vanguard/radiomics_baseline"
SCRIPTS_DIR="${PROJ_DIR}/scripts"

eval "$(micromamba shell hook --shell bash)"
micromamba activate vanguard

cd "${PROJ_DIR}"
mkdir -p logs

echo "[$(date)] Starting peritumor configuration sweep"
echo "  Config: configs/sweep_test_peritumor.yaml"
echo "  Configs: 12  |  Extractions: 12 (all unique)"

python "${SCRIPTS_DIR}/run_ablations.py" \
    "${PROJ_DIR}/configs/sweep_test_peritumor.yaml" \
    --generated-dir "${PROJ_DIR}/configs/generated"

echo "[$(date)] Peritumor sweep complete"
