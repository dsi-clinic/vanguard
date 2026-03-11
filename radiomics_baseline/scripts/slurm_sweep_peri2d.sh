#!/bin/bash
#SBATCH --job-name=rad_peri2d
#SBATCH --output=logs/sweep_peri2d_%j.out
#SBATCH --error=logs/sweep_peri2d_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=general
# ---------------------------------------------------------------------------
# Sweep A: 2D peritumor ring only
#
# 1 peri_mode × 2 force_2d × 2 radii × 3 classifiers = 12 configs
# 4 unique extractions + 8 training-only runs
#
# Submit alongside slurm_sweep_peri3d.sh for 2× parallelism:
#   JOB_2D=$(sbatch --parsable slurm_sweep_peri2d.sh)
#   JOB_3D=$(sbatch --parsable slurm_sweep_peri3d.sh)
#   sbatch --dependency=afterok:${JOB_2D}:${JOB_3D} slurm_collect_results.sh
# ---------------------------------------------------------------------------

set -euo pipefail

PROJ_DIR="$HOME/vanguard/radiomics_baseline"
SCRIPTS_DIR="${PROJ_DIR}/scripts"

eval "$(micromamba shell hook --shell bash)"
micromamba activate vanguard
mkdir -p logs

echo "[$(date)] Starting 2D peritumor sweep"

python "${SCRIPTS_DIR}/run_ablations.py" \
    "${PROJ_DIR}/configs/sweep_2d_vs_3d.yaml" \
    --generated-dir "${PROJ_DIR}/configs/generated"

echo "[$(date)] 2D sweep complete"