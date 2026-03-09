#!/bin/bash
#SBATCH --job-name=rad_kinetic
#SBATCH --output=logs/sweep_kinetic_%j.out
#SBATCH --error=logs/sweep_kinetic_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=general
# ---------------------------------------------------------------------------
# Sweep: kinetic maps vs raw phases vs combined × classifier
#
# 3 image configs × 3 classifiers = 9 experiments
#
# Submit AFTER kinetic map generation completes:
#   JOB_GEN=$(sbatch --parsable scripts/slurm_generate_kinetic_maps.sh)
#   sbatch --dependency=afterok:${JOB_GEN} scripts/slurm_sweep_kinetic.sh
# ---------------------------------------------------------------------------

set -euo pipefail

PROJ_DIR="$HOME/vanguard/radiomics_baseline"
SCRIPTS_DIR="${PROJ_DIR}/scripts"

eval "$(micromamba shell hook --shell bash)"
micromamba activate vanguard

cd "${PROJ_DIR}"
mkdir -p logs

echo "[$(date)] Starting kinetic maps sweep"

python "${SCRIPTS_DIR}/run_ablations.py" \
    "${PROJ_DIR}/configs/sweep_kinetic_maps.yaml" \
    --generated-dir "${PROJ_DIR}/configs/generated"

echo "[$(date)] Kinetic maps sweep complete"
