#!/bin/bash
#SBATCH --job-name=rad_sites
#SBATCH --output=logs/sweep_sites_%j.out
#SBATCH --error=logs/sweep_sites_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --partition=general
# ---------------------------------------------------------------------------
# Sweep: per-clinical-site models  (logistic / elasticnet only)
#
#   DUKE   n=280   PCR 23 %
#   ISPY1  n=167   PCR 29 %
#   ISPY2  n=980   PCR 32 %
#   NACT   n=64    PCR 17 %  (small — results should be interpreted cautiously)
#
# 4 site-specific models = 4 configs  (1 shared extraction, training only)
# Shared extraction: reuses peri5_multiphase_logreg features (must exist).
#
# Submit:
#   sbatch scripts/slurm_sweep_sites.sh
# ---------------------------------------------------------------------------

set -euo pipefail

PROJ_DIR="$HOME/vanguard/radiomics_baseline"
SCRIPTS_DIR="${PROJ_DIR}/scripts"

eval "$(micromamba shell hook --shell bash)"
micromamba activate vanguard

cd "${PROJ_DIR}"
mkdir -p logs

echo "[$(date)] Starting per-site model sweep"
echo "  Config: configs/sweep_sites.yaml"
echo "  Configs: 4  |  Extractions: 0 (reuses peri5_multiphase_logreg features)"

python "${SCRIPTS_DIR}/run_ablations.py" \
    "${PROJ_DIR}/configs/sweep_sites.yaml" \
    --generated-dir "${PROJ_DIR}/configs/generated"

echo "[$(date)] Per-site model sweep complete"
