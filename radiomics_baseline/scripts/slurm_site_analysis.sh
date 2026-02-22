#!/bin/bash
#SBATCH --job-name=rad_site_analysis
#SBATCH --output=logs/site_analysis_%j.out
#SBATCH --error=logs/site_analysis_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --partition=general
# ---------------------------------------------------------------------------
# Site-level A/B analysis — per-site evaluation + leave-one-site-out (LOSO)
#
# Trains on the existing mixed split and evaluates per clinical site
# (DUKE, ISPY1, ISPY2, NACT, ...).  Also runs LOSO to measure cross-site
# generalisation.
#
# Prerequisite: extraction must already exist at FEATURES_DIR.
# The default below points at the shared peri-5 mm, 2-phase extraction.
# Adjust FEATURES_DIR if you want to run on a different extraction.
#
# Submit:
#   sbatch scripts/slurm_site_analysis.sh
#
# Or after feature-sel sweep completes (ensures extraction exists):
#   sbatch --dependency=afterok:${JOB_FS} scripts/slurm_site_analysis.sh
# ---------------------------------------------------------------------------

set -euo pipefail

PROJ_DIR="$HOME/vanguard/radiomics_baseline"
SCRIPTS_DIR="${PROJ_DIR}/scripts"

# Point at whichever extraction you want to analyse by site.
# The shared extraction from the feature-sel / subtypes sweep lives here:
FEATURES_DIR="${PROJ_DIR}/outputs/shared_extraction/peri5_multiphase"

eval "$(micromamba shell hook --shell bash)"
micromamba activate vanguard

cd "${PROJ_DIR}"
mkdir -p logs

echo "[$(date)] Starting site-level A/B analysis"
echo "  Features : ${FEATURES_DIR}"
echo "  Output   : ${PROJ_DIR}/outputs/site_analysis"

python "${SCRIPTS_DIR}/site_analysis.py" \
    --features-dir   "${FEATURES_DIR}" \
    --labels         "${PROJ_DIR}/labels.csv" \
    --splits         "${PROJ_DIR}/splits_train_test_ready.csv" \
    --output         "${PROJ_DIR}/outputs/site_analysis" \
    --classifier     logistic \
    --logreg-penalty  elasticnet \
    --logreg-l1-ratio 0.5 \
    --corr-threshold  0.9 \
    --k-best          50 \
    --feature-selection kbest \
    --grid-search

echo "[$(date)] Site analysis complete"
echo "  Results: ${PROJ_DIR}/outputs/site_analysis/"
