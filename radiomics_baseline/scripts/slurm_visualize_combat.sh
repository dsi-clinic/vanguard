#!/bin/bash
#SBATCH --job-name=combat_viz
#SBATCH --output=/home/summe/vanguard/radiomics_baseline/logs/combat_viz_%j.out
#SBATCH --error=/home/summe/vanguard/radiomics_baseline/logs/combat_viz_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --partition=general
# ---------------------------------------------------------------------------
# Generate ComBat harmonization diagnostic visualizations.
#
# Run AFTER extraction is complete (needs features_train_final.csv).
#
# Usage:
#   sbatch scripts/slurm_visualize_combat.sh
#
# To change the extraction directory or harmonization mode, edit the
# variables below.
# ---------------------------------------------------------------------------

set -euo pipefail

REPO_DIR="/home/summe/vanguard/radiomics_baseline"
SCRIPTS_DIR="${REPO_DIR}/scripts"

# --- Configuration: edit these as needed ---
FEATURES_DIR="${REPO_DIR}/outputs/shared_extraction/rerun_bin100_kinsubonly"
LABELS="${REPO_DIR}/labels.csv"
OUTPUT_DIR="${REPO_DIR}/outputs/combat_viz"
HARMONIZATION_MODE="combat_param"
CV_FOLDS=5
# -------------------------------------------

mkdir -p "${REPO_DIR}/logs" "${OUTPUT_DIR}"

eval "$(micromamba shell hook --shell bash)"
micromamba activate vanguard

cd "${REPO_DIR}"

echo "[$(date)] Starting ComBat visualizations"
echo "  Features dir: ${FEATURES_DIR}"
echo "  Harmonization: ${HARMONIZATION_MODE}"
echo "  Output: ${OUTPUT_DIR}"

python "${SCRIPTS_DIR}/visualize_combat.py" \
    --features-dir "${FEATURES_DIR}" \
    --labels "${LABELS}" \
    --output-dir "${OUTPUT_DIR}" \
    --harmonization-mode "${HARMONIZATION_MODE}" \
    --cv-folds "${CV_FOLDS}"

echo "[$(date)] Done."
