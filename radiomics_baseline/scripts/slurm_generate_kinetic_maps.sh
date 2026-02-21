#!/bin/bash
#SBATCH --job-name=kinetic_maps
#SBATCH --output=logs/kinetic_maps_%j.out
#SBATCH --error=logs/kinetic_maps_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --partition=general
# ---------------------------------------------------------------------------
# Generate kinetic parameter maps for all patients.
# Run this BEFORE the kinetic extraction sweep.
#
# Usage:
#   sbatch scripts/slurm_generate_kinetic_maps.sh
# ---------------------------------------------------------------------------

set -euo pipefail

PROJ_DIR="$HOME/vanguard/radiomics_baseline"
SCRIPTS_DIR="${PROJ_DIR}/scripts"
IMAGES="/net/projects2/vanguard/MAMA-MIA-syn60868042/images"
MASKS="/net/projects2/vanguard/MAMA-MIA-syn60868042/segmentations/expert"
SPLITS="${PROJ_DIR}/splits_train_test_ready.csv"
KINETIC_OUT="${PROJ_DIR}/kinetic_maps"

eval "$(micromamba shell hook --shell bash)"
micromamba activate vanguard

cd "${PROJ_DIR}"
mkdir -p logs

echo "[$(date)] Starting kinetic map generation"

python "${SCRIPTS_DIR}/generate_kinetic_maps.py" \
    --images "${IMAGES}" \
    --masks  "${MASKS}" \
    --splits "${SPLITS}" \
    --output-dir "${KINETIC_OUT}" \
    --mask-pattern "{pid}.nii.gz" \
    --n-jobs 8 \
    --generate-tpeak-voxel \
    --summary-csv "${PROJ_DIR}/outputs/kinetic_maps_summary.csv"

echo "[$(date)] Kinetic map generation complete"
