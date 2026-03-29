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
# Run this before radiomics experiments that read kinetic maps.
#
# Usage:
#   sbatch slurm/slurm_generate_kinetic_maps.sh
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"
SCRIPTS_DIR="${REPO_DIR}/scripts"
OUT_ROOT="${REPO_DIR}"
IMAGES="${IMAGES:-/net/projects2/vanguard/MAMA-MIA-syn60868042/images}"
MASKS="${MASKS:-/net/projects2/vanguard/MAMA-MIA-syn60868042/segmentations/expert}"
SPLITS="${REPO_DIR}/splits_train_test_ready.csv"
KINETIC_OUT="${OUT_ROOT}/kinetic_maps"

eval "$(micromamba shell hook --shell bash)"
micromamba activate vanguard

cd "${OUT_ROOT}"
mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/outputs" "${KINETIC_OUT}"

echo "[$(date)] Starting kinetic map generation"

python "${SCRIPTS_DIR}/generate_kinetic_maps.py" \
    --images "${IMAGES}" \
    --masks  "${MASKS}" \
    --splits "${SPLITS}" \
    --output-dir "${KINETIC_OUT}" \
    --mask-pattern "{pid}.nii.gz" \
    --n-jobs 8 \
    --generate-tpeak-voxel \
    --generate-subtraction \
    --summary-csv "${OUT_ROOT}/outputs/kinetic_maps_summary.csv"

echo "[$(date)] Kinetic map generation complete"
