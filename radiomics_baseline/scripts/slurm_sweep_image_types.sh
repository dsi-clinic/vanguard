#!/bin/bash
#SBATCH --job-name=rad_image_types
#SBATCH --output=logs/sweep_image_types_%j.out
#SBATCH --error=logs/sweep_image_types_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --partition=general
# ---------------------------------------------------------------------------
# Sweep: image-type comparison — 5 configs, 5 unique extractions
#
#   1. Raw DCE phases only
#   2. Kinetic maps only  (E_early, E_peak, slope_in, slope_out, AUC)
#   3. Subtraction images only  (wash_in, wash_out)
#   4. Raw phases + kinetic maps
#   5. All: raw phases + kinetic maps + subtraction images
#
# Prerequisite: kinetic maps AND subtraction images must already exist.
#   Check with: ls kinetic_maps/DUKE_001/
#   If subtraction maps are missing, regenerate with:
#     sbatch scripts/slurm_generate_kinetic_maps.sh  (add --generate-subtraction)
#
# Submit:
#   sbatch scripts/slurm_sweep_image_types.sh
# ---------------------------------------------------------------------------

set -euo pipefail

PROJ_DIR="$HOME/vanguard/radiomics_baseline"
SCRIPTS_DIR="${PROJ_DIR}/scripts"

eval "$(micromamba shell hook --shell bash)"
micromamba activate vanguard

cd "${PROJ_DIR}"
mkdir -p logs

echo "[$(date)] Starting image-type comparison sweep"
echo "  Config: configs/sweep_kinetic_maps.yaml"
echo "  Configs: 5  |  Extractions: 5"

python "${SCRIPTS_DIR}/run_ablations.py" \
    "${PROJ_DIR}/configs/sweep_kinetic_maps.yaml" \
    --generated-dir "${PROJ_DIR}/configs/generated"

echo "[$(date)] Image-type sweep complete"
