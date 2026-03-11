#!/bin/bash
#SBATCH --job-name=rad_image_types
#SBATCH --output=logs/sweep_image_types_%j.out
#SBATCH --error=logs/sweep_image_types_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --partition=general
# ---------------------------------------------------------------------------
# Sweep: image-type comparison — 4 configs, 2 new extractions
#
#   Raw phases baseline: covered by exp_peri5_multiphase_logreg (already done)
#
#   1. Kinetic maps only        (E_early, E_peak, slope_in, slope_out, AUC)  [-5]
#   2. Subtraction images only  (wash_in, wash_out)                           [-2]
#   3. Raw phases + kinetic maps                                               [-7]
#   4. All: raw + kinetic + subtraction                                        [-9]
#
# Configs -2 and -5 have completed extractions and will skip to training only.
# Configs -7 and -9 require new extractions (~60-90 min each at 8 CPUs).
#
# Prerequisite: kinetic maps AND subtraction images must already exist.
#   Check with: ls kinetic_maps/DUKE_001/
#   If subtraction maps are missing, regenerate with:
#     sbatch scripts/slurm_generate_kinetic_maps.sh
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
echo "  Configs: 4  |  New extractions: 2 (-7, -9); existing: 2 (-2, -5)"

python "${SCRIPTS_DIR}/run_ablations.py" \
    "${PROJ_DIR}/configs/sweep_kinetic_maps.yaml" \
    --generated-dir "${PROJ_DIR}/configs/generated"

echo "[$(date)] Image-type sweep complete"
