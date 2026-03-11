#!/bin/bash
#SBATCH --job-name=rad_subtypes
#SBATCH --output=logs/sweep_subtypes_%j.out
#SBATCH --error=logs/sweep_subtypes_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --partition=general
# ---------------------------------------------------------------------------
# Sweep: subtype-specific models — 5 configs, 1 shared extraction
#
#   triple_negative  (n≈494, PCR 37%)
#   luminal_a        (n≈381, PCR 17%)
#   luminal          (n≈206, PCR 12%)
#   her2_enriched    (n≈166, PCR 46%)
#   luminal_b        (n≈155, PCR 37%)
#
# This sweep reuses the same peri-5 mm, 2-phase extraction as the
# feature-selection sweep.  Submit it AFTER slurm_sweep_feature_sel.sh
# so the shared extraction is guaranteed to exist:
#
#   JOB_FS=$(sbatch --parsable scripts/slurm_sweep_feature_sel.sh)
#   sbatch --dependency=afterok:${JOB_FS} scripts/slurm_sweep_subtypes.sh
#
# (submit_all.sh handles this automatically)
# ---------------------------------------------------------------------------

set -euo pipefail

PROJ_DIR="$HOME/vanguard/radiomics_baseline"
SCRIPTS_DIR="${PROJ_DIR}/scripts"

eval "$(micromamba shell hook --shell bash)"
micromamba activate vanguard

cd "${PROJ_DIR}"
mkdir -p logs

echo "[$(date)] Starting subtype-specific model sweep"
echo "  Config: configs/sweep_test_subtypes.yaml"
echo "  Configs: 5  |  Extractions: 1 shared (reused from feature-sel sweep)"

python "${SCRIPTS_DIR}/run_ablations.py" \
    "${PROJ_DIR}/configs/sweep_test_subtypes.yaml" \
    --generated-dir "${PROJ_DIR}/configs/generated"

echo "[$(date)] Subtype sweep complete"
