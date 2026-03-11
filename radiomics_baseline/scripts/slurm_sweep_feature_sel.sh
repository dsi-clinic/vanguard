#!/bin/bash
#SBATCH --job-name=rad_feat_sel
#SBATCH --output=logs/sweep_feature_sel_%j.out
#SBATCH --error=logs/sweep_feature_sel_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --partition=general
# ---------------------------------------------------------------------------
# Sweep: mRMR vs ANOVA k-best feature selection — 6 configs, 1 shared extraction
#
#   kbest  × K=10,20,50
#   mrmr   × K=10,20,50
#
# Extraction runs once (peri-5 mm, 2 phases); all 6 training runs reuse it.
# The shared extraction output is also used by slurm_sweep_subtypes.sh —
# submit that job with --dependency=afterok:<this job id> so the extraction
# is guaranteed to exist before the subtypes sweep starts.
#
# Submit:
#   JOB_FS=$(sbatch --parsable scripts/slurm_sweep_feature_sel.sh)
#   sbatch --dependency=afterok:${JOB_FS} scripts/slurm_sweep_subtypes.sh
# ---------------------------------------------------------------------------

set -euo pipefail

PROJ_DIR="$HOME/vanguard/radiomics_baseline"
SCRIPTS_DIR="${PROJ_DIR}/scripts"

eval "$(micromamba shell hook --shell bash)"
micromamba activate vanguard

cd "${PROJ_DIR}"
mkdir -p logs

echo "[$(date)] Starting feature-selection sweep"
echo "  Config: configs/sweep_feature_selection.yaml"
echo "  Configs: 6  |  Extractions: 1 shared"

python "${SCRIPTS_DIR}/run_ablations.py" \
    "${PROJ_DIR}/configs/sweep_feature_selection.yaml" \
    --generated-dir "${PROJ_DIR}/configs/generated"

echo "[$(date)] Feature-selection sweep complete"
