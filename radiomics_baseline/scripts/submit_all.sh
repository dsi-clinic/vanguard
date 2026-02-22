#!/bin/bash
# ---------------------------------------------------------------------------
# submit_all.sh — Submit the full experimental sweep
#
# Job graph
# ---------
#
#  [image_types]   [peritumor]   [feature_sel] ──→ [subtypes]
#       |                |             |                 |
#       └────────────────┴─────────────┴─────────────────┘
#                                                         |
#                                           [site_analysis]  (runs immediately
#                                            on existing features, no dependency)
#
# image_types, peritumor, feature_sel, site_analysis start immediately.
# subtypes waits for feature_sel to finish (shared extraction dependency).
#
# Total jobs submitted: 5
#
# Config summary
# --------------
#   image_types   : 5 configs,  5 extractions    sweep_kinetic_maps.yaml
#   peritumor     : 12 configs, 12 extractions   sweep_test_peritumor.yaml
#   feature_sel   : 6 configs,  1 extraction     sweep_feature_selection.yaml
#   subtypes      : 5 configs,  0 extractions*   sweep_test_subtypes.yaml
#   site_analysis : (uses existing features)     site_analysis.py
#   ──────────────────────────────────────────
#   Total         : 28 training runs, ~18 extractions
#
# * subtypes reuses the extraction created by feature_sel
#
# Prerequisites
# -------------
#   1. Kinetic maps + subtraction images must exist:
#        ls kinetic_maps/DUKE_001/   # should show wash_in, wash_out, kinetic_*
#      If missing, rerun:
#        sbatch scripts/slurm_generate_kinetic_maps.sh  --generate-subtraction
#
#   2. At least one prior extraction must exist for site_analysis.
#      The script defaults to outputs/shared_extraction/peri5_multiphase.
#      Adjust FEATURES_DIR in slurm_site_analysis.sh if needed.
#
# Usage
# -----
#   bash scripts/submit_all.sh           # submit everything
#   bash scripts/submit_all.sh --dry     # print commands without submitting
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN="${1:-}"

mkdir -p "$HOME/vanguard/radiomics_baseline/logs"

echo "============================================================"
echo "  Full radiomics experimental sweep"
echo "============================================================"
echo ""
echo "  image_types   5 configs,  5 extractions"
echo "  peritumor    12 configs, 12 extractions"
echo "  feature_sel   6 configs,  1 extraction"
echo "  subtypes      5 configs,  0 extractions  (waits for feature_sel)"
echo "  site_analysis             (uses existing features)"
echo ""

if [[ "$DRY_RUN" == "--dry" ]]; then
    echo "[DRY RUN] Would submit:"
    echo "  JOB_IT=\$(sbatch --parsable ${SCRIPTS_DIR}/slurm_sweep_image_types.sh)"
    echo "  JOB_PT=\$(sbatch --parsable ${SCRIPTS_DIR}/slurm_sweep_peritumor.sh)"
    echo "  JOB_FS=\$(sbatch --parsable ${SCRIPTS_DIR}/slurm_sweep_feature_sel.sh)"
    echo "  JOB_ST=\$(sbatch --parsable --dependency=afterok:\${JOB_FS} ${SCRIPTS_DIR}/slurm_sweep_subtypes.sh)"
    echo "  JOB_SA=\$(sbatch --parsable ${SCRIPTS_DIR}/slurm_site_analysis.sh)"
    exit 0
fi

# ---- image types sweep (starts immediately) ----
JOB_IT=$(sbatch --parsable "${SCRIPTS_DIR}/slurm_sweep_image_types.sh")
echo "Submitted image-types sweep    : job ${JOB_IT}  (slurm_sweep_image_types.sh)"

# ---- peritumor sweep (starts immediately) ----
JOB_PT=$(sbatch --parsable "${SCRIPTS_DIR}/slurm_sweep_peritumor.sh")
echo "Submitted peritumor sweep      : job ${JOB_PT}  (slurm_sweep_peritumor.sh)"

# ---- feature-selection sweep (starts immediately; also creates shared extraction) ----
JOB_FS=$(sbatch --parsable "${SCRIPTS_DIR}/slurm_sweep_feature_sel.sh")
echo "Submitted feature-sel sweep    : job ${JOB_FS}  (slurm_sweep_feature_sel.sh)"

# ---- subtype sweep (waits for feature_sel so shared extraction exists) ----
JOB_ST=$(sbatch --parsable \
    --dependency=afterok:${JOB_FS} \
    "${SCRIPTS_DIR}/slurm_sweep_subtypes.sh")
echo "Submitted subtypes sweep       : job ${JOB_ST}  (slurm_sweep_subtypes.sh, after ${JOB_FS})"

# ---- site analysis (starts immediately on existing features) ----
JOB_SA=$(sbatch --parsable "${SCRIPTS_DIR}/slurm_site_analysis.sh")
echo "Submitted site analysis        : job ${JOB_SA}  (slurm_site_analysis.sh)"

echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f logs/sweep_image_types_${JOB_IT}.out"
echo "  tail -f logs/sweep_peritumor_${JOB_PT}.out"
echo "  tail -f logs/sweep_feature_sel_${JOB_FS}.out"
echo "  tail -f logs/sweep_subtypes_${JOB_ST}.out"
echo "  tail -f logs/site_analysis_${JOB_SA}.out"
echo ""
echo "Results will be written to:"
echo "  radiomics_baseline/outputs/          (per-experiment metrics.json + model.pkl)"
echo "  radiomics_baseline/configs/ablation_summary.csv  (merged results table)"
