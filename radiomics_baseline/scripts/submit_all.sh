#!/bin/bash
# ---------------------------------------------------------------------------
# submit_all.sh — Submit the full experimental sweep
#
# Job graph (all jobs start immediately — no inter-job dependencies needed
# because shared extractions already exist from prior runs)
# --------------------------------------------------------------------------
#
#  [image_types] [peri_3d] [peri_2d] [feature_sel] [subtypes] [sites]
#       |             |         |           |             |         |
#       └─────────────┴─────────┴───────────┴─────────────┴─────────┘
#                                                                    |
#                                                      [site_analysis]
#
# Config summary
# --------------
#   image_types   : 4 configs,  2 new extractions  sweep_kinetic_maps.yaml
#                   (-2 subtraction, -5 kinetic done; -7 and -9 are new)
#   peri_3d       : 6 configs,  1 new extraction   sweep_peritumor_3d.yaml
#                   (5 done; only 3d+force2d+5mm remaining)
#   peri_2d       : 6 configs,  6 new extractions  sweep_peritumor_2d.yaml
#   feature_sel   : 6 configs,  0 new extractions  sweep_feature_selection.yaml
#                   (all done; re-runs training only)
#   subtypes      : 5 configs,  0 extractions      sweep_test_subtypes.yaml
#                   (all done; re-runs training only)
#   sites         : 4 configs,  0 extractions      sweep_sites.yaml
#                   (new, reuses peri5_multiphase_logreg features)
#   site_analysis : (uses existing features)        site_analysis.py
#   ──────────────────────────────────────────────────────────────────────
#   Total         : 31 training runs, ~9 new extractions
#
# Prerequisites
# -------------
#   1. Kinetic maps + subtraction images must exist:
#        ls kinetic_maps/DUKE_001/   # should show wash_in, wash_out, kinetic_*
#      If missing, run first:
#        sbatch scripts/slurm_generate_kinetic_maps.sh
#
#   2. peri5_multiphase_logreg extraction must exist (for sites and subtypes):
#        ls outputs/shared_extraction/
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
echo "  image_types   4 configs,  2 new extractions"
echo "  peri_3d       6 configs,  1 new extraction  (5 already done)"
echo "  peri_2d       6 configs,  6 new extractions"
echo "  feature_sel   6 configs,  0 new extractions (already done)"
echo "  subtypes      5 configs,  0 extractions     (already done)"
echo "  sites         4 configs,  0 extractions     (new, training only)"
echo "  site_analysis             (uses existing features)"
echo ""

if [[ "$DRY_RUN" == "--dry" ]]; then
    echo "[DRY RUN] Would submit:"
    echo "  JOB_IT=\$(sbatch --parsable ${SCRIPTS_DIR}/slurm_sweep_image_types.sh)"
    echo "  JOB_P3=\$(sbatch --parsable ${SCRIPTS_DIR}/slurm_sweep_peritumor_3d.sh)"
    echo "  JOB_P2=\$(sbatch --parsable ${SCRIPTS_DIR}/slurm_sweep_peritumor_2d.sh)"
    echo "  JOB_FS=\$(sbatch --parsable ${SCRIPTS_DIR}/slurm_sweep_feature_sel.sh)"
    echo "  JOB_ST=\$(sbatch --parsable ${SCRIPTS_DIR}/slurm_sweep_subtypes.sh)"
    echo "  JOB_SI=\$(sbatch --parsable ${SCRIPTS_DIR}/slurm_sweep_sites.sh)"
    echo "  JOB_SA=\$(sbatch --parsable ${SCRIPTS_DIR}/slurm_site_analysis.sh)"
    exit 0
fi

# ---- image-type sweep (starts immediately) ----
JOB_IT=$(sbatch --parsable "${SCRIPTS_DIR}/slurm_sweep_image_types.sh")
echo "Submitted image-types sweep    : job ${JOB_IT}  (slurm_sweep_image_types.sh)"

# ---- peritumor 3D sweep (starts immediately; skips 5 completed configs) ----
JOB_P3=$(sbatch --parsable "${SCRIPTS_DIR}/slurm_sweep_peritumor_3d.sh")
echo "Submitted peritumor-3d sweep   : job ${JOB_P3}  (slurm_sweep_peritumor_3d.sh)"

# ---- peritumor 2D sweep (starts immediately) ----
JOB_P2=$(sbatch --parsable "${SCRIPTS_DIR}/slurm_sweep_peritumor_2d.sh")
echo "Submitted peritumor-2d sweep   : job ${JOB_P2}  (slurm_sweep_peritumor_2d.sh)"

# ---- feature-selection sweep (re-runs training; extraction already done) ----
JOB_FS=$(sbatch --parsable "${SCRIPTS_DIR}/slurm_sweep_feature_sel.sh")
echo "Submitted feature-sel sweep    : job ${JOB_FS}  (slurm_sweep_feature_sel.sh)"

# ---- subtype sweep (re-runs training; extraction already done) ----
JOB_ST=$(sbatch --parsable "${SCRIPTS_DIR}/slurm_sweep_subtypes.sh")
echo "Submitted subtypes sweep       : job ${JOB_ST}  (slurm_sweep_subtypes.sh)"

# ---- per-site model sweep (training only; reuses existing extraction) ----
JOB_SI=$(sbatch --parsable "${SCRIPTS_DIR}/slurm_sweep_sites.sh")
echo "Submitted sites sweep          : job ${JOB_SI}  (slurm_sweep_sites.sh)"

# ---- site analysis (uses existing features) ----
JOB_SA=$(sbatch --parsable "${SCRIPTS_DIR}/slurm_site_analysis.sh")
echo "Submitted site analysis        : job ${JOB_SA}  (slurm_site_analysis.sh)"

echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f logs/sweep_image_types_${JOB_IT}.out"
echo "  tail -f logs/sweep_peritumor_3d_${JOB_P3}.out"
echo "  tail -f logs/sweep_peritumor_2d_${JOB_P2}.out"
echo "  tail -f logs/sweep_feature_sel_${JOB_FS}.out"
echo "  tail -f logs/sweep_subtypes_${JOB_ST}.out"
echo "  tail -f logs/sweep_sites_${JOB_SI}.out"
echo "  tail -f logs/site_analysis_${JOB_SA}.out"
echo ""
echo "Results will be written to:"
echo "  radiomics_baseline/outputs/          (per-experiment metrics.json + model.pkl)"
echo "  radiomics_baseline/configs/ablation_summary.csv  (merged results table)"
