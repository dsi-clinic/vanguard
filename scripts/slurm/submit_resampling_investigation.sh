#!/usr/bin/env bash
#
# Resampling investigation: runs all 5 diagnostic scripts and writes outputs to
# /ess/scratch/scratch1/t-9sbose/resampled_investigation/
#
# LIGHTWEIGHT scripts (01, 02, 03, 05) run inline on the head node — they only
# read CSVs, JSON files, and NIfTI headers, which is well within head-node limits.
#
# HEAVY script (04) loads full NPZ arrays for multiple cases and is submitted to
# Slurm.
#
# USAGE
#   bash scripts/slurm/submit_resampling_investigation.sh
#
# Override any path via environment variables:
#   OUT_ROOT=... PARTITION=... bash scripts/slurm/submit_resampling_investigation.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
INV_SCRIPTS="${REPO_ROOT}/scripts/investigate_resampling"

PYTHON="${PYTHON:-${HOME}/micromamba/envs/vanguard/bin/python}"
PARTITION="${PARTITION:-tier1q}"

# --- Input paths ---
FEATURES_CSV="${FEATURES_CSV:-/ess/scratch/scratch1/t-9sbose/vanguard_experiments/independent_signal_resampled_ispy2/features_full_labeled.csv}"
BASELINE_CSV="${BASELINE_CSV:-${REPO_ROOT}/results/independent_signal_q3_summary.csv}"
RESAMPLED_SUMMARY_CSV="${RESAMPLED_SUMMARY_CSV:-/ess/scratch/scratch1/t-9sbose/vanguard_experiments/independent_signal_resampled_ispy2/ablation_summary.csv}"
RESAMPLED_MASK_DIR="${RESAMPLED_MASK_DIR:-/gpfs/data/karczmar-lab/workspaces/saritbose/resampled-tumor-masks}"
NATIVE_MASK_DIR="${NATIVE_MASK_DIR:-/ess/scratch/scratch1/annawoodard/MAMA-MIA-syn60868042/segmentations/expert}"
NATIVE_SEG_ROOT="${NATIVE_SEG_ROOT:-/gpfs/data/karczmar-lab/workspaces/saritbose/vessel_segmentations}"
RESAMPLED_SEG_ROOT="${RESAMPLED_SEG_ROOT:-/gpfs/data/karczmar-lab/workspaces/saritbose/resampled-vessel-segmentations}"
CENTERLINE_ROOT="${CENTERLINE_ROOT:-/gpfs/data/karczmar-lab/workspaces/saritbose/resamples_centerlines_tc4d}"

# --- Output paths ---
OUT_ROOT="${OUT_ROOT:-/ess/scratch/scratch1/t-9sbose/resampled_investigation}"
mkdir -p "${OUT_ROOT}"

# 5 cases for visual spot-check; chosen to cover different pCR outcomes if possible
SPOT_CHECK_CASES="ISPY2_100899 ISPY2_102011 ISPY2_102212 ISPY2_103301 ISPY2_103651"

echo "========================================================"
echo "Resampling investigation"
echo "  out_root : ${OUT_ROOT}"
echo "========================================================"
echo

# ----------------------------------------------------------------
# Script 01: Feature distribution audit
# (lightweight — reads one CSV, produces histograms)
# ----------------------------------------------------------------
echo "[1/5] Feature distribution audit ..."
"${PYTHON}" "${INV_SCRIPTS}/01_feature_distribution_audit.py" \
    --features-csv "${FEATURES_CSV}" \
    --out-dir "${OUT_ROOT}/01_feature_distribution"
echo "  Done."
echo

# ----------------------------------------------------------------
# Script 02: Spacing audit
# (lightweight — reads NIfTI headers only, does NOT load voxel data)
# ----------------------------------------------------------------
echo "[2/5] Voxel spacing audit ..."
"${PYTHON}" "${INV_SCRIPTS}/02_spacing_audit.py" \
    --resampled-mask-dir "${RESAMPLED_MASK_DIR}" \
    --native-mask-dir "${NATIVE_MASK_DIR}" \
    --out-dir "${OUT_ROOT}/02_spacing_audit" \
    --native-sample-n 50
echo "  Done."
echo

# ----------------------------------------------------------------
# Script 03: AUC comparison plot
# (lightweight — reads two CSVs)
# ----------------------------------------------------------------
echo "[3/5] AUC comparison plot ..."
"${PYTHON}" "${INV_SCRIPTS}/03_auc_comparison_plot.py" \
    --baseline-csv "${BASELINE_CSV}" \
    --resampled-csv "${RESAMPLED_SUMMARY_CSV}" \
    --out-dir "${OUT_ROOT}/03_auc_comparison"
echo "  Done."
echo

# ----------------------------------------------------------------
# Script 04: Segmentation visual check (HEAVY — submit to Slurm)
# ----------------------------------------------------------------
echo "[4/5] Segmentation visual check (submitting to Slurm) ..."
VIS_JOB_ID="$(
  sbatch --parsable \
    --partition="${PARTITION}" \
    --cpus-per-task=4 \
    --mem=32G \
    --time=01:00:00 \
    --output="${REPO_ROOT}/logs/resampling-vis-%j.out" \
    --error="${REPO_ROOT}/logs/resampling-vis-%j.err" \
    --wrap="bash -lc '
      eval \"\$(micromamba shell hook -s bash)\"
      micromamba activate vanguard
      cd ${REPO_ROOT}
      ${PYTHON} ${INV_SCRIPTS}/04_seg_visual_check.py \
          --native-seg-root ${NATIVE_SEG_ROOT} \
          --resampled-seg-root ${RESAMPLED_SEG_ROOT} \
          --cohort ISPY2 \
          --case-ids ${SPOT_CHECK_CASES} \
          --timepoint 0 \
          --out-dir ${OUT_ROOT}/04_seg_visual_check
    '"
)"
echo "  Submitted job ${VIS_JOB_ID} (check: squeue -j ${VIS_JOB_ID})"
echo

# ----------------------------------------------------------------
# Script 05: Morphometry JSON physical-plausibility audit
# (lightweight — reads JSON files; morphometry JSONs are ~1MB each, no voxels)
# ----------------------------------------------------------------
echo "[5/5] Morphometry JSON plausibility audit ..."
SPACING_CSV="${OUT_ROOT}/02_spacing_audit/resampled_spacings.csv"
"${PYTHON}" "${INV_SCRIPTS}/05_morphometry_json_audit.py" \
    --centerline-root "${CENTERLINE_ROOT}" \
    --cohort ISPY2 \
    --resampled-spacing-csv "${SPACING_CSV}" \
    --out-dir "${OUT_ROOT}/05_morphometry_audit"
echo "  Done."
echo

echo "========================================================"
echo "Investigation outputs written to: ${OUT_ROOT}"
echo ""
echo "Slurm visual check job: ${VIS_JOB_ID}"
echo "  Monitor: squeue -j ${VIS_JOB_ID}"
echo "  Logs:    ${REPO_ROOT}/logs/resampling-vis-${VIS_JOB_ID}.out"
echo ""
echo "Output structure:"
echo "  01_feature_distribution/  — block stats, NaN rates, feature histograms"
echo "  02_spacing_audit/         — voxel spacing comparison, morph scale factors"
echo "  03_auc_comparison/        — baseline vs resampled grouped bar chart"
echo "  04_seg_visual_check/      — native vs resampled MIP overlays (Slurm job)"
echo "  05_morphometry_audit/     — vessel length/radius physical plausibility"
echo "========================================================"
