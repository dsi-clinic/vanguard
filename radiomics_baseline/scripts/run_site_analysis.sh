#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_site_analysis.sh  —  A/B testing by clinical site
#
# Runs two complementary analyses on pre-extracted radiomics features:
#   1. Per-site evaluation  — train on the full mixed split, break down
#      AUC / sensitivity / specificity per clinical site on the test set.
#   2. Leave-one-site-out (LOSO)  — for each site, train on all other sites
#      and test on the held-out site (measures cross-site generalisation).
#
# Prerequisites
# -------------
# Extraction must already have been run.  Point --features-dir at the
# directory containing features_train_final.csv and features_test_final.csv.
# The default below uses the shared peri-5 mm extraction output.
#
# Outputs  (written to --output dir)
# ------------------------------------
#   per_site_metrics.json   — AUC / sens / spec per site (analysis 1)
#   loso_metrics.json       — AUC / sens / spec per site (analysis 2)
#   predictions_loso.csv    — patient-level LOSO predictions
#   roc_per_site.png        — overlaid ROC curves (analysis 1)
#   roc_loso.png            — overlaid ROC curves (analysis 2)
#   summary.csv             — both analyses in one table
#
# Usage
# -----
#   bash scripts/run_site_analysis.sh
#   bash scripts/run_site_analysis.sh --features-dir /path/to/extraction
#
# Run from the radiomics_baseline/ directory.
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

# ---- defaults (override on the command line) ----
FEATURES_DIR="${BASE_DIR}/outputs/shared_extraction/peri5_multiphase"
LABELS="${BASE_DIR}/labels.csv"
SPLITS="${BASE_DIR}/splits_train_test_ready.csv"
OUTPUT="${BASE_DIR}/outputs/site_analysis"

# Pass any extra CLI args straight through to site_analysis.py
EXTRA_ARGS=("$@")

echo "============================================================"
echo "  Site-level A/B analysis"
echo "  features-dir : ${FEATURES_DIR}"
echo "  output       : ${OUTPUT}"
echo "============================================================"

python "${SCRIPT_DIR}/site_analysis.py" \
    --features-dir  "${FEATURES_DIR}" \
    --labels        "${LABELS}" \
    --splits        "${SPLITS}" \
    --output        "${OUTPUT}" \
    --classifier    logistic \
    --logreg-penalty elasticnet \
    --logreg-l1-ratio 0.5 \
    --corr-threshold 0.9 \
    --k-best         50 \
    --feature-selection kbest \
    --grid-search \
    "${EXTRA_ARGS[@]}"
