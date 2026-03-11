#!/bin/bash
#SBATCH --job-name=rad_test_kfold
#SBATCH --output=logs/test_kfold_%j.out
#SBATCH --error=logs/test_kfold_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --partition=general
# ---------------------------------------------------------------------------
# Quick smoke-test for the evaluation-framework k-fold integration.
#
# Points radiomics_train.py directly at pre-existing feature CSVs so no
# extraction is re-run.  Skips grid-search to keep wall-time short.
#
# Submit from radiomics_baseline/:
#   sbatch scripts/slurm_test_kfold.sh
# ---------------------------------------------------------------------------

set -euo pipefail

PROJ_DIR="$HOME/vanguard/radiomics_baseline"
SCRIPTS_DIR="${PROJ_DIR}/scripts"

eval "$(micromamba shell hook --shell bash)"
micromamba activate vanguard

cd "${PROJ_DIR}"
mkdir -p logs

# Pre-existing extraction: kinetic maps only (5 features × ~1492 patients)
EXTRACT_DIR="${PROJ_DIR}/outputs/shared_extraction/kinetic_maps_logreg_image_patterns-5_classifier-logistic"
TRAIN_CSV="${EXTRACT_DIR}/features_train_final.csv"
TEST_CSV="${EXTRACT_DIR}/features_test_final.csv"

# Separate output so we never overwrite real sweep results
OUT_DIR="${PROJ_DIR}/outputs/test_kfold_refactor/training"

echo "[$(date)] Launching training-only test (k-fold via evaluation framework)"

python "${SCRIPTS_DIR}/radiomics_train.py" \
    --train-features "${TRAIN_CSV}" \
    --test-features  "${TEST_CSV}" \
    --labels         "${PROJ_DIR}/labels.csv" \
    --output         "${OUT_DIR}" \
    --classifier     logistic \
    --logreg-penalty elasticnet \
    --logreg-l1-ratio 0.5 \
    --corr-threshold 0.9 \
    --k-best         50 \
    --cv-folds       5 \
    --include-subtype

echo "[$(date)] Done — outputs in ${OUT_DIR}"
echo ""
echo "Key files to check:"
echo "  ${OUT_DIR}/metrics.json           <- flat keys (auc_test, auc_train_cv, …)"
echo "  ${OUT_DIR}/predictions.csv        <- test predictions"
echo "  ${OUT_DIR}/plots/roc_curve.png    <- framework ROC"
echo "  ${OUT_DIR}/cv/metrics.json        <- mean±std AUC over 5 folds"
echo "  ${OUT_DIR}/cv/metrics_per_fold.json"
echo "  ${OUT_DIR}/cv/predictions.csv     <- OOF predictions (fold-labelled)"
