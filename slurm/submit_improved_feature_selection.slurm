#!/bin/bash
#SBATCH --job-name=featsel-clinical
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -eo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-${REPO_ROOT:-$PWD}}"

# Step 1: Build features CSV with clinical columns enabled
CONFIG="${REPO_ROOT}/configs/ispy2_with_clinical.yaml"
EXPERIMENT_DIR="${REPO_ROOT}/experiments/ispy2_with_clinical"
INPUT_CSV="${EXPERIMENT_DIR}/features_engineered_labeled.csv"

mkdir -p "${REPO_ROOT}/logs" "${EXPERIMENT_DIR}"
cd "${REPO_ROOT}"

source ~/.bashrc || true
if command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook -s bash)"
  micromamba activate vanguard
fi

echo "=== Step 1: Build features with clinical columns ==="
PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python -c "
from tabular.tabular_cohort import prepare_data
from config import load_config
from pathlib import Path
config = load_config('${CONFIG}')
prepare_data(config, Path('${EXPERIMENT_DIR}'))
"

echo ""
echo "=== Step 2: Feature selection (elastic-net) with clinical + site-grouped CV ==="
PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python scripts/feature_selection.py \
  "${INPUT_CSV}" \
  -o "${EXPERIMENT_DIR}/feature_selection_clinical_sitecv" \
  --encode-clinical \
  --site-group-cv

echo ""
echo "=== Step 3: XGBoost interaction with clinical + site-grouped CV ==="
PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python scripts/xgboost_interaction_pipeline.py \
  "${INPUT_CSV}" \
  -o "${EXPERIMENT_DIR}/xgboost_interaction_clinical_sitecv" \
  --encode-clinical \
  --site-group-cv

echo ""
echo "=== Step 4: Baselines without improvements (for comparison) ==="

echo "--- Feature selection: clinical only (standard CV) ---"
PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python scripts/feature_selection.py \
  "${INPUT_CSV}" \
  -o "${EXPERIMENT_DIR}/feature_selection_clinical_stdcv" \
  --encode-clinical

echo "--- XGBoost: clinical only (standard CV) ---"
PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python scripts/xgboost_interaction_pipeline.py \
  "${INPUT_CSV}" \
  -o "${EXPERIMENT_DIR}/xgboost_interaction_clinical_stdcv" \
  --encode-clinical

echo ""
echo "=== All runs complete ==="
