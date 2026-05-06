#!/bin/bash
#SBATCH --job-name=xgboost-interaction
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -eo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-${REPO_ROOT:-$PWD}}"
INPUT_CSV="${INPUT_CSV:-vanguard/experiments/second_order_features/features_engineered_labeled.csv}"
OUTDIR="${OUTDIR:-experiments/xgboost_interaction}"

mkdir -p "${REPO_ROOT}/logs"
cd "${REPO_ROOT}"

source ~/.bashrc || true
if command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook -s bash)"
  micromamba activate vanguard
fi

echo "=== XGBoost interaction pipeline ==="
echo "Input: ${INPUT_CSV}"
echo "Output: ${OUTDIR}"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU detected"
echo ""

PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python scripts/xgboost_interaction_pipeline.py \
  "${INPUT_CSV}" \
  -o "${OUTDIR}"
