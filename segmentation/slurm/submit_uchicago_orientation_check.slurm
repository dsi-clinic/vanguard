#!/usr/bin/env bash
#SBATCH --job-name=uc-orient
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=logs/uc-orient-%j.out
#SBATCH --error=logs/uc-orient-%j.err

set -eo pipefail

# Scores the three candidate UChicagoDataset.preprocess transforms (passthrough,
# mirrored, order_only) against the BreastDivider ground-truth breast masks, by
# running the pinned breast model on one phase-0 case per sub-source under each.
#
# Why this and not the smoke test: the smoke test only proves inference RUNS.
# `mirrored` and `order_only` produce identically-shaped arrays, so both run --
# they differ only in whether the anatomy is left-right reflected. Only scoring
# against ground truth separates them. See the script docstring.
#
# Submit from the repo root:
#   PROJECT_ROOT="$(pwd)" sbatch segmentation/slurm/submit_uchicago_orientation_check.slurm

cd "${PROJECT_ROOT}"
source ~/.bashrc || true
micromamba activate vanguard || true

WORK_DIR="/tmp/uc_orient_${SLURM_JOB_ID:-local}"
mkdir -p "${WORK_DIR}" results
trap 'rm -rf "${WORK_DIR}" || true' EXIT

BREAST_MODEL="${BREAST_MODEL:-${PROJECT_ROOT}/vanguard-blood-vessel-segmentation/trained_models/breast_model.pth}"
OUT_CSV="${OUT_CSV:-${PROJECT_ROOT}/results/uchicago_orientation_check.csv}"

echo "=== UChicago orientation check ==="
echo "Node: $(hostname)  Date: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u "${PROJECT_ROOT}/scripts/validate_uchicago_orientation.py" \
  --breast-model-path "${BREAST_MODEL}" \
  --work-dir          "${WORK_DIR}" \
  --out-csv           "${OUT_CSV}"

echo "=== finished: $(date) ==="
echo "Results: ${OUT_CSV}"
