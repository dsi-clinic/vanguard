#!/usr/bin/env bash
#SBATCH --job-name=uch-pcr-arm
#SBATCH --partition=gpuq
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
mkdir -p "${REPO_ROOT}/logs"
cd "${REPO_ROOT}"

set +u
source "${HOME}/.bashrc" || true
eval "$(micromamba shell hook -s bash)"
micromamba activate vanguard
set -u

: "${CACHE_DIR:?Set CACHE_DIR to the labeled UChicago temporal cache}"
: "${COHORT_MANIFEST:?Set COHORT_MANIFEST to the Anna-confirmed larger cohort}"
: "${GNN_CHECKPOINT:?Set GNN_CHECKPOINT to the pretrained graph encoder}"
: "${GRAPH_FREE_CHECKPOINT:?Set GRAPH_FREE_CHECKPOINT to the pretrained graph-free encoder}"
: "${ARM:?Set ARM to one of the four downstream arm names}"
: "${OUTDIR:?Set OUTDIR to the selected arm result directory}"

python -u -m gnn.temporal_finetune \
  --cache-dir "${CACHE_DIR}" \
  --cohort-manifest "${COHORT_MANIFEST}" \
  --gnn-checkpoint "${GNN_CHECKPOINT}" \
  --graph-free-checkpoint "${GRAPH_FREE_CHECKPOINT}" \
  --outdir "${OUTDIR}" \
  --tabular-reference "${TABULAR_REFERENCE:-experiments/uchicago_tabular_bar_floored/gate_auc_ci.csv}" \
  --epochs "${EPOCHS:-50}" \
  --learning-rate "${LEARNING_RATE:-0.0003}" \
  --inner-val-fraction "${INNER_VAL_FRACTION:-0.2}" \
  --seed "${SEED:-0}" \
  --arm "${ARM}" \
  --device cuda
