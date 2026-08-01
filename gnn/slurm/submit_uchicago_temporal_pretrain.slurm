#!/usr/bin/env bash
#SBATCH --job-name=uch-temporal-ssl
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

: "${CACHE_DIR:?Set CACHE_DIR to the unlabeled UChicago temporal cache}"
: "${OUTDIR:?Set OUTDIR to a new pretraining result directory}"

python -u -m gnn.pretrain.uchicago \
  --cache-dir "${CACHE_DIR}" \
  --outdir "${OUTDIR}" \
  --epochs "${EPOCHS:-100}" \
  --hidden-dim "${HIDDEN_DIM:-32}" \
  --num-gcn-layers "${NUM_GCN_LAYERS:-2}" \
  --num-gru-layers "${NUM_GRU_LAYERS:-1}" \
  --dropout "${DROPOUT:-0.2}" \
  --learning-rate "${LEARNING_RATE:-0.001}" \
  --val-fraction "${VAL_FRACTION:-0.2}" \
  --seed "${SEED:-0}" \
  --arm "${ARM:-both}" \
  --device cuda
