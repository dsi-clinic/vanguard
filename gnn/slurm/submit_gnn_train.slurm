#!/usr/bin/env bash
#SBATCH --job-name=gnn-train-mvp
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# MVP training run for gnn/train.py: single train/val split, plain GCN, no
# k-fold CV yet (see gnn/README.md). Defaults point at the 8-case smoke cache
# built by gnn/build_dataset.py (see LAB_NOTEBOOK.md, job 12482766) so this
# validates the loader -> model -> training wiring cheaply before a run
# against the full ~1500-case cache.
#
# Usage:
#   sbatch gnn/slurm/submit_gnn_train.slurm
#   CACHE_DIR=/gpfs/.../gnn_cache sbatch gnn/slurm/submit_gnn_train.slurm

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
mkdir -p "${REPO_ROOT}/logs"
cd "${REPO_ROOT}"

set +u
source "${HOME}/.bashrc" || true
if command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook -s bash)"
  micromamba activate vanguard
fi
set -u

export PYTHONUNBUFFERED=1

ROOT="${ROOT:-/gpfs/data/karczmar-lab/workspaces/saritbose/centerlines_tc4d/studies}"
LABELS_PATH="${LABELS_PATH:-/gpfs/data/karczmar-lab/workspaces/saritbose/pcr_labels.csv}"
CACHE_DIR="${CACHE_DIR:-/gpfs/data/karczmar-lab/workspaces/spencervenancio/gnn_cache_smoke}"
ID_COLUMN="${ID_COLUMN:-case_id}"
LABEL_COLUMN="${LABEL_COLUMN:-pcr}"
NODE_FEATURES="${NODE_FEATURES:-peak_time,radius}"
VAL_FRAC="${VAL_FRAC:-0.3}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-4}"
DEVICE="${DEVICE:-cpu}"

ARGS=(
  python -u -m gnn.train
  --root "${ROOT}"
  --labels-path "${LABELS_PATH}"
  --cache-dir "${CACHE_DIR}"
  --id-column "${ID_COLUMN}"
  --label-column "${LABEL_COLUMN}"
  --node-features "${NODE_FEATURES}"
  --val-frac "${VAL_FRAC}"
  --seed "${SEED}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --device "${DEVICE}"
)

if [[ -n "${CASES:-}" ]]; then
  ARGS+=(--cases "${CASES}")
fi
if [[ -n "${DATASET_FILTER:-}" ]]; then
  ARGS+=(--dataset-filter "${DATASET_FILTER}")
fi
if [[ -n "${OUTDIR:-}" ]]; then
  ARGS+=(--outdir "${OUTDIR}")
fi

"${ARGS[@]}"
