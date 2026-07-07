#!/usr/bin/env bash
#SBATCH --job-name=gnn-train
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# K-fold training run for gnn/train.py: builds a per-graph cohort table from a
# VanguardCenterlineDataset cache and hands it to the shared evaluation
# framework (evaluation.build_splits.create_splits_for_dataframe), the same
# way tabular/train.py and deepsets/train.py do (see gnn/README.md). Defaults
# to the 8-case smoke config so this validates the loader -> model -> CV
# wiring cheaply before a run against the full cache (configs/gnn.yaml).
#
# Usage:
#   sbatch gnn/slurm/submit_gnn_train.slurm
#   CONFIG=configs/gnn.yaml sbatch gnn/slurm/submit_gnn_train.slurm

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

CONFIG="${CONFIG:-configs/gnn_smoke.yaml}"

ARGS=(python -u -m gnn.train --config "${CONFIG}")
if [[ -n "${OUTDIR:-}" ]]; then
  ARGS+=(--outdir "${OUTDIR}")
fi

"${ARGS[@]}"
