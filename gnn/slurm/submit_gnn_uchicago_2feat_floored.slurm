#!/usr/bin/env bash
#SBATCH --job-name=gnn-uch-2feat-floored
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# 2-feature ([peak_time, radius]) floored voxel GNN, 5 seeds, for the all-models
# comparison. Subsets the existing 7-feature floored voxel cache (no rebuild).
# Predictions land in voxel2feat_seed{n} so the gate can pool them.
#
#   sbatch gnn/slurm/submit_gnn_uchicago_2feat_floored.slurm

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

OUT_ROOT="/gpfs/data/karczmar-lab/workspaces/spencervenancio/experiments/gnn_uchicago_2feat_floored"
SEED="${SLURM_ARRAY_TASK_ID}"

python -u -m gnn.train \
    --config configs/uchicago_gnn_2feat_floored.yaml \
    --random-state "${SEED}" \
    --outdir "${OUT_ROOT}/voxel2feat_seed${SEED}"
