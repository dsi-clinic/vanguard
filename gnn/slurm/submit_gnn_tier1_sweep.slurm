#!/usr/bin/env bash
#SBATCH --job-name=gnn-tier1-sweep
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# Tier-1 readout sweep (gnn/PLAN_advanced_modeling.md, D2.1): the mean+max
# readout (gnn_pooling="mean_max") across 5 seeds, on the SAME cache and folds
# as the Tier-0 sweep. Pairs seed-for-seed against the already-computed Tier-0
# (mean-pool) runs under experiments/gnn_tier0_sweep/tier0_seed{0..4}/ -- both
# arms share cache, features, folds, and all Tier-0 training knobs, so only the
# readout differs (D5). No cache rebuild: the readout is a model change, the
# graphs are identical, so we reuse gnn_cache_tier0_sweep.
#
# Only 5 jobs (mean_max x 5 seeds); the Tier-0 mean-pool comparison arm already
# ran in the Tier-0 sweep. Analyze with analysis.gnn_tier1_readout_gate_g1.
#
# Usage:
#   sbatch gnn/slurm/submit_gnn_tier1_sweep.slurm

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

OUT_ROOT="/gpfs/data/karczmar-lab/workspaces/spencervenancio/experiments/gnn_tier1_sweep"
SEED="${SLURM_ARRAY_TASK_ID}"
OUTDIR="${OUT_ROOT}/tier1_meanmax_seed${SEED}"

echo "tier1 task: seed=${SEED} pooling=mean_max outdir=${OUTDIR}"

python -u -m gnn.train \
    --config configs/gnn_tier1_meanmax.yaml \
    --random-state "${SEED}" \
    --outdir "${OUTDIR}"
