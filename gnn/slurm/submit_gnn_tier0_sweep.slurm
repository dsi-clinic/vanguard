#!/usr/bin/env bash
#SBATCH --job-name=gnn-tier0-sweep
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# Tier-0 optimization-hygiene sweep: the frozen baseline (configs/gnn.yaml
# behavior, via gnn_tier0_sweep_baseline.yaml) vs. the Tier-0 stack
# (gnn_tier0_sweep_tier0.yaml), each across 5 seeds, on the SAME freshly-built
# cache and the SAME frozen folds -- so a paired ΔAUC by (seed x fold) isolates
# optimization from representation/architecture (see gnn/PLAN_advanced_modeling.md,
# decisions D0.1-D0.6 and Tier 0 / Gate G0).
#
# 10 tasks = 2 arms x 5 seeds. Both sweep configs point at the dedicated
# cache gnn_cache_tier0_sweep (built by the companion build job, NOT the shared
# gnn_cache) and write outputs to a persistent gpfs path (this worktree is
# temporary). --random-state sets the per-fold torch seed offset AND the
# stratified-split shuffle, so each seed is an independent CV draw.
#
# Usage (submit AFTER the cache build, chained on its job id):
#   BUILD_JID=$(sbatch --parsable \
#       CACHE_DIR=/gpfs/.../gnn_cache_tier0_sweep gnn/slurm/submit_gnn_build.slurm)
#   sbatch --dependency=afterok:${BUILD_JID} gnn/slurm/submit_gnn_tier0_sweep.slurm

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

OUT_ROOT="/gpfs/data/karczmar-lab/workspaces/spencervenancio/experiments/gnn_tier0_sweep"

# Index 0-4 = baseline arm, 5-9 = tier0 arm; seeds 0..4 within each.
CONFIGS=(
  configs/gnn_tier0_sweep_baseline.yaml configs/gnn_tier0_sweep_baseline.yaml \
  configs/gnn_tier0_sweep_baseline.yaml configs/gnn_tier0_sweep_baseline.yaml \
  configs/gnn_tier0_sweep_baseline.yaml \
  configs/gnn_tier0_sweep_tier0.yaml configs/gnn_tier0_sweep_tier0.yaml \
  configs/gnn_tier0_sweep_tier0.yaml configs/gnn_tier0_sweep_tier0.yaml \
  configs/gnn_tier0_sweep_tier0.yaml
)
SEEDS=(0 1 2 3 4 0 1 2 3 4)
TAGS=(
  baseline_seed0 baseline_seed1 baseline_seed2 baseline_seed3 baseline_seed4 \
  tier0_seed0 tier0_seed1 tier0_seed2 tier0_seed3 tier0_seed4
)

i="${SLURM_ARRAY_TASK_ID}"
CONFIG="${CONFIGS[$i]}"
SEED="${SEEDS[$i]}"
TAG="${TAGS[$i]}"
OUTDIR="${OUT_ROOT}/${TAG}"

echo "task ${i}: config=${CONFIG} seed=${SEED} outdir=${OUTDIR}"

python -u -m gnn.train \
    --config "${CONFIG}" \
    --random-state "${SEED}" \
    --outdir "${OUTDIR}"
