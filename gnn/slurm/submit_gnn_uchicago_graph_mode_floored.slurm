#!/usr/bin/env bash
#SBATCH --job-name=gnn-uch-graph-mode-floored
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# Phase 2.2: floored voxel/segment/junction graph-mode comparison on UChicago.
# 10 tasks = 2 modes x 5 seeds. Both arms read their own floored cache (segment /
# junction, built with KINETIC_BASELINE_FLOOR_FRAC=0.05) under the SAME frozen
# predefined folds and the SAME baseline (D0.6) training knobs as the voxel
# baseline arm -- so the only thing that varies across voxel/segment/junction is
# the graph representation. Voxel is already run (gnn_uchicago_tier_sweep_floored
# baseline_seed*); this adds the other two modes. Predictions land in the same
# {mode}_seed{n} layout so analysis/gnn_graph_mode_gate.py can pool them.
#
# Usage (build the two caches first; a dependency handles ordering):
#   sbatch --dependency=afterok:<seg_build>:<junc_build> \
#       gnn/slurm/submit_gnn_uchicago_graph_mode_floored.slurm

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

OUT_ROOT="/gpfs/data/karczmar-lab/workspaces/spencervenancio/experiments/gnn_uchicago_graph_mode_floored"

# Index 0-4 segment, 5-9 junction; seeds 0..4 within each mode.
CONFIGS=(
  configs/uchicago_gnn_segment_floored.yaml configs/uchicago_gnn_segment_floored.yaml \
  configs/uchicago_gnn_segment_floored.yaml configs/uchicago_gnn_segment_floored.yaml \
  configs/uchicago_gnn_segment_floored.yaml \
  configs/uchicago_gnn_junction_floored.yaml configs/uchicago_gnn_junction_floored.yaml \
  configs/uchicago_gnn_junction_floored.yaml configs/uchicago_gnn_junction_floored.yaml \
  configs/uchicago_gnn_junction_floored.yaml
)
SEEDS=(0 1 2 3 4 0 1 2 3 4)
TAGS=(
  segment_seed0 segment_seed1 segment_seed2 segment_seed3 segment_seed4 \
  junction_seed0 junction_seed1 junction_seed2 junction_seed3 junction_seed4
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
