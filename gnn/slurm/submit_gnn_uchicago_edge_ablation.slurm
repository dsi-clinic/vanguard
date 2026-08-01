#!/usr/bin/env bash
#SBATCH --job-name=gnn-uchicago-edge-ablation
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-14
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# Does the voxel GNN use the vessel topology at all?
#
# 15 tasks = 3 arms x 5 seeds:
#   none   -- the real skeleton adjacency (reference)
#   rewire -- degree-preserving random rewiring; same node features, node count,
#             edge count and degree sequence, only *which* nodes are adjacent changes
#   drop   -- edge_index emptied; GCNConv's self-loops reduce the model to a
#             per-node map plus a permutation-invariant readout
#
# All three arms read the SAME floored 7-feature voxel cache
# (gnn_cache_voxel_richfeat_floored) and the SAME frozen predefined folds, and the
# ablation is applied at TRAIN time to the cloned per-fold graphs
# (gnn/train.py::_apply_edge_ablation), so no cache rebuild is needed and a paired
# dAUC by (seed x fold) isolates the ablation and nothing else.
#
# Motivation: at num_layers=2 on skeleton graphs whose diameter is 290-618 hops,
# each node sees only its immediate neighbourhood before global_mean_pool averages
# everything -- so the graph may be contributing nothing but a short moving average.
# If all three arms score the same, topology is provably unused, and every
# graph-mode null to date is explained by the operator rather than by the
# vasculature. See docs/design/gnn_graph_representation_plan.md Stage 0.1.
#
# As with the tier sweep, folds are split_mode=predefined, so --random-state varies
# torch init + inner-val split only, NOT fold membership.
#
# Usage (cache already built -- no build dependency needed):
#   sbatch gnn/slurm/submit_gnn_uchicago_edge_ablation.slurm

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

OUT_ROOT="/ess/scratch/scratch1/t-9svena/uchicago/experiments/gnn_edge_ablation"

# Index 0-4 none, 5-9 rewire, 10-14 drop; seeds 0..4 within each arm.
CONFIGS=(
  configs/uchicago_gnn_edge_none.yaml configs/uchicago_gnn_edge_none.yaml \
  configs/uchicago_gnn_edge_none.yaml configs/uchicago_gnn_edge_none.yaml \
  configs/uchicago_gnn_edge_none.yaml \
  configs/uchicago_gnn_edge_rewire.yaml configs/uchicago_gnn_edge_rewire.yaml \
  configs/uchicago_gnn_edge_rewire.yaml configs/uchicago_gnn_edge_rewire.yaml \
  configs/uchicago_gnn_edge_rewire.yaml \
  configs/uchicago_gnn_edge_drop.yaml configs/uchicago_gnn_edge_drop.yaml \
  configs/uchicago_gnn_edge_drop.yaml configs/uchicago_gnn_edge_drop.yaml \
  configs/uchicago_gnn_edge_drop.yaml
)
SEEDS=(0 1 2 3 4 0 1 2 3 4 0 1 2 3 4)
TAGS=(
  none_seed0 none_seed1 none_seed2 none_seed3 none_seed4 \
  rewire_seed0 rewire_seed1 rewire_seed2 rewire_seed3 rewire_seed4 \
  drop_seed0 drop_seed1 drop_seed2 drop_seed3 drop_seed4
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
