#!/usr/bin/env bash
#SBATCH --job-name=gnn-loco-build
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Builds the ONE cache a GNN LOCO sweep needs: every node_features (and, for
# node_mode=junction, edge_features) name in the mode's full vocabulary, not
# just the config's own gnn_node_features list. This is the "build once with
# the full feature superset" step -- see docs/loco_feature_importance.md and
# evaluation/loco_gnn.py. Topology (segments/junctions/edges) doesn't depend
# on which features are requested, and the segment/junction builders already
# compute their full vocabulary unconditionally, so this costs the same as a
# normal build -- it just means every later LOCO training run (submit_gnn_
# loco_sweep.slurm) can select any narrower feature subset from this same
# cache without a rebuild.
#
# CONFIG must be the LOCO config (the same one submit_gnn_loco_sweep.slurm
# will use) -- its model_params.gnn_node_mode picks the vocabulary, and its
# data_paths.{gnn_centerline_root,gnn_dce_root,gnn_labels_path,gnn_cache_dir}
# are reused verbatim so build and sweep can never point at different data.
#
# Usage:
#   sbatch --export=CONFIG=configs/loco_gnn_junction.yaml gnn/slurm/submit_gnn_loco_build.slurm
#   # then, once this completes:
#   sbatch --dependency=afterok:<this_job_id> \
#     --export=CONFIG=configs/loco_gnn_junction.yaml,OUTDIR=experiments/loco_gnn_junction \
#     gnn/slurm/submit_gnn_loco_sweep.slurm

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

CONFIG="${CONFIG:?CONFIG is required (the LOCO config to build the superset cache for)}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"

read -r NODE_MODE NODE_FEATURES EDGE_FEATURES LABELS_PATH CACHE_DIR ROOT DCE_ROOT < <(
  python - "${CONFIG}" <<'PY'
import sys
from load_cohort import load_config
from evaluation.loco_gnn import superset_features_for_mode, superset_edge_features_for_mode

cfg = load_config(sys.argv[1])
node_mode = str(cfg.model_params.gnn_node_mode)
node_features = ",".join(superset_features_for_mode(node_mode))
edge_features = ",".join(superset_edge_features_for_mode(node_mode)) or "-"
dp = cfg.data_paths
print(
    node_mode,
    node_features,
    edge_features,
    dp.gnn_labels_path,
    dp.gnn_cache_dir,
    dp.gnn_centerline_root,
    dp.gnn_dce_root,
)
PY
)

echo "Building GNN LOCO superset cache: mode=${NODE_MODE} cache=${CACHE_DIR}"
echo "  node_features=${NODE_FEATURES}"
echo "  edge_features=${EDGE_FEATURES}"
mkdir -p "${CACHE_DIR}"

ARGS=(
  python -u -m gnn.build_dataset
  --root "${ROOT}"
  --dce-root "${DCE_ROOT}"
  --labels-path "${LABELS_PATH}"
  --cache-dir "${CACHE_DIR}"
  --node-mode "${NODE_MODE}"
  --node-features "${NODE_FEATURES}"
  --num-workers "${NUM_WORKERS}"
)
if [[ "${EDGE_FEATURES}" != "-" ]]; then
  ARGS+=(--edge-features "${EDGE_FEATURES}")
fi
if [[ "${FORCE_REBUILD:-0}" == "1" ]]; then
  ARGS+=(--force-rebuild)
fi

"${ARGS[@]}"
