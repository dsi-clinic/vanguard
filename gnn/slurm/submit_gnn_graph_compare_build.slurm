#!/usr/bin/env bash
#SBATCH --job-name=gnn-graph-compare-build
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# Builds the three matched caches (voxel / segment / junction) for the graph-
# representation comparison, one array task per arm. Fully config-driven: the
# node mode, matched node/edge features, folded labels file, and cache dir are
# read from each arm's config via load_config, so a v2 rerun with extended
# features only needs new configs (point CONFIGS at them) -- no edits here.
#
# The three arms MUST share the same folded labels file (split_mode: predefined
# consumes its `fold` column) and the same cohort -- generate it once first:
#   python -m gnn.make_fold_assignments --labels-path <pcr_labels.csv> \
#       --out <workspace>/pcr_labels_folded.csv
# and make sure each config's gnn_labels_path points at it.
#
# Usage:
#   sbatch gnn/slurm/submit_gnn_graph_compare_build.slurm
#   CONFIGS="configs/a.yaml configs/b.yaml configs/c.yaml" \
#       sbatch gnn/slurm/submit_gnn_graph_compare_build.slurm   # e.g. scratch overrides / v2
#
# See gnn/DESIGN_segment_graph.md and experiments/graph_repr_compare_v1/README.md.

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

# Space-separated list of the three arm configs (override for scratch/v2 runs).
CONFIGS="${CONFIGS:-configs/gnn_voxel_compare.yaml configs/gnn_segment_compare.yaml configs/gnn_junction_compare.yaml}"
read -r -a CONFIG_ARR <<< "${CONFIGS}"
CONFIG="${CONFIG_ARR[${SLURM_ARRAY_TASK_ID}]}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"

echo "Building arm config: ${CONFIG}"

# Pull the mode / matched features / paths straight from the config so build and
# train can never disagree (the cache manifest also enforces this on load).
read -r NODE_MODE NODE_FEATURES EDGE_FEATURES LABELS_PATH CACHE_DIR ROOT DCE_ROOT < <(
  python - "${CONFIG}" <<'PY'
import sys
from load_cohort import load_config

cfg = load_config(sys.argv[1])
mp, dp = cfg.model_params, cfg.data_paths
node_features = ",".join(mp.gnn_node_features)
edge_features = ",".join(mp.gnn_edge_features) if mp.gnn_edge_features else "-"
print(
    mp.gnn_node_mode,
    node_features,
    edge_features,
    dp.gnn_labels_path,
    dp.gnn_cache_dir,
    dp.gnn_centerline_root,
    dp.gnn_dce_root,
)
PY
)

echo "mode=${NODE_MODE} cache=${CACHE_DIR} labels=${LABELS_PATH}"
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

"${ARGS[@]}"
