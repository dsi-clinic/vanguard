#!/usr/bin/env bash
#SBATCH --job-name=gnn-graph-compare-train
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# Trains the three matched arms (voxel / segment / junction) of the graph-
# representation comparison, one array task per arm, each reading its own
# pre-built cache. All three share the same frozen folds (split_mode: predefined
# on the folded labels file), so cross-validated AUC differences reflect the
# representation, not the split. Outputs land under
# experiments/<CAMPAIGN>/<node_mode>/ ; run analysis/gnn_graph_compare_plot.py
# afterwards to roll them up.
#
# Build the caches first (gnn/slurm/submit_gnn_graph_compare_build.slurm).
#
# Usage:
#   sbatch gnn/slurm/submit_gnn_graph_compare_train.slurm
#   CONFIGS="..." CAMPAIGN=graph_repr_compare_v2 \
#       sbatch gnn/slurm/submit_gnn_graph_compare_train.slurm   # scratch overrides / v2
#
# See gnn/DESIGN_segment_graph.md and experiments/<CAMPAIGN>/README.md.

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

CONFIGS="${CONFIGS:-configs/gnn_voxel_compare.yaml configs/gnn_segment_compare.yaml configs/gnn_junction_compare.yaml}"
CAMPAIGN="${CAMPAIGN:-graph_repr_compare_v1}"
read -r -a CONFIG_ARR <<< "${CONFIGS}"
CONFIG="${CONFIG_ARR[${SLURM_ARRAY_TASK_ID}]}"

# Tag the arm's output subdir by its node mode (read from the config).
NODE_MODE="$(python - "${CONFIG}" <<'PY'
import sys
from load_cohort import load_config
print(load_config(sys.argv[1]).model_params.gnn_node_mode)
PY
)"

OUTDIR="experiments/${CAMPAIGN}/${NODE_MODE}"
echo "Training arm config=${CONFIG} mode=${NODE_MODE} -> ${OUTDIR}"

python -u -m gnn.train --config "${CONFIG}" --outdir "${OUTDIR}"
