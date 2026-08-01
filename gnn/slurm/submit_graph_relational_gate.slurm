#!/usr/bin/env bash
#SBATCH --job-name=graph-relational-gate
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Stage 2 gate: does the vessel GRAPH carry pCR signal the per-node means discard?
#
# Computes per-case RELATIONAL features (gnn/relational_features.py -- transit
# slope, arrival monotonicity, sibling asymmetry, radius tapering) from the already
# built floored voxel cache, then scores a plain logistic regression on them
# against the published 7-means bar, the protocol-residualized bar, and a
# protocol-only floor, with bootstrap CIs and paired deltas.
#
# On Slurm rather than the login node because it loads all 179 cached graphs
# (~283 MB) and runs Dijkstra per component; the analysis itself is single-core and
# takes a couple of minutes. No DCE reload and no cache rebuild.
#
# See docs/design/gnn_graph_representation_plan.md Stage 2.
#
# Usage:
#   sbatch gnn/slurm/submit_graph_relational_gate.slurm

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

python -u -m analysis.graph_relational_gate \
  --cache-dir /ess/scratch/scratch1/t-9svena/uchicago/gnn_cache_voxel_richfeat_floored \
  --tabular-features experiments/uchicago_tabular_bar_floored/tabular_baseline_features.csv \
  --dce-root /ess/scratch/scratch1/t-9svena/uchicago/preprocessing_out_v5/dce \
  --out-dir experiments/uchicago_graph_relational
