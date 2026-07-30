#!/usr/bin/env bash
#SBATCH --job-name=arrival-field-structure
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Does the arrival-time field have ANY spatial structure on the vessel tree?
#
# Separates "my root definition is wrong" from "the arrival field is spatially
# unstructured" -- the two readings of the Stage 2 relational features coming out at
# chance with arrival_monotonic_frac == 0.490. Measures edge-wise agreement of
# arrival times against a within-case shuffled null and against radius as a positive
# control (calibre must be smooth along a centreline, so if radius shows structure
# and arrival does not, the graph reconstruction is sound).
#
# On Slurm rather than the login node because it loads all 179 cached graphs
# (~283 MB) and runs Dijkstra per component. No DCE reload, no cache rebuild.
#
# See docs/design/gnn_graph_representation_plan.md Stage 2.
#
# Usage:
#   sbatch gnn/slurm/submit_arrival_field_structure.slurm

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

python -u -m analysis.arrival_field_structure \
  --cache-dir /ess/scratch/scratch1/t-9svena/uchicago/gnn_cache_voxel_richfeat_floored \
  --out-dir experiments/uchicago_arrival_field
