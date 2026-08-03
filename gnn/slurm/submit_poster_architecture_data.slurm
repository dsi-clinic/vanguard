#!/usr/bin/env bash
#SBATCH --job-name=poster-arch-data
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Real-data inputs for the poster's TikZ architecture figure: one Duke vessel
# subgraph, its junction/segment derivations, its measured enhancement curves,
# and the DCE-MRI frames the graph sits on.
#
# Duke (MAMA-MIA) rather than UChicago on purpose: the poster is public, so the
# printed MRI must come from the public cohort.
#
# Loads a raw 4D DCE series (~0.6 GB), so it runs on Slurm, not the head node.
#
# Usage:
#   sbatch gnn/slurm/submit_poster_architecture_data.slurm
#   CASE=DUKE_882 sbatch gnn/slurm/submit_poster_architecture_data.slurm

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

CASE="${CASE:-DUKE_107}"
SLAB_HALF_WIDTH="${SLAB_HALF_WIDTH:-4}"
TARGET_NODES="${TARGET_NODES:-28}"

python -m analysis.poster_architecture_data \
  --case-id "${CASE}" \
  --centerline-root /gpfs/data/karczmar-lab/workspaces/saritbose/centerlines_tc4d/studies \
  --dce-root /gpfs/data/karczmar-lab/MAMA-MIA-syn60868042/images \
  --slab-half-width "${SLAB_HALF_WIDTH}" \
  --target-nodes "${TARGET_NODES}" \
  --out-dir presentations/symposium_poster/figures/architecture_data
