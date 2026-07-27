#!/usr/bin/env bash
#SBATCH --job-name=graph-mode-gate-uchicago
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Phase 2.2 gate: voxel vs segment vs junction (floored UChicago) on pooled OOF
# AUC + bootstrap 95% CIs, plus paired dAUC of each structural mode vs. voxel.
# Reads small prediction CSVs + refits the tiny 7-means tabular bar + bootstraps;
# kept off the head node like the other gate jobs.
#
# Usage (after the graph-mode training array finishes):
#   sbatch gnn/slurm/submit_graph_mode_gate_uchicago.slurm

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

python -m analysis.gnn_graph_mode_gate \
  --voxel-root /gpfs/data/karczmar-lab/workspaces/spencervenancio/experiments/gnn_uchicago_tier_sweep_floored \
  --voxel-arm baseline \
  --graph-mode-root /gpfs/data/karczmar-lab/workspaces/spencervenancio/experiments/gnn_uchicago_graph_mode_floored \
  --graph-qc /ess/scratch/scratch1/t-9svena/uchicago/gnn_cache_voxel_richfeat_floored/processed/graph_qc.csv \
  --tabular-features experiments/uchicago_tabular_bar_floored/tabular_baseline_features.csv \
  --out-dir experiments/uchicago_graph_mode_floored
