#!/usr/bin/env bash
#SBATCH --job-name=tabular-gate-uchicago
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Phase 1 gate: GNN (baseline/tier1) vs the vessel-only tabular bar, all on
# pooled OOF AUC over the 179 UChicago cases with bootstrap 95% CIs, plus a
# paired dAUC test (GNN - best tabular). Reads only small CSVs; refits the
# tiny tabular models + runs 2000 bootstraps -- fast, but kept off the head
# node so the login node stays for coordination only.
#
# Usage:
#   sbatch gnn/slurm/submit_tabular_gate_uchicago.slurm

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

python -m analysis.gnn_tabular_gate \
  --gnn-sweep-root /gpfs/data/karczmar-lab/workspaces/spencervenancio/experiments/gnn_uchicago_tier_sweep_floored \
  --graph-qc /ess/scratch/scratch1/t-9svena/uchicago/gnn_cache_voxel_richfeat_floored/processed/graph_qc.csv \
  --tabular-features experiments/uchicago_tabular_bar_floored/tabular_baseline_features.csv \
  --out-dir experiments/uchicago_tabular_bar_floored
