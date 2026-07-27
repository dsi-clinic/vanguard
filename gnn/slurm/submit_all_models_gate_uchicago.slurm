#!/usr/bin/env bash
#SBATCH --job-name=all-models-gate-uchicago
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# All-models summary forest plot (tabular + every GNN arm) on pooled OOF AUC +
# bootstrap 95% CI. Run after the 2-feat training array finishes.
#   sbatch --dependency=afterok:<2feat_array> gnn/slurm/submit_all_models_gate_uchicago.slurm

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

GPFS=/gpfs/data/karczmar-lab/workspaces/spencervenancio/experiments
python -m analysis.gnn_all_models_gate \
  --tier-sweep-root "${GPFS}/gnn_uchicago_tier_sweep_floored" \
  --graph-mode-root "${GPFS}/gnn_uchicago_graph_mode_floored" \
  --twofeat-root "${GPFS}/gnn_uchicago_2feat_floored" \
  --graph-qc /ess/scratch/scratch1/t-9svena/uchicago/gnn_cache_voxel_richfeat_floored/processed/graph_qc.csv \
  --tabular-features experiments/uchicago_tabular_bar_floored/tabular_baseline_features.csv \
  --out-dir experiments/uchicago_all_models
