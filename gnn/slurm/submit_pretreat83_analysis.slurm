#!/usr/bin/env bash
#SBATCH --job-name=pretreat83-analysis
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Analysis leg for the cleaned pretreatment cohort: the GNN-vs-tabular gate,
# then the HER2/subtype breakdown of the very predictions that gate produced.
#
# The two run in ONE job, in this order, on purpose. gnn_subtype_breakdown.py
# consumes oof_predictions.csv written by gnn_tabular_gate.py, so the subgroup
# AUCs are stratifications of the same fits as the headline AUC rather than a
# second, independently-fit set of numbers that could quietly disagree with it.
#
# Depends on the training array and the tabular bar having finished (chain with
# --dependency=afterok:<train_jobid>:<tabular_jobid>).
#
# Usage:
#   sbatch --dependency=afterok:JOBA:JOBB gnn/slurm/submit_pretreat83_analysis.slurm

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

OUT_DIR="${OUT_DIR:-experiments/uchicago_pretreat83}"
SWEEP_ROOT="${SWEEP_ROOT:-/ess/scratch/scratch1/t-9svena/uchicago/experiments/gnn_pretreat83}"
CACHE_DIR="${CACHE_DIR:-/ess/scratch/scratch1/t-9svena/uchicago/gnn_cache_voxel_richfeat_floored_pretreat83}"
MANIFEST="${MANIFEST:-/gpfs/data/karczmar-lab/vanguard/dce2d_internal_ultrafast_pretreatment_cohort_v1/dce2d_internal_ultrafast_manifest.csv}"

# Only the baseline arm exists on this cohort -- the tier ladder was not re-run
# (see gnn/slurm/submit_gnn_uchicago_pretreat83.slurm for why).
python -m analysis.gnn_tabular_gate \
  --gnn-sweep-root "${SWEEP_ROOT}" \
  --graph-qc "${CACHE_DIR}/processed/graph_qc.csv" \
  --tabular-features "${OUT_DIR}/tabular_baseline_features.csv" \
  --out-dir "${OUT_DIR}" \
  --arms baseline

python -m analysis.gnn_subtype_breakdown \
  --oof-predictions "${OUT_DIR}/oof_predictions.csv" \
  --manifest "${MANIFEST}" \
  --out-dir "${OUT_DIR}"
