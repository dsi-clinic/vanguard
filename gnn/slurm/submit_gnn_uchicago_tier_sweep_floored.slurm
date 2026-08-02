#!/usr/bin/env bash
#SBATCH --job-name=gnn-uchicago-tier-sweep-floored
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-14
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# UChicago replication of the PLAN_advanced_modeling.md tier ladder, Tiers 0-1.
# 15 tasks = 3 arms x 5 seeds:
#   baseline (frozen hygiene) | tier0 (hygiene stack) | tier1 (tier0 + mean_max readout)
# All three arms read the SAME v5 voxel cache (gnn_cache_voxel_richfeat, 7-feature
# set) and the SAME frozen predefined folds, so a paired dAUC by (seed x fold)
# isolates each tier's change (Gate G0: tier0 vs baseline; Gate G1: tier1 vs tier0).
#
# NOTE (vs. the Duke tier0 sweep): folds are split_mode=predefined here (UChicago
# ships patient-grouped folds, no patient column to reshuffle safely), so
# --random-state varies torch init + inner-val split only, NOT fold membership.
# The paired test is therefore less powerful than Duke's random-CV draws; this is
# the deliberate leak-safe trade (LAB_NOTEBOOK 2026-07-26).
#
# Usage (cache already built -- no build dependency needed):
#   sbatch gnn/slurm/submit_gnn_uchicago_tier_sweep.slurm

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

OUT_ROOT="/ess/scratch/scratch1/t-9svena/uchicago/experiments/gnn_tier_sweep_floored"

# Index 0-4 baseline, 5-9 tier0, 10-14 tier1; seeds 0..4 within each arm.
CONFIGS=(
  configs/uchicago_gnn_tier_baseline_floored.yaml configs/uchicago_gnn_tier_baseline_floored.yaml \
  configs/uchicago_gnn_tier_baseline_floored.yaml configs/uchicago_gnn_tier_baseline_floored.yaml \
  configs/uchicago_gnn_tier_baseline_floored.yaml \
  configs/uchicago_gnn_tier0_floored.yaml configs/uchicago_gnn_tier0_floored.yaml \
  configs/uchicago_gnn_tier0_floored.yaml configs/uchicago_gnn_tier0_floored.yaml \
  configs/uchicago_gnn_tier0_floored.yaml \
  configs/uchicago_gnn_tier1_floored.yaml configs/uchicago_gnn_tier1_floored.yaml \
  configs/uchicago_gnn_tier1_floored.yaml configs/uchicago_gnn_tier1_floored.yaml \
  configs/uchicago_gnn_tier1_floored.yaml
)
SEEDS=(0 1 2 3 4 0 1 2 3 4 0 1 2 3 4)
TAGS=(
  baseline_seed0 baseline_seed1 baseline_seed2 baseline_seed3 baseline_seed4 \
  tier0_seed0 tier0_seed1 tier0_seed2 tier0_seed3 tier0_seed4 \
  tier1_seed0 tier1_seed1 tier1_seed2 tier1_seed3 tier1_seed4
)

i="${SLURM_ARRAY_TASK_ID}"
CONFIG="${CONFIGS[$i]}"
SEED="${SEEDS[$i]}"
TAG="${TAGS[$i]}"
OUTDIR="${OUT_ROOT}/${TAG}"

echo "task ${i}: config=${CONFIG} seed=${SEED} outdir=${OUTDIR}"

python -u -m gnn.train \
    --config "${CONFIG}" \
    --random-state "${SEED}" \
    --outdir "${OUTDIR}"
