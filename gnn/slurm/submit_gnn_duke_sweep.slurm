#!/usr/bin/env bash
#SBATCH --job-name=gnn-duke-sweep
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# DUKE-only readout re-run: mean pool vs mean+max, 5 seeds each, restricted to
# the DUKE cohort (the in-house data most resembles DUKE, so cross-cohort batch
# effect is out of scope -- see gnn/CONCEPTS_STUDY_GUIDE.md #9). Both arms carry
# all Tier-0 training knobs and filter the SAME full cache
# (gnn_cache_tier0_sweep) to DUKE via gnn_dataset_include -- no rebuild. Folds
# are stratified on the pCR label (stratum_col: null) since a DUKE-only cohort
# makes dataset-stratification trivial; ~13 positives/fold over 280 DUKE cases.
#
# Answers: does the mean+max readout win hold when the model is TRAINED on DUKE
# only (not just measured on the DUKE slice of a full-cohort model)? Pair
# duke_meanmax_seedS vs duke_mean_seedS by seed x fold.
#
# 10 tasks = 2 arms x 5 seeds. Index 0-4 = mean, 5-9 = mean_max; seeds 0..4.
#
# Usage:
#   sbatch gnn/slurm/submit_gnn_duke_sweep.slurm

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

OUT_ROOT="/gpfs/data/karczmar-lab/workspaces/spencervenancio/experiments/gnn_duke_sweep"

CONFIGS=(
  configs/gnn_duke_mean.yaml configs/gnn_duke_mean.yaml configs/gnn_duke_mean.yaml \
  configs/gnn_duke_mean.yaml configs/gnn_duke_mean.yaml \
  configs/gnn_duke_meanmax.yaml configs/gnn_duke_meanmax.yaml configs/gnn_duke_meanmax.yaml \
  configs/gnn_duke_meanmax.yaml configs/gnn_duke_meanmax.yaml
)
SEEDS=(0 1 2 3 4 0 1 2 3 4)
TAGS=(
  duke_mean_seed0 duke_mean_seed1 duke_mean_seed2 duke_mean_seed3 duke_mean_seed4 \
  duke_meanmax_seed0 duke_meanmax_seed1 duke_meanmax_seed2 duke_meanmax_seed3 duke_meanmax_seed4
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
