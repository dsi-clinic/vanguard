#!/usr/bin/env bash
#SBATCH --job-name=gnn-uchicago-pretreat83
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# Re-estimate the UChicago voxel-GNN baseline arm on the CLEANED PRETREATMENT
# cohort (83 exams / 83 patients; 82 with a v5 centerline). 5 tasks = the
# baseline arm x 5 seeds.
#
# Only the baseline arm is re-run, not the full tier ladder. The tier and
# graph-mode gates are already answered on the legacy cohort (LAB_NOTEBOOK
# 2026-07-27: no arm separates from any other), so re-running them at HALF the
# sample size cannot un-answer them -- it would only produce wider intervals
# around the same tie. What the cleaner cohort actually changes is the headline
# question, "what is the honest vessel-only pCR AUC", so that is what is re-run:
# this arm plus the tabular 7-means bar it is paired against.
#
# Config differs from configs/uchicago_gnn_tier_baseline_floored.yaml ONLY in
# the labels file and cache dir, so old-vs-new is a cohort comparison and not a
# model comparison.
#
# Wall time is deliberately 4h, not the 24h the legacy sweep used: a
# maintenance reservation (2026-07-30 07:00) blocks anything that cannot finish
# before it, and 82 cases x 25 epochs runs in minutes.
#
# Usage (cache already built -- see the CACHE_DIR build in the experiment README):
#   sbatch gnn/slurm/submit_gnn_uchicago_pretreat83.slurm

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

OUT_ROOT="${OUT_ROOT:-/ess/scratch/scratch1/t-9svena/uchicago/experiments/gnn_pretreat83}"
CONFIG="${CONFIG:-configs/uchicago_gnn_tier_baseline_floored_pretreat.yaml}"

SEED="${SLURM_ARRAY_TASK_ID}"
OUTDIR="${OUT_ROOT}/baseline_seed${SEED}"

echo "task ${SLURM_ARRAY_TASK_ID}: config=${CONFIG} seed=${SEED} outdir=${OUTDIR}"

python -u -m gnn.train \
    --config "${CONFIG}" \
    --random-state "${SEED}" \
    --outdir "${OUTDIR}"
