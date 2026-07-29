#!/usr/bin/env bash
#SBATCH --job-name=duke-forecast-smoke
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Plumbing smoke test of the voxel contrast-forecasting pilot
# (docs/design/contrast_pretraining.md) on DUKE MAMA-MIA *placeholder* data.
# NOT a scientific result -- Duke is not the target cohort; this only checks the
# build + §7.i/§7.ii gate pipeline runs end-to-end on real voxel graphs (each
# case loads a 4D DCE volume, so it must run here, not on the login node).
#
# Small by design: a handful of Duke cases, CPU-only, ~5-20s/case build. Sizing
# from LAB_NOTEBOOK.md voxel-build profile (graph_build ~3.4s, timeseries_load
# ~1.7s per case; peak RAM well under 16G for a few cases). Do NOT scale this to
# the full cohort without rethinking resources.
#
# Usage:
#   sbatch gnn/slurm/submit_duke_forecast_smoke.slurm
#   CASES="DUKE_001,DUKE_002" OUTDIR=experiments/foo sbatch gnn/slurm/submit_duke_forecast_smoke.slurm

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

CENTERLINE_ROOT="${CENTERLINE_ROOT:-/gpfs/data/karczmar-lab/workspaces/saritbose/centerlines_tc4d/studies}"
DCE_ROOT="${DCE_ROOT:-/gpfs/data/karczmar-lab/MAMA-MIA-syn60868042/images}"
LABELS_PATH="${LABELS_PATH:-/ess/scratch/scratch1/t-9svena/gnn_caches/pcr_labels_folded_matched_cohort.csv}"
CASES="${CASES:-DUKE_001,DUKE_002,DUKE_005,DUKE_009,DUKE_010,DUKE_012}"
NODE_MODE="${NODE_MODE:-voxel}"
OUTDIR="${OUTDIR:-experiments/contrast_pretrain_duke_smoke}"
INPUT_LEN="${INPUT_LEN:-3}"
TARGET_LEN="${TARGET_LEN:-2}"
# Duke has variable frame counts across cases (T=5, T=4 observed) -- the §8b
# open question. For this SMOKE only, shrink the horizon to the shortest case
# (truncate); the real variable-length policy is Spencer's to decide.
MIN_FRAMES_POLICY="${MIN_FRAMES_POLICY:-truncate}"
VAL_CASES="${VAL_CASES:-2}"
EPOCHS="${EPOCHS:-40}"
HIDDEN_DIM="${HIDDEN_DIM:-16}"
SEED="${SEED:-0}"

echo "mode=${NODE_MODE} cases=${CASES} outdir=${OUTDIR} horizon=${INPUT_LEN}/${TARGET_LEN}"

python -u -m gnn.pretrain.duke_forecast \
  --centerline-root "${CENTERLINE_ROOT}" \
  --dce-root "${DCE_ROOT}" \
  --labels-path "${LABELS_PATH}" \
  --cases "${CASES}" \
  --outdir "${OUTDIR}" \
  --node-mode "${NODE_MODE}" \
  --input-len "${INPUT_LEN}" \
  --target-len "${TARGET_LEN}" \
  --min-frames-policy "${MIN_FRAMES_POLICY}" \
  --val-cases "${VAL_CASES}" \
  --epochs "${EPOCHS}" \
  --hidden-dim "${HIDDEN_DIM}" \
  --seed "${SEED}"
