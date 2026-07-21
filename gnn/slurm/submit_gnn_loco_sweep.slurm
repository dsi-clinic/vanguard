#!/usr/bin/env bash
#SBATCH --job-name=gnn-loco-sweep
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Runs the GNN LOCO sweep (evaluation.loco_gnn.run_gnn_loco, dispatched via
# evaluation/loco.py's CLI): one full-baseline training run + one leave-one-
# covariate-out run per feature in the mode's vocabulary (or loco.covariates,
# if the config restricts it), all reading the SAME superset cache built by
# submit_gnn_loco_build.slurm -- no rebuild happens between runs (see
# docs/loco_feature_importance.md). Runs serially within this one job rather
# than as a Slurm array: GNN's per-mode feature vocabularies are small
# (single digits to ~20), so an N+1-run sweep is cheap once the cache exists;
# revisit as an array (mirroring modeling/run_arm_fold.py's arm-index
# pattern) only if a future covariate list or per-run training time makes
# that no longer true.
#
# Build the superset cache first (gnn/slurm/submit_gnn_loco_build.slurm) --
# this job will fail loudly (RuntimeError from gnn/data_loader.py) if the
# cache at CONFIG's data_paths.gnn_cache_dir doesn't already have the full
# feature vocabulary for CONFIG's model_params.gnn_node_mode.
#
# Usage:
#   sbatch --dependency=afterok:<build_job_id> \
#     --export=CONFIG=configs/loco_gnn_junction.yaml,OUTDIR=experiments/loco_gnn_junction \
#     gnn/slurm/submit_gnn_loco_sweep.slurm

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

CONFIG="${CONFIG:?CONFIG is required (LOCO config with a top-level loco: block, family: gnn)}"
OUTDIR="${OUTDIR:?OUTDIR is required}"

python -u -m evaluation.loco --config "${CONFIG}" --outdir "${OUTDIR}"
