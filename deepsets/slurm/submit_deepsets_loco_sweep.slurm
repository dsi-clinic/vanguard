#!/usr/bin/env bash
#SBATCH --job-name=deepsets-loco-sweep
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Runs the DeepSets LOCO sweep (evaluation.loco_deepsets.run_deepsets_loco,
# dispatched via evaluation/loco.py's CLI): one full-baseline training run +
# one leave-one-covariate-out run per column in the loco_superset regime (28
# by default, or loco.covariates if the config restricts it), all reading the
# SAME manifest/case tensors -- no rebuild happens between runs (see
# docs/loco_feature_importance.md). Runs serially within this one job rather
# than as a Slurm array, matching gnn/slurm/submit_gnn_loco_sweep.slurm's
# reasoning: 28 covariates is cheap once the manifest exists.
#
# Unlike GNN, this needs NO new build script: the existing
# slurm/submit_deepsets_build_merge.sh already builds whatever regime CONFIG
# names, entirely config-driven. Build the manifest first with a LOCO config
# whose model_params.deepsets_point_feature_set is "loco_superset" --
# CONFIG here must be that SAME config (or another with the same
# data_paths.deepsets_manifest_csv), or this job fails loudly (ValueError
# from evaluation/loco_deepsets.py::ensure_deepsets_superset_manifest-style
# checks inside SavedSetLookup.get) rather than silently training on the
# wrong columns.
#
# Usage:
#   CONFIG=configs/loco_deepsets.yaml OUT_ROOT=experiments/loco_deepsets \
#     ./slurm/submit_deepsets_build_merge.sh
#   # then, once the manifest merge job completes:
#   sbatch --dependency=afterok:<merge_job_id> \
#     --export=CONFIG=configs/loco_deepsets.yaml,OUTDIR=experiments/loco_deepsets/loco \
#     deepsets/slurm/submit_deepsets_loco_sweep.slurm

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

CONFIG="${CONFIG:?CONFIG is required (LOCO config with a top-level loco: block, family: deepsets, and data_paths.deepsets_manifest_csv pointing at an already-built loco_superset manifest)}"
OUTDIR="${OUTDIR:?OUTDIR is required}"

python -u -m evaluation.loco --config "${CONFIG}" --outdir "${OUTDIR}"
