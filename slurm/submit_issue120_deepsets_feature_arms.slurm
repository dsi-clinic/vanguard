#!/bin/bash
# Issue #120: submit the full Deep Sets pipeline (build → merge → train) for each
# point-feature config. One array task per config; each task only runs
# submit_deepsets_pipeline.sh (which enqueues build array, merge, train jobs).
#
# Usage (from repository root, so SLURM_SUBMIT_DIR resolves correctly):
#   sbatch slurm/submit_issue120_deepsets_feature_arms.slurm
#
# Optional environment overrides (exported into submit_deepsets_pipeline.sh):
#   PARTITION=general BUILD_SHARDS=8
#SBATCH --job-name=issue120-deepsets-arms
#SBATCH --partition=general
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/issue120-deepsets-arms-%A-%a.out
#SBATCH --error=logs/issue120-deepsets-arms-%A-%a.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p "${REPO_ROOT}/logs"
cd "${REPO_ROOT}"

source "${HOME}/.bashrc" || true
if command -v micromamba >/dev/null 2>&1; then
  set +u
  eval "$(micromamba shell hook -s bash)"
  micromamba activate vanguard
  set -u
fi

PARTITION="${PARTITION:-general}"
BUILD_SHARDS="${BUILD_SHARDS:-8}"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    CONFIG="configs/deepsets_ispy2_pointfeat_baseline.yaml"
    OUT_ROOT="experiments/deepsets_ispy2_pointfeat_baseline"
    ;;
  1)
    CONFIG="configs/deepsets_ispy2_pointfeat_geom_topo.yaml"
    OUT_ROOT="experiments/deepsets_ispy2_pointfeat_geom_topo"
    ;;
  2)
    CONFIG="configs/deepsets_ispy2_pointfeat_geom_topo_dynamic.yaml"
    OUT_ROOT="experiments/deepsets_ispy2_pointfeat_geom_topo_dynamic"
    ;;
  3)
    CONFIG="configs/deepsets_ispy2_pointfeat_geom_topo_plus_curvature.yaml"
    OUT_ROOT="experiments/deepsets_ispy2_pointfeat_geom_topo_plus_curvature"
    ;;
  4)
    CONFIG="configs/deepsets_ispy2_pointfeat_geom_topo_no_shells.yaml"
    OUT_ROOT="experiments/deepsets_ispy2_pointfeat_geom_topo_no_shells"
    ;;
  5)
    CONFIG="configs/deepsets_ispy2_pointfeat_curvature_plus_dynamic.yaml"
    OUT_ROOT="experiments/deepsets_ispy2_pointfeat_curvature_plus_dynamic"
    ;;
  *)
    echo "Unexpected SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

export CONFIG OUT_ROOT PARTITION BUILD_SHARDS
bash "${REPO_ROOT}/slurm/submit_deepsets_pipeline.sh"
