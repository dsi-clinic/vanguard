#!/usr/bin/env bash
#SBATCH --job-name=poster-pipeline-fig
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Pipeline figure for the symposium poster: raw ultrafast DCE-MRI -> vessel
# segmentation -> centerline -> voxel graph -> model, on one real case.
# Loads raw 4D DCE, so it runs on Slurm, not the head node.
#
# Usage:
#   sbatch gnn/slurm/submit_poster_pipeline_figure.slurm
#   CASES="id1 id2" sbatch gnn/slurm/submit_poster_pipeline_figure.slurm

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

UROOT=/ess/scratch/scratch1/t-9svena/uchicago/preprocessing_out_v5
OUT_DIR=presentations/symposium_poster_2026-08
GRAPH_WINDOW="${GRAPH_WINDOW:-55}"
CASES="${CASES:-her2_naclike_1_3_1119430059532994273898497231645274078651810083937}"

for CASE in ${CASES}; do
  echo "=================== ${CASE} ==================="
  python -m analysis.poster_pipeline_figure \
    --case-id "${CASE}" \
    --centerline-root "${UROOT}/centerlines" \
    --dce-root "${UROOT}/dce" \
    --graph-window "${GRAPH_WINDOW}" \
    --out-path "${OUT_DIR}/pipeline.png"
done
