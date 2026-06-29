#!/usr/bin/env bash
# Run the Deep Sets smoke test directly on an interactive compute node (no sbatch).
# build (single shard, writes deepsets_manifest.csv) -> train. ~8 ISPY2 cases.
#
#   salloc ...            # grab an interactive node first
#   cd /ess/home/home1/t-9svena/vanguard
#   bash slurm/run_deepsets_smoke_interactive.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG="${CONFIG:-${REPO_ROOT}/configs/deepsets_ispy2_smoke.yaml}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/experiments/deepsets_ispy2_smoke}"
mkdir -p "${OUT_ROOT}"

echo "[smoke] repo=${REPO_ROOT}"
echo "[smoke] config=${CONFIG}"
echo "[smoke] out_root=${OUT_ROOT}"

echo "[smoke] ===== stage 1/2: build (single shard) ====="
MODE=build-single REPO_ROOT="${REPO_ROOT}" CONFIG="${CONFIG}" OUT_ROOT="${OUT_ROOT}" \
  bash "${REPO_ROOT}/slurm/deepsets_job.slurm"

echo "[smoke] ===== stage 2/2: train ====="
MODE=train REPO_ROOT="${REPO_ROOT}" CONFIG="${CONFIG}" OUTDIR="${OUT_ROOT}/train" \
  bash "${REPO_ROOT}/slurm/deepsets_job.slurm"

echo "[smoke] done. outputs under ${OUT_ROOT}"
echo "[smoke]   manifest: ${OUT_ROOT}/deepsets_manifest.csv"
echo "[smoke]   metrics : $(ls "${OUT_ROOT}"/train/**/metrics.json 2>/dev/null | head -1 || echo '<none yet>')"
