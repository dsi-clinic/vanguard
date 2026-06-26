#!/usr/bin/env bash
#SBATCH --job-name=seg-3d-compare
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/seg-3d-compare-%j.out
#SBATCH --error=logs/seg-3d-compare-%j.err

# Render 3D comparison figures (PNG + rotating MP4) for 6 randomly selected
# cases (2 per cohort: ISPY1, ISPY2, NACT).
#
# All 6 cases run sequentially inside this single job (~5 min each).
# Override defaults by setting env vars before calling sbatch, or use
# submit_compare_seg_3d.sh which handles that for you.
#
# Env vars (all optional):
#   OUT_DIR       — output directory  (default: ~/vanguard_qc_pngs)
#   SEED          — random seed       (default: 42)
#   STRIDE        — voxel stride      (default: 3)
#   N_PER_COHORT  — cases per cohort  (default: 2)
#   N_FRAMES      — MP4 frame count   (default: 60)

set -euo pipefail

cd "${REPO_ROOT}"

set +u
source ~/.bashrc || true
if command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook -s bash)"
  micromamba activate vanguard
fi
set -u

python QC/compare_seg_3d.py \
  --out-dir      "${OUT_DIR:-${HOME}/vanguard_qc_pngs}" \
  --seed         "${SEED:-42}" \
  --stride       "${STRIDE:-3}" \
  --n-per-cohort "${N_PER_COHORT:-2}" \
  --n-frames     "${N_FRAMES:-60}"
