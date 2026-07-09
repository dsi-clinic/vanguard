#!/usr/bin/env bash
#SBATCH --job-name=gnn-pcr-gaussian
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=0-15
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# pCR-as-Gaussian learnability sweep: degrades the gnn_pcr_dummy_check
# leakage canary (clean 0/1 "pcr_dummy" feature, AUC=1.0) into a graded
# class-conditional Gaussian signal and sweeps SNR (Cohen's d, holding
# noise_std=1.0 and varying the separation between class means) to trace val
# AUC from ~1.0 down to ~0.5. See gnn/README.md#pcr-as-gaussian-learnability-sweep
# and gnn.train._apply_pcr_dummy_noise.
#
# Reuses the existing gnn_cache_pcr_dummy dataset cache -- no rebuild -- via
# configs/gnn_pcr_gaussian.yaml (gnn_allow_manifest_mismatch: true, a
# deliberate, verified bypass of the cache-manifest guard for this specific
# legacy pre-manifest cache; see that config's header comment).
#
# 15 main grid points (5 Cohen's-d levels x 3 seeds, class0_mean=0.0,
# noise_std=1.0, class1_mean=d) + 1 confirmatory point (class1_mean=2.0,
# noise_std=2.0, seed=0) that holds d=1.0 but doubles both raw values, to
# verify AUC depends on the ratio (post-standardization), not absolute scale.
#
# Usage:
#   sbatch gnn/slurm/submit_gnn_pcr_gaussian_sweep.slurm

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

CLASS1_MEANS=(0.0 0.0 0.0  0.5 0.5 0.5  1.0 1.0 1.0  2.0 2.0 2.0  4.0 4.0 4.0  2.0)
NOISE_STDS=(  1.0 1.0 1.0  1.0 1.0 1.0  1.0 1.0 1.0  1.0 1.0 1.0  1.0 1.0 1.0  2.0)
SEEDS=(       0   1   2    0   1   2    0   1   2    0   1   2    0   1   2    0)
TAGS=(d0.0_seed0 d0.0_seed1 d0.0_seed2 d0.5_seed0 d0.5_seed1 d0.5_seed2 \
      d1.0_seed0 d1.0_seed1 d1.0_seed2 d2.0_seed0 d2.0_seed1 d2.0_seed2 \
      d4.0_seed0 d4.0_seed1 d4.0_seed2 d1.0_scale2x_seed0)

i="${SLURM_ARRAY_TASK_ID}"
CLASS1_MEAN="${CLASS1_MEANS[$i]}"
NOISE_STD="${NOISE_STDS[$i]}"
SEED="${SEEDS[$i]}"
TAG="${TAGS[$i]}"

OUTDIR="experiments/gnn_pcr_gaussian_grid/${TAG}"

python -u -m gnn.train --config configs/gnn_pcr_gaussian.yaml \
    --pcr-dummy-class1-mean "${CLASS1_MEAN}" \
    --pcr-dummy-noise-std "${NOISE_STD}" \
    --pcr-dummy-noise-seed "${SEED}" \
    --outdir "${OUTDIR}"
