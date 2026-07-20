#!/usr/bin/env bash
#SBATCH --job-name=breast-split-diagnose
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# One-off: recompute raw tumor-in-breast-mask overlap fraction for every case
# in the existing qc/breast_split_v1/breast_split_qc_summary.csv sample,
# regardless of gate outcome -- diagnosing why analysis/breast_split_qc.py's
# Phase 3 run excluded 23/30 cases for tumor_mostly_outside_breast_mask.
# See analysis/breast_split_diagnose.py.

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

python -m analysis.breast_split_diagnose
