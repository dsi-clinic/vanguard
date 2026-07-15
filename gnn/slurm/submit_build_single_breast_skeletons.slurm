#!/usr/bin/env bash
#SBATCH --job-name=build-single-breast-skeletons
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Phase 5a of the harmonized single-breast dataset plan: precompute single-
# breast skeletons for the FULL bilateral cohort (all datasets), not just the
# Phase 3 QC sample. Writes breast_split_manifest.csv, the authoritative
# record of which bilateral cases are in the harmonized cohort -- see
# gnn/build_single_breast_skeletons.py.
#
# Usage:
#   sbatch gnn/slurm/submit_build_single_breast_skeletons.slurm
#   OUT_ROOT=/ess/scratch/.../breast_split_skeletons sbatch ...

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

OUT_ROOT="${OUT_ROOT:-/ess/scratch/scratch1/t-9svena/breast_split_skeletons}"

python -m gnn.build_single_breast_skeletons --out-root "${OUT_ROOT}"
