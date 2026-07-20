#!/usr/bin/env bash
#SBATCH --job-name=breast-split-qc
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Phase 3 visual QC for the bilateral -> single-breast splitter
# (analysis/breast_split_qc.py): samples real bilateral cases with
# skeleton+breast_mask+tumor_mask available, runs gnn.breast_split.split_case
# on each, and writes a before/after overlay panel + summary CSV + HTML index
# to OUTDIR. Sequential, single-task -- 30 cases of full-resolution mask
# loading is too much I/O/RAM for the head node but not large enough to need
# an array job.
#
# Usage:
#   sbatch gnn/slurm/submit_breast_split_qc.slurm
#   N_DUKE=18 N_ISPY2=12 OUTDIR=qc/breast_split_v1 sbatch gnn/slurm/submit_breast_split_qc.slurm

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

N_DUKE="${N_DUKE:-18}"
N_ISPY1="${N_ISPY1:-0}"
N_ISPY2="${N_ISPY2:-12}"
N_NACT="${N_NACT:-0}"
SEED="${SEED:-0}"
OUTDIR="${OUTDIR:-qc/breast_split_v1}"

python -m analysis.breast_split_qc \
  --n-duke "${N_DUKE}" \
  --n-ispy1 "${N_ISPY1}" \
  --n-ispy2 "${N_ISPY2}" \
  --n-nact "${N_NACT}" \
  --seed "${SEED}" \
  --outdir "${OUTDIR}" \
  --overwrite
