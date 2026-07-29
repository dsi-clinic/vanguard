#!/usr/bin/env bash
#SBATCH --job-name=floor-activation-check
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Verification: render where the kinetic baseline floor activates, per case, for
# the whole UChicago cohort. Loads raw 4D DCE, so it runs on Slurm, not the head
# node. Writes one directory of figures per case under OUT_DIR.
#
# Usage:
#   sbatch gnn/slurm/submit_floor_activation_check.slurm
#   CASES="id1 id2" sbatch gnn/slurm/submit_floor_activation_check.slurm
#   OUT_DIR=/path/to/dir sbatch gnn/slurm/submit_floor_activation_check.slurm

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
OUT_DIR="${OUT_DIR:-/gpfs/data/karczmar-lab/workspaces/spencervenancio/precontrast_floor_experiment}"

# Omitting --case-id renders every case under the centerline root.
CASE_ARGS=()
if [[ -n "${CASES:-}" ]]; then
  CASE_ARGS=(--case-id ${CASES})
fi

python -m analysis.floor_activation_map \
  "${CASE_ARGS[@]}" \
  --centerline-root "${UROOT}/centerlines" \
  --dce-root "${UROOT}/dce" \
  --floor-frac 0.05 \
  --out-dir "${OUT_DIR}"
