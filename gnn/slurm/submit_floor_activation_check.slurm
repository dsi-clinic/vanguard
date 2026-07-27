#!/usr/bin/env bash
#SBATCH --job-name=floor-activation-check
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Verification: render where the kinetic baseline floor activates, on the top-down
# center axial slice, for one representative UChicago case per sub-family. Loads
# raw 4D DCE, so it runs on Slurm, not the head node.
#
# Usage:
#   sbatch gnn/slurm/submit_floor_activation_check.slurm
#   CASES="id1 id2" sbatch gnn/slurm/submit_floor_activation_check.slurm

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
CASES="${CASES:-her2_naclike_1_3_1044741369180520970681559947209255863618251733856 simbiosys_1_3_1004851971177365454526032714777574610025402498462 uch_nac_1_3_1004294011319227272751368633606051346978841464528}"

for CASE in ${CASES}; do
  echo "=================== ${CASE} ==================="
  python -m analysis.floor_activation_map \
    --case-id "${CASE}" \
    --centerline-root "${UROOT}/centerlines" \
    --dce-root "${UROOT}/dce" \
    --floor-frac 0.05 \
    --out-dir experiments/floor_activation_check
done
