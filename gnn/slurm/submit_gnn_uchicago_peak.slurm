#!/usr/bin/env bash
#SBATCH --job-name=gnn-uch-peak
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# Peak-denominator voxel GNN, 5 seeds. Pairs against the floored baseline arm
# (gnn_uchicago_tier_sweep_floored/baseline_seed*) on the SAME frozen predefined
# folds and the same 7-feature voxel set, so the paired dAUC isolates the
# enhancement denominator: max_t|S(t)| vs max(S0, 0.05*max_t|S(t)|).
#
#   sbatch gnn/slurm/submit_gnn_uchicago_peak.slurm            # all 5 seeds
#   sbatch --array=0 gnn/slurm/submit_gnn_uchicago_peak.slurm  # one seed
#
# Keep --time well under the next maintenance reservation or the job will sit
# pending on ReqNodeNotAvail (see `scontrol show reservation`).

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

OUT_ROOT="/gpfs/data/karczmar-lab/workspaces/spencervenancio/experiments/gnn_uchicago_peak"
SEED="${SLURM_ARRAY_TASK_ID}"

python -u -m gnn.train \
    --config configs/uchicago_gnn_tier_baseline_peak.yaml \
    --random-state "${SEED}" \
    --outdir "${OUT_ROOT}/peak_seed${SEED}"
