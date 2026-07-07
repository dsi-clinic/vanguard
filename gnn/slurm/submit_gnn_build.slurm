#!/usr/bin/env bash
#SBATCH --job-name=gnn-build-full
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Builds the full VanguardCenterlineDataset cache (all datasets under
# ROOT, every case with a matching label in LABELS_PATH) via
# gnn/build_dataset.py. See docs/slurm-site.md for how tier1q/karczmar-lab
# were chosen -- the older `--partition=general` used by other scripts in
# this repo does not exist on this cluster.
#
# Usage:
#   sbatch gnn/slurm/submit_gnn_build.slurm
#   CACHE_DIR=/gpfs/.../my_cache sbatch gnn/slurm/submit_gnn_build.slurm
#   FORCE_REBUILD=1 sbatch gnn/slurm/submit_gnn_build.slurm  # archive + rebuild
#   ALLOW_MANIFEST_MISMATCH=1 sbatch gnn/slurm/submit_gnn_build.slurm  # bypass cache_manifest.json check
#
# All variables below default to the real cluster paths from CLAUDE.md and
# can be overridden via environment (see gnn/build_dataset.py --help for
# what each one maps to). NUM_WORKERS defaults to the job's allocated
# --cpus-per-task (8) and fans the per-case build out across that many
# worker processes, since each case is built independently from its own
# files; set NUM_WORKERS=1 to force sequential building instead.

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

ROOT="${ROOT:-/gpfs/data/karczmar-lab/workspaces/saritbose/centerlines_tc4d/studies}"
DCE_ROOT="${DCE_ROOT:-/gpfs/data/karczmar-lab/MAMA-MIA-syn60868042/images}"
LABELS_PATH="${LABELS_PATH:-/gpfs/data/karczmar-lab/workspaces/saritbose/pcr_labels.csv}"
CACHE_DIR="${CACHE_DIR:-/gpfs/data/karczmar-lab/workspaces/spencervenancio/gnn_cache}"
ID_COLUMN="${ID_COLUMN:-case_id}"
LABEL_COLUMN="${LABEL_COLUMN:-pcr}"
NODE_FEATURES="${NODE_FEATURES:-peak_time,radius}"
MAX_MISSING_LABEL_FRAC="${MAX_MISSING_LABEL_FRAC:-0.1}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"

mkdir -p "${CACHE_DIR}"

ARGS=(
  python -u gnn/build_dataset.py
  --root "${ROOT}"
  --dce-root "${DCE_ROOT}"
  --labels-path "${LABELS_PATH}"
  --cache-dir "${CACHE_DIR}"
  --id-column "${ID_COLUMN}"
  --label-column "${LABEL_COLUMN}"
  --node-features "${NODE_FEATURES}"
  --max-missing-label-frac "${MAX_MISSING_LABEL_FRAC}"
  --num-workers "${NUM_WORKERS}"
)

if [[ -n "${CASES:-}" ]]; then
  ARGS+=(--cases "${CASES}")
fi
if [[ "${NO_CACHE:-0}" == "1" ]]; then
  ARGS+=(--no-cache)
fi
if [[ "${FORCE_REBUILD:-0}" == "1" ]]; then
  ARGS+=(--force-rebuild)
fi
if [[ "${ALLOW_MANIFEST_MISMATCH:-0}" == "1" ]]; then
  ARGS+=(--allow-manifest-mismatch)
fi

"${ARGS[@]}"
