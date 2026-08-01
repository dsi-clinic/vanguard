#!/usr/bin/env bash
#SBATCH --job-name=uch-temporal-cache
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Build either the unlabeled SSL cache or labeled longitudinal cache. Required
# paths are explicit because the uc-uf-pretrain preprocessing location is not a
# checked-in repository fact.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
mkdir -p "${REPO_ROOT}/logs"
cd "${REPO_ROOT}"

set +u
source "${HOME}/.bashrc" || true
eval "$(micromamba shell hook -s bash)"
micromamba activate vanguard
set -u

: "${MANIFEST:?Set MANIFEST to the UChicago exam manifest CSV}"
: "${CENTERLINE_ROOT:?Set CENTERLINE_ROOT to aligned UChicago centerlines}"
: "${DCE_ROOT:?Set DCE_ROOT to aligned UChicago DCE phases/timestamps}"
: "${CACHE_DIR:?Set CACHE_DIR to a new temporal-cache output directory}"

ARGS=(
  python -u -m gnn.build_uchicago_temporal_cache
  --manifest "${MANIFEST}"
  --centerline-root "${CENTERLINE_ROOT}"
  --dce-root "${DCE_ROOT}"
  --cache-dir "${CACHE_DIR}"
)
if [[ -n "${EXCLUDE_LABELED_MANIFEST:-}" ]]; then
  ARGS+=(--exclude-labeled-manifest "${EXCLUDE_LABELED_MANIFEST}")
fi
if [[ -n "${CASES:-}" ]]; then
  ARGS+=(--cases "${CASES}")
fi
if [[ -n "${SHARD_INDEX:-}" || -n "${NUM_SHARDS:-}" ]]; then
  : "${SHARD_INDEX:?Set both SHARD_INDEX and NUM_SHARDS}"
  : "${NUM_SHARDS:?Set both SHARD_INDEX and NUM_SHARDS}"
  ARGS+=(--shard-index "${SHARD_INDEX}" --num-shards "${NUM_SHARDS}")
fi
if [[ "${RESUME:-0}" == "1" ]]; then
  ARGS+=(--resume)
fi
if [[ "${DEFER_MANIFEST:-0}" == "1" ]]; then
  ARGS+=(--defer-manifest)
fi
"${ARGS[@]}"
