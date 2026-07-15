#!/usr/bin/env bash
#SBATCH --job-name=voxel-breast-split-build
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# Builds the three matched arms (mixed baseline / single breast / native
# unilateral only) of the harmonized single-breast dataset minimal
# experiment (Phase 5, LAB_NOTEBOOK.md 2026-07-14), one array task per arm.
# Fully config-driven, mirroring gnn/slurm/submit_gnn_graph_compare_build.slurm:
# node mode, matched features, labels file, cache dir, and (when set)
# breast-split settings are all read from each arm's config.
#
# Each arm's labels file is DELIBERATELY restricted to a matched sub-cohort
# (see the configs' own comments), so a large fraction of the full ~1506-case
# discovered tree is expected to be "missing" a label relative to that file
# -- exact known counts, computed in LAB_NOTEBOOK.md 2026-07-14: arm 1
# 174/1506 (11.6%), arm 2's breast_split exclusions 159/1506 (10.6%), arm 3
# 471/1506 (31.3%). MAX_MISSING_LABEL_FRAC/MAX_MISSING_BREAST_SPLIT_FRAC
# below are calibrated with real margin over those exact numbers, not
# blindly loosened -- a genuinely anomalous drop (e.g. 50%+) would still
# raise.
#
# Usage:
#   sbatch gnn/slurm/submit_voxel_breast_split_build.slurm
#   CONFIGS="configs/a.yaml configs/b.yaml configs/c.yaml" sbatch ...

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

CONFIGS="${CONFIGS:-configs/gnn_voxel_mixed_baseline.yaml configs/gnn_voxel_single_breast.yaml configs/gnn_voxel_unilateral_only.yaml}"
read -r -a CONFIG_ARR <<< "${CONFIGS}"
CONFIG="${CONFIG_ARR[${SLURM_ARRAY_TASK_ID}]}"
NUM_WORKERS="${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
MAX_MISSING_LABEL_FRAC="${MAX_MISSING_LABEL_FRAC:-0.35}"
MAX_MISSING_BREAST_SPLIT_FRAC="${MAX_MISSING_BREAST_SPLIT_FRAC:-0.15}"

echo "Building arm config: ${CONFIG}"

read -r NODE_MODE NODE_FEATURES LABELS_PATH CACHE_DIR ROOT DCE_ROOT \
  BREAST_SPLIT_MODE BREAST_SPLIT_ROOT PATIENT_INFO_DIR < <(
  python - "${CONFIG}" <<'PY'
import sys
from load_cohort import load_config

cfg = load_config(sys.argv[1])
mp, dp = cfg.model_params, cfg.data_paths
node_features = ",".join(mp.gnn_node_features)
print(
    mp.gnn_node_mode,
    node_features,
    dp.gnn_labels_path,
    dp.gnn_cache_dir,
    dp.gnn_centerline_root,
    dp.gnn_dce_root,
    mp.gnn_breast_split_mode or "-",
    dp.gnn_breast_split_skeleton_root or "-",
    dp.gnn_patient_info_dir or "-",
)
PY
)

echo "mode=${NODE_MODE} cache=${CACHE_DIR} labels=${LABELS_PATH} breast_split=${BREAST_SPLIT_MODE}"
mkdir -p "${CACHE_DIR}"

ARGS=(
  python -u -m gnn.build_dataset
  --root "${ROOT}"
  --dce-root "${DCE_ROOT}"
  --labels-path "${LABELS_PATH}"
  --cache-dir "${CACHE_DIR}"
  --node-mode "${NODE_MODE}"
  --node-features "${NODE_FEATURES}"
  --num-workers "${NUM_WORKERS}"
  --max-missing-label-frac "${MAX_MISSING_LABEL_FRAC}"
)
if [[ "${BREAST_SPLIT_MODE}" != "-" ]]; then
  ARGS+=(
    --breast-split "${BREAST_SPLIT_MODE}"
    --breast-split-skeleton-root "${BREAST_SPLIT_ROOT}"
    --patient-info-dir "${PATIENT_INFO_DIR}"
    --max-missing-breast-split-frac "${MAX_MISSING_BREAST_SPLIT_FRAC}"
  )
fi

"${ARGS[@]}"
