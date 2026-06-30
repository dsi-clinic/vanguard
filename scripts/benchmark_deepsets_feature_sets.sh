#!/usr/bin/env bash
# Issue #120: Deep Sets point-feature configs (build → merge → train).
# With Slurm: runs submit_deepsets_pipeline.sh per arm (cluster).
# Without Slurm: build, merge, write runtime YAML, train locally.
#
# For a single Slurm array that launches the original six ISPY2 pipelines, prefer:
#   sbatch slurm/submit_issue120_deepsets_feature_arms.slurm
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PARTITION="${PARTITION:-general}"
BUILD_SHARDS="${BUILD_SHARDS:-8}"
FORCE_BUILD="${FORCE_BUILD:-0}"
FORCE_MERGE="${FORCE_MERGE:-0}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"

run_arm() {
  local config_rel="$1"
  local out_root="$2"
  local config="${REPO_ROOT}/${config_rel}"
  echo "=== ${config_rel} -> ${out_root} ==="
  mkdir -p "${out_root}"
  if command -v sbatch >/dev/null 2>&1; then
    CONFIG="${config}" OUT_ROOT="${out_root}" BUILD_SHARDS="${BUILD_SHARDS}" PARTITION="${PARTITION}" \
      FORCE_BUILD="${FORCE_BUILD}" FORCE_MERGE="${FORCE_MERGE}" FORCE_RETRAIN="${FORCE_RETRAIN}" \
      "${REPO_ROOT}/slurm/submit_deepsets_pipeline.sh"
    return
  fi
  (
    cd "${REPO_ROOT}"
    export PYTHONPATH="${REPO_ROOT}"
    if [[ "${FORCE_BUILD}" != "1" ]] && [[ -f "${out_root}/deepsets_manifest.csv" ]]; then
      echo "Skipping local build for ${out_root}"
    else
      python build_deepsets_dataset.py --config "${config}" --output-dir "${out_root}"
    fi
    if compgen -G "${out_root}/manifest_parts/deepsets_manifest_part_*.csv" >/dev/null 2>&1; then
      python merge_deepsets_manifest.py --output-dir "${out_root}"
    fi
    export BENCH_CONFIG_PATH="${config}"
    export BENCH_OUT_ROOT="${out_root}"
    python - <<'PY'
import os
from pathlib import Path

import yaml

base_path = Path(os.environ["BENCH_CONFIG_PATH"])
out_root = Path(os.environ["BENCH_OUT_ROOT"])
runtime_path = out_root / "deepsets_runtime_config.yaml"
manifest_path = (out_root / "deepsets_manifest.csv").resolve()
base = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
base.setdefault("data_paths", {})
base["data_paths"]["deepsets_manifest_csv"] = str(manifest_path)
runtime_path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
print(runtime_path)
PY
    if [[ "${FORCE_RETRAIN}" != "1" ]] && find "${out_root}/train" -name metrics.json -print -quit | grep -q .; then
      echo "Skipping local train for ${out_root}"
    else
      python train_deepsets.py --config "${out_root}/deepsets_runtime_config.yaml" --outdir "${out_root}/train"
    fi
  )
}

run_arm "configs/deepsets_ispy2_pointfeat_baseline.yaml" "${REPO_ROOT}/experiments/deepsets_ispy2_pointfeat_baseline"
run_arm "configs/deepsets_ispy2_pointfeat_geom_topo.yaml" "${REPO_ROOT}/experiments/deepsets_ispy2_pointfeat_geom_topo"
run_arm "configs/deepsets_ispy2_pointfeat_geom_topo_dynamic.yaml" "${REPO_ROOT}/experiments/deepsets_ispy2_pointfeat_geom_topo_dynamic"
run_arm "configs/deepsets_ispy2_pointfeat_geom_topo_plus_curvature.yaml" "${REPO_ROOT}/experiments/deepsets_ispy2_pointfeat_geom_topo_plus_curvature"
run_arm "configs/deepsets_ispy2_pointfeat_geom_topo_no_shells.yaml" "${REPO_ROOT}/experiments/deepsets_ispy2_pointfeat_geom_topo_no_shells"
run_arm "configs/deepsets_ispy2_pointfeat_curvature_plus_dynamic.yaml" "${REPO_ROOT}/experiments/deepsets_ispy2_pointfeat_curvature_plus_dynamic"
run_arm "configs/deepsets_mamamia_pointfeat_baseline.yaml" "${REPO_ROOT}/experiments/deepsets_mamamia_pointfeat_baseline"
run_arm "configs/deepsets_mamamia_pointfeat_geom_topo_plus_curvature.yaml" "${REPO_ROOT}/experiments/deepsets_mamamia_pointfeat_geom_topo_plus_curvature"
