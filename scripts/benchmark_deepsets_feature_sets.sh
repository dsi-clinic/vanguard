#!/usr/bin/env bash
# Issue #120: three Deep Sets builds with pinned configs.
# With Slurm: runs submit_deepsets_pipeline.sh per arm (cluster).
# Without Slurm: build, merge, write runtime YAML, train locally.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PARTITION="${PARTITION:-general}"
BUILD_SHARDS="${BUILD_SHARDS:-8}"

run_arm() {
  local config_rel="$1"
  local out_root="$2"
  local config="${REPO_ROOT}/${config_rel}"
  echo "=== ${config_rel} -> ${out_root} ==="
  mkdir -p "${out_root}"
  if command -v sbatch >/dev/null 2>&1; then
    CONFIG="${config}" OUT_ROOT="${out_root}" BUILD_SHARDS="${BUILD_SHARDS}" PARTITION="${PARTITION}" \
      "${REPO_ROOT}/slurm/submit_deepsets_pipeline.sh"
    return
  fi
  (
    cd "${REPO_ROOT}"
    export PYTHONPATH="${REPO_ROOT}"
    python build_deepsets_dataset.py --config "${config}" --output-dir "${out_root}"
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
    python train_deepsets.py --config "${out_root}/deepsets_runtime_config.yaml" --outdir "${out_root}/train"
  )
}

run_arm "configs/deepsets_ispy2_pointfeat_baseline.yaml" "${REPO_ROOT}/experiments/deepsets_ispy2_pointfeat_baseline"
run_arm "configs/deepsets_ispy2_pointfeat_geom_topo.yaml" "${REPO_ROOT}/experiments/deepsets_ispy2_pointfeat_geom_topo"
run_arm "configs/deepsets_ispy2_pointfeat_geom_topo_dynamic.yaml" "${REPO_ROOT}/experiments/deepsets_ispy2_pointfeat_geom_topo_dynamic"
