"""Resolve Slurm resource defaults for a Deep Sets pipeline config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

DYNAMIC_FEATURE_SETS = {
    "geometry_topology_dynamic",
    "curvature_plus_dynamic",
}


def _load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def resolve_profile(config_path: Path) -> dict[str, str | int]:
    """Return build/train Slurm defaults derived from the YAML config."""
    config = _load_config(config_path)
    model_params = config.get("model_params", {}) or {}
    feature_toggles = config.get("feature_toggles", {}) or {}
    point_feature_set = str(model_params.get("deepsets_point_feature_set", "baseline"))
    dataset_include = feature_toggles.get("dataset_include")
    all_cohorts = dataset_include in (None, "null", "none", "")

    build_time = "04:00:00"
    build_shards = 8
    if point_feature_set in DYNAMIC_FEATURE_SETS:
        build_time = "08:00:00"
        build_shards = 12
    if all_cohorts:
        build_shards = max(build_shards, 16)
        build_time = "08:00:00"

    train_time = "08:00:00"
    train_gpus = 0
    device = str(model_params.get("device", "cpu")).lower()
    if device == "cuda":
        train_gpus = 1

    return {
        "point_feature_set": point_feature_set,
        "all_cohorts": int(all_cohorts),
        "build_time": build_time,
        "build_shards": build_shards,
        "train_time": train_time,
        "train_gpus": train_gpus,
    }


def main() -> None:
    """Print JSON Slurm defaults for one Deep Sets config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(resolve_profile(args.config), sort_keys=True))


if __name__ == "__main__":
    main()
