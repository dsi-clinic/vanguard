"""Centralized configuration defaults and loading helpers.

Students should edit YAML files under ``configs/``. This module defines the full
shape of the runtime config so defaults are not scattered across the codebase.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_ABLATION_ARMS: list[dict[str, Any]] = [
    {"name": "tumor_size_only", "selected_features": ["tumor_size"]},
    {"name": "morph_only", "selected_features": ["morph"]},
    {"name": "graph_only", "selected_features": ["graph"]},
    {"name": "kinematic_only", "selected_features": ["kinematic"]},
    {
        "name": "tumor_size_plus_morph",
        "selected_features": ["tumor_size", "morph"],
    },
    {
        "name": "tumor_size_plus_graph",
        "selected_features": ["tumor_size", "graph"],
    },
    {
        "name": "tumor_size_plus_kinematic",
        "selected_features": ["tumor_size", "kinematic"],
    },
    {
        "name": "clinical_plus_tumor_size",
        "selected_features": ["clinical", "tumor_size"],
    },
    {
        "name": "clinical_plus_tumor_size_plus_morph",
        "selected_features": ["clinical", "tumor_size", "morph"],
    },
    {
        "name": "clinical_plus_tumor_size_plus_graph",
        "selected_features": ["clinical", "tumor_size", "graph"],
    },
    {
        "name": "clinical_plus_tumor_size_plus_kinematic",
        "selected_features": ["clinical", "tumor_size", "kinematic"],
    },
    {
        "name": "clinical_plus_tumor_size_plus_graph_plus_kinematic",
        "selected_features": ["clinical", "tumor_size", "graph", "kinematic"],
    },
]

DEFAULT_CONFIG: dict[str, Any] = {
    "baseline_arm_name": None,
    "feature_toggles": {
        "use_vascular": True,
        "use_morphometry": True,
        "require_centerline_file": True,
        "include_missing_centerline_rows": False,
        "use_clinical": False,
        "include_site_features": False,
        "dataset_include": ["ISPY2"],
        "bilateral_filter": None,
        "use_tumor_local_features": True,
        "use_tumor_graph_features_json": True,
        "tumor_mask_file_pattern": "{case_id}.nii.gz",
        "tumor_mask_threshold": 0.5,
        "tumor_radius_voxels": [0, 2, 4, 8],
        "use_radiomics": False,
        "merge_how": "inner",
        "selected_features": None,
        "centerline_file_pattern": "{case_id}_skeleton_4d_exam_mask.npy",
        "toy_perfect_feature": False,
        "toy_only": False,
    },
    "experiment_setup": {
        "name": "vanguard_run",
        "base_outdir": "./experiments",
    },
    "model_params": {
        "model": "lr",
        "penalty": "elasticnet",
        "solver": "saga",
        "l1_ratio": 0.5,
        "C": 1.0,
        "n_splits": 5,
        "random_state": 42,
        "max_iter": 3000,
        "feature_select_enabled": False,
        "feature_select_mode": "global_topk",
        "feature_select_k": 128,
        "feature_select_k_kin": 0,
        "feature_select_kin_method": "topk_auc",
        "feature_select_kinematic_prefixes": ["kinematic_"],
        "feature_select_max_abs_corr": None,
        "feature_select_max_zero_rate": None,
        "feature_select_min_non_na_rate": 0.2,
        "feature_select_min_n_unique": 2,
        "feature_select_mrmr_redundancy_weight": 1.0,
        "feature_select_mrmr_include_baseline": True,
        "feature_select_corr_gate_against_baseline": True,
        "nested_tune_enabled": False,
        "nested_inner_splits": 3,
        "nested_inner_random_state": 42,
        "nested_c_grid": [1.0],
        "nested_l1_ratio_grid": [0.5],
        "nested_feature_select_mode_grid": ["block_kinematic"],
        "nested_k_kin_grid": [0, 16, 32, 64],
        "nested_kin_method_grid": ["topk_auc", "mrmr"],
        "nested_max_abs_corr_grid": [0.95],
        "nested_max_zero_rate_grid": [0.95],
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_leaf": 1,
        "xgb_learning_rate": 0.05,
        "xgb_subsample": 0.8,
        "xgb_colsample_bytree": 0.8,
        "xgb_min_child_weight": 1.0,
        "xgb_reg_alpha": 0.0,
        "xgb_reg_lambda": 1.0,
        "xgb_gamma": 0.0,
        "use_group_split": False,
        "group_col": "site",
        "stratum_col": "tumor_subtype",
        "device": "auto",
        "batch_size": 8,
        "epochs": 25,
        "hidden_dim": 32,
        "num_layers": 2,
        "dropout": 0.2,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "pooling": "mean_max_logcount",
        "early_stopping_patience": 0,
        "restore_best_epoch": False,
        "max_grad_norm": 0.0,
        "lr_scheduler": "none",
        "lr_scheduler_factor": 0.5,
        "lr_scheduler_patience": 5,
        "loss": "weighted_bce",
        "focal_alpha": 0.25,
        "focal_gamma": 2.0,
        "deepsets_local_radius_floor_mm": 50.0,
        "deepsets_local_radius_scale": 2.0,
        "deepsets_local_radius_cap_mm": 60.0,
    },
    "data_paths": {
        "centerline_root": "",
        "tumor_mask_root": "/net/projects2/vanguard/MAMA-MIA-syn60868042/segmentations/expert",
        "patient_info_dir": "",
        "clinical_excel": "",
        "labels_csv": "",
        "label_column": "pcr",
        "id_column": "case_id",
        "radiomics_csv": "",
        "deepsets_manifest_csv": "",
        "deepsets_label_column": "label",
    },
    "ablation_arms": deepcopy(DEFAULT_ABLATION_ARMS),
    # Optional multipliers for run_ablation_matrix (Issue #116 / #117 style experiments).
    "model_families": None,
    "split_mode_matrix": None,
    "model_family_overrides": {},
    "baseline_run_name": None,
    "export_subtype_summary": False,
}


class ConfigNode(dict[str, Any]):
    """Nested config mapping with both dict-style and attribute-style access."""

    def __getattr__(self, key: str) -> Any:
        """Return config values via attribute syntax."""
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        """Set config values via attribute syntax."""
        self[key] = self._wrap(value)

    def __setitem__(self, key: str, value: Any) -> None:
        """Recursively wrap nested mappings on assignment."""
        super().__setitem__(key, self._wrap(value))

    @classmethod
    def _wrap(cls: type[ConfigNode], value: Any) -> Any:
        """Convert nested dicts to config nodes and recurse through lists."""
        if isinstance(value, ConfigNode):
            return value
        if isinstance(value, dict):
            wrapped = cls()
            for nested_key, nested_value in value.items():
                wrapped[nested_key] = nested_value
            return wrapped
        if isinstance(value, list):
            return [cls._wrap(item) for item in value]
        return value

    def to_dict(self) -> dict[str, Any]:
        """Return a plain nested dict for serialization."""
        return to_plain_data(self)


def to_plain_data(value: Any) -> Any:
    """Convert config nodes and nested containers back to plain Python data."""
    if isinstance(value, ConfigNode):
        return {key: to_plain_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_plain_data(item) for item in value]
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge nested dicts, replacing non-dict values wholesale."""
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(config_path: Path) -> ConfigNode:
    """Load a YAML config, apply defaults, and return an attribute-friendly node."""
    with Path(config_path).open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must load as a mapping: {config_path}")
    return ConfigNode._wrap(_deep_merge(DEFAULT_CONFIG, loaded))
