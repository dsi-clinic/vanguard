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
        "deepsets_support_mask_pattern": "{case_id}_skeleton_4d_exam_support_mask.npy",
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
        # Split source: "random" (stratified/group k-fold, default) or
        # "predefined" (honor a hardcoded train/val column named ``split_col``).
        "split_mode": "random",
        "split_col": "fold",
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
        "inner_val_fraction": 0.2,
        "max_grad_norm": 0.0,
        "lr_scheduler": "none",
        "lr_scheduler_factor": 0.5,
        "lr_scheduler_patience": 5,
        "use_layer_norm": False,
        "attention_hidden_dim": 32,
        "loss": "weighted_bce",
        "focal_alpha": 0.25,
        "focal_gamma": 2.0,
        "deepsets_local_radius_floor_mm": 50.0,
        "deepsets_local_radius_scale": 2.0,
        "deepsets_local_radius_cap_mm": 60.0,
        "deepsets_inclusion_rule": "local_radius_with_fallback",
        "deepsets_compare_inclusion_rules": [],
        "deepsets_point_feature_set": "baseline",
        # Node granularity for the GNN graph. "voxel" = one node per skeleton
        # voxel; "segment" = one node per vessel segment (line graph). Must
        # match the node_mode the target gnn_cache_dir was built with (the
        # cache manifest check enforces this), and gnn_node_features must use
        # the matching vocabulary (voxel names vs seg_* names). See
        # gnn/DESIGN_segment_graph.md.
        "gnn_node_mode": "voxel",
        "gnn_node_features": ["peak_time", "radius"],
        # Kinetic baseline floor the cache was built with (gnn/kinetics.py). A
        # build-time property of the cache; declared here so the train-time
        # dataset load requests the same value and the cache-manifest check
        # matches (0.0 = historical/unfloored; e.g. 0.05 for a floored cache).
        "gnn_kinetic_baseline_floor_frac": 0.0,
        # Edge-ablation control: does the model use the vessel topology at all?
        # "none" (default) trains on the real skeleton adjacency; "rewire"
        # randomizes which nodes are adjacent while preserving every node's
        # degree; "drop" removes edges entirely, reducing the GCN to a per-node
        # map plus a permutation-invariant readout. Applied at TRAIN time to the
        # cloned per-fold graphs, so all three arms share one cache -- no rebuild.
        # Not defined for junction mode (edge_attr is aligned with edge_index).
        # See gnn/train.py::_apply_edge_ablation.
        "gnn_edge_ablation": "none",
        # Seed for the "rewire" arm's degree-preserving swaps, offset per graph by
        # the graph's cache index so a case rewires identically across folds/seeds.
        "gnn_edge_ablation_seed": 0,
        # Whole-graph readout for GCNClassifier (voxel/segment): "mean"
        # (default, the original single global_mean_pool -- fully backward
        # compatible), "mean_max", or "mean_max_sum". Concatenating max/sum
        # pools keeps the distributional tails that mean-pool discards; see
        # gnn/PLAN_advanced_modeling.md decision D2.1 and gnn/model.py
        # POOLING_WIDTHS. Junction mode (EdgeGNNClassifier) ignores this and
        # always mean-pools for now.
        "gnn_pooling": "mean",
        # Edge features for node_mode="junction" (segment-as-edge, Option A):
        # the segment summary rides on the edges there. Empty for voxel/segment
        # mode, which carry no edge features. Uses the seg_* vocabulary (see
        # gnn.segment_graph / gnn.junction_graph).
        "gnn_edge_features": [],
        # Graph-level clinical/demographic covariates (gnn.clinical), opt-in
        # and mode-agnostic (voxel/segment/junction all support them the same
        # way, unlike node/edge features). Empty = no graph-level features,
        # fully backward compatible. Requires gnn_patient_info_dir or
        # gnn_clinical_excel under data_paths. See gnn/clinical.py for the
        # supported column vocabulary and why breast_density/SITE_COLUMNS are
        # excluded from casual use.
        "gnn_graph_features": [],
        # Cases with no clinical row for the requested gnn_graph_features are
        # dropped before the (expensive) graph build, the same way a missing
        # label is via max_missing_label_frac -- see gnn/data_loader.py.
        "gnn_max_missing_clinical_frac": 0.1,
        # Class-conditional Gaussian noise layered onto the "pcr_dummy"
        # leakage-canary feature at train time (gnn/train.py), never at cache
        # -build time -- see _apply_pcr_dummy_noise. Defaults (0.0, 1.0, 0.0)
        # exactly reproduce the original deterministic 0/1 broadcast.
        "gnn_pcr_dummy_class0_mean": 0.0,
        "gnn_pcr_dummy_class1_mean": 1.0,
        "gnn_pcr_dummy_noise_std": 0.0,
        "gnn_pcr_dummy_noise_seed": 0,
        # Harmonized single-breast dataset (gnn.breast_split): None (default,
        # fully backward compatible) keeps the exam-level mixed-breast
        # skeleton for every case; "single" substitutes bilateral cases with
        # their precomputed single-breast skeleton
        # (gnn_breast_split_skeleton_root under data_paths) and drops any
        # bilateral case the splitter excluded or that has no clinical row
        # for laterality. Native unilateral cases are always unchanged.
        "gnn_breast_split_mode": None,
        # Cases dropped for being bilateral-but-excluded-by-the-splitter (or
        # missing a clinical row for laterality) are logged the same way a
        # missing label is via gnn_max_missing_label_frac -- see
        # gnn/data_loader.py.
        "gnn_max_missing_breast_split_frac": 0.1,
    },
    "data_paths": {
        "centerline_root": "",
        "tumor_mask_root": "/gpfs/data/karczmar-lab/MAMA-MIA-syn60868042/segmentations/expert",
        "patient_info_dir": "",
        "clinical_excel": "",
        "labels_csv": "",
        "label_column": "pcr",
        "id_column": "case_id",
        "radiomics_csv": "",
        "deepsets_manifest_csv": "",
        "deepsets_label_column": "label",
        "vessel_segmentation_root": "",
        "gnn_centerline_root": "",
        "gnn_dce_root": "",
        "gnn_labels_path": "",
        "gnn_cache_dir": "",
        "gnn_id_column": "case_id",
        "gnn_label_column": "pcr",
        "gnn_cases": None,
        "gnn_dataset_include": None,
        "gnn_allow_manifest_mismatch": False,
        # Clinical data source for gnn_graph_features (model_params), mirroring
        # the shared patient_info_dir/clinical_excel above but kept
        # GNN-specific so a GNN run's config stays self-contained, consistent
        # with every other gnn_* data_paths key.
        "gnn_patient_info_dir": "",
        "gnn_clinical_excel": "",
        # Root of precomputed single-breast skeletons (gnn.build_single_breast_skeletons),
        # required when model_params.gnn_breast_split_mode is set.
        "gnn_breast_split_skeleton_root": "",
    },
    # Dataset-adapter selection (multi-dataset redesign, Step 1). Read only by
    # cohorts/factory.py; no pipeline stage consumes it yet. When ``name`` is
    # None the factory returns no adapter and the pipeline uses today's behavior.
    "dataset": {
        "name": None,  # "mamamia" | "uchicago"
        "cohort": None,  # mamamia only: "duke" | "ispy1" | "ispy2" | "nact"
        # dataset_root: on-disk base dir, injected into the adapter (§10 decision 5).
        "root": "",
        # split_policy knob (§10 decision 3): "auto" defers to the adapter's
        # default; "compute"/"provided" force that policy for the run.
        "split_policy": "auto",
    },
    "ablation_arms": deepcopy(DEFAULT_ABLATION_ARMS),
    # Optional multipliers for run_ablation_matrix (Issue #116 / #117 style experiments).
    "model_families": None,
    "split_mode_matrix": None,
    "model_family_overrides": {},
    "baseline_run_name": None,
    "export_subtype_summary": False,
    # Issue #151 driver block (see run_top_features_eval.py, configs/top_features_eval.yaml).
    "top_features_eval": None,
    # LOCO (Leave-One-Covariate-Out) driver block -- see evaluation/loco.py and
    # docs/loco_feature_importance.md for the expected schema.
    "loco": None,
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
