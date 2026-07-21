"""Tests for evaluation.loco_tabular's arm-generation and reshaping logic.

The tabular LOCO path reuses modeling.ablation.run_ablation_matrix wholesale
for actual training (already-exercised infra, not re-tested here) -- what's
new and worth testing directly is arm generation and the reshaping of
run_ablation_matrix's output (ablation_summary.csv/ablation_fold_auc.csv)
into the shared loco_summary.csv/loco_fold_deltas.csv schema. See
evaluation/loco_tabular.py and docs/loco_feature_importance.md.
"""

from __future__ import annotations

import pandas as pd
import pytest

from evaluation.loco_tabular import (
    _melt_ablation_fold_to_loco,
    _melt_ablation_summary_to_loco,
    _resolve_covariates,
    build_tabular_loco_arms,
)

_N_COVARIATES = 3  # feat_a, feat_b, feat_c
_N_ARMS = _N_COVARIATES + 1  # + 1 full baseline arm


def test_build_tabular_loco_arms_shape() -> None:
    """N+1 arms: one full baseline plus one leave-one-out arm per covariate."""
    arms = build_tabular_loco_arms(
        covariates=["feat_a", "feat_b", "feat_c"],
        selected_blocks=["tumor_size"],
        baseline_arm_name="full_baseline",
    )
    assert len(arms) == _N_ARMS
    assert arms[0]["name"] == "full_baseline"
    assert arms[0]["explicit_model_columns"] == ["feat_a", "feat_b", "feat_c"]
    by_name = {arm["name"]: arm for arm in arms}
    assert by_name["loco_drop_feat_a"]["explicit_model_columns"] == ["feat_b", "feat_c"]
    assert by_name["loco_drop_feat_b"]["explicit_model_columns"] == ["feat_a", "feat_c"]
    # Every arm keeps the same block selection; only explicit_model_columns changes.
    assert all(arm["selected_features"] == ["tumor_size"] for arm in arms)


def test_build_tabular_loco_arms_requires_covariates() -> None:
    """An empty covariate list is rejected rather than silently producing no arms."""
    with pytest.raises(ValueError, match="non-empty"):
        build_tabular_loco_arms(covariates=[], selected_blocks=["tumor_size"])


def test_resolve_covariates_requires_explicit_or_block() -> None:
    """Neither loco.covariates nor loco.covariates_from_block set -> fail loud."""
    from config import ConfigNode

    loco_cfg = ConfigNode._wrap({})
    with pytest.raises(ValueError, match="loco.covariates"):
        _resolve_covariates(loco_cfg, base_config=ConfigNode._wrap({}), outdir=None)


def test_resolve_covariates_explicit_list_rejects_duplicates() -> None:
    """A duplicated covariate name in loco.covariates is rejected."""
    from config import ConfigNode

    loco_cfg = ConfigNode._wrap({"covariates": ["a", "b", "a"]})
    with pytest.raises(ValueError, match="duplicates"):
        _resolve_covariates(loco_cfg, base_config=ConfigNode._wrap({}), outdir=None)


def test_resolve_covariates_explicit_list_passthrough() -> None:
    """An explicit loco.covariates list is returned as-is, in order."""
    from config import ConfigNode

    loco_cfg = ConfigNode._wrap({"covariates": ["a", "b", "c"]})
    covariates = _resolve_covariates(
        loco_cfg, base_config=ConfigNode._wrap({}), outdir=None
    )
    assert covariates == ["a", "b", "c"]


def test_melt_ablation_summary_to_loco() -> None:
    """ablation_summary.csv reshapes into one row per (covariate, metric)."""
    summary_df = pd.DataFrame(
        [
            {
                "run_name": "full_baseline",
                "arm_name": "full_baseline",
                "auc_mean": 0.80,
                "auc_std": 0.02,
                "ap_mean": 0.70,
                "ap_std": 0.03,
            },
            {
                "run_name": "loco_drop_feat_a",
                "arm_name": "loco_drop_feat_a",
                "auc_mean": 0.65,
                "auc_std": 0.05,
                "ap_mean": 0.60,
                "ap_std": 0.04,
            },
            {
                "run_name": "loco_drop_feat_b",
                "arm_name": "loco_drop_feat_b",
                "auc_mean": 0.79,
                "auc_std": 0.03,
                "ap_mean": 0.69,
                "ap_std": 0.03,
            },
        ]
    )
    loco_df = _melt_ablation_summary_to_loco(
        summary_df, baseline_arm_name="full_baseline"
    )

    # 2 covariates x 2 metrics (auc, ap); baseline itself is excluded.
    expected_rows = 2 * 2
    assert len(loco_df) == expected_rows
    assert set(loco_df["covariate"]) == {"feat_a", "feat_b"}
    assert set(loco_df["metric"]) == {"auc", "ap"}

    feat_a_auc = loco_df[
        (loco_df["covariate"] == "feat_a") & (loco_df["metric"] == "auc")
    ].iloc[0]
    assert feat_a_auc["baseline_mean"] == pytest.approx(0.80)
    assert feat_a_auc["loco_mean"] == pytest.approx(0.65)
    assert feat_a_auc["delta_mean"] == pytest.approx(0.65 - 0.80)

    feat_b_auc = loco_df[
        (loco_df["covariate"] == "feat_b") & (loco_df["metric"] == "auc")
    ].iloc[0]
    # Removing feat_b barely hurt performance -> small delta, feat_b less important.
    assert feat_b_auc["delta_mean"] == pytest.approx(0.79 - 0.80)
    assert abs(feat_a_auc["delta_mean"]) > abs(feat_b_auc["delta_mean"])


def test_melt_ablation_summary_to_loco_missing_baseline_raises() -> None:
    """A summary table missing the named baseline arm fails loud, not silently."""
    summary_df = pd.DataFrame(
        [{"run_name": "x", "arm_name": "x", "auc_mean": 0.5, "auc_std": 0.1}]
    )
    with pytest.raises(ValueError, match="not found"):
        _melt_ablation_summary_to_loco(summary_df, baseline_arm_name="full_baseline")


def test_melt_ablation_fold_to_loco() -> None:
    """ablation_fold_auc.csv reshapes into one row per (covariate, fold)."""
    fold_df = pd.DataFrame(
        [
            {
                "run_name": "full_baseline",
                "arm_name": "full_baseline",
                "fold": 0,
                "auc": 0.82,
            },
            {
                "run_name": "full_baseline",
                "arm_name": "full_baseline",
                "fold": 1,
                "auc": 0.78,
            },
            {
                "run_name": "loco_drop_feat_a",
                "arm_name": "loco_drop_feat_a",
                "fold": 0,
                "auc": 0.60,
            },
            {
                "run_name": "loco_drop_feat_a",
                "arm_name": "loco_drop_feat_a",
                "fold": 1,
                "auc": 0.65,
            },
        ]
    )
    loco_fold_df = _melt_ablation_fold_to_loco(
        fold_df, baseline_arm_name="full_baseline"
    )

    # 1 covariate x 2 folds; baseline excluded.
    expected_rows = 2
    assert len(loco_fold_df) == expected_rows
    fold0 = loco_fold_df[loco_fold_df["fold"] == 0].iloc[0]
    assert fold0["covariate"] == "feat_a"
    assert fold0["baseline_value"] == pytest.approx(0.82)
    assert fold0["loco_value"] == pytest.approx(0.60)
    assert fold0["delta"] == pytest.approx(0.60 - 0.82)
