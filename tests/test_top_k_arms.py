"""Tests for Issue #151 top-K arm helpers and explicit column selection."""

from __future__ import annotations

import pandas as pd
import pytest

from analysis.feature_ranking import rank_features_by_univariate_auc
from analysis.top_k_arms import (
    assemble_top_features_experiment_arms,
    build_topk_ablation_arms,
    modeling_columns_for_blocks,
)
from tabular_cohort import select_features


def test_select_features_explicit_order() -> None:
    """Explicit columns preserve caller order and subset blocks."""
    frame = pd.DataFrame(
        {
            "case_id": ["a", "b"],
            "pcr": [0, 1],
            "age": [40.0, 50.0],
            "tumor_size_vol": [1.0, 2.0],
            "morph_x": [0.1, 0.2],
            "morph_y": [0.3, 0.4],
        }
    )
    out = select_features(
        frame,
        selected_blocks=["clinical", "tumor_size", "morph"],
        label_col="pcr",
        explicit_model_columns=["morph_y", "age", "tumor_size_vol"],
    )
    modeling = [c for c in out.columns if c not in ("case_id", "pcr")]
    assert modeling == ["morph_y", "age", "tumor_size_vol"]


def test_select_features_explicit_rejects_annotations() -> None:
    """Annotation / label columns must not appear in explicit lists."""
    frame = pd.DataFrame(
        {
            "case_id": ["a"],
            "pcr": [1],
            "age": [50.0],
        }
    )
    with pytest.raises(ValueError, match="annotation"):
        select_features(
            frame,
            selected_blocks=["clinical"],
            label_col="pcr",
            explicit_model_columns=["age", "case_id"],
        )


EXPECTED_RANKED_MORPH_COLS = 2
EXPECTED_TOTAL_ARMS = 3


def test_modeling_columns_for_blocks() -> None:
    """Non-annotation columns are grouped by feature block."""
    cohort_frame = pd.DataFrame(
        {
            "case_id": ["x"],
            "pcr": [1],
            "age": [40.0],
            "morph_a": [0.1],
        }
    )
    cols = modeling_columns_for_blocks(
        cohort_frame, blocks=["clinical", "morph"], label_col="pcr"
    )
    assert set(cols) == {"age", "morph_a"}


def test_build_topk_ablation_arms_names() -> None:
    """Arm names encode method and K; explicit lists baseline then top features."""
    tiny = pd.DataFrame(
        {
            "f1": [0.0, 1.0, 0.5, 1.0],
            "f2": [0.1, 0.0, 0.2, 0.1],
        }
    )
    rank_df = rank_features_by_univariate_auc(tiny, [0, 1, 0, 1])
    arms = build_topk_ablation_arms(
        rank_df,
        baseline_model_columns=["age"],
        k_values=[1],
        arm_name_prefix="zz",
        superset_blocks=["clinical", "morph"],
        ranking_method_label="univariate_auc",
    )
    assert len(arms) == 1
    assert arms[0]["name"] == "zz_univariate_auc_k1"
    assert arms[0]["explicit_model_columns"] == ["age", "f1"]


def test_assemble_top_features_smoke() -> None:
    """End-to-end arm assembly returns baseline plus one arm per K."""
    cohort_frame = pd.DataFrame(
        {
            "case_id": [f"c{i}" for i in range(20)],
            "pcr": [0, 1] * 10,
            "age": [float(i) for i in range(20)],
            "tumor_size_vol": [float(i) * 0.1 for i in range(20)],
            "morph_signal": [float(i % 2) for i in range(20)],
            "morph_noise": [0.1 * float(i) for i in range(20)],
        }
    )
    arms, ranked = assemble_top_features_experiment_arms(
        cohort_frame,
        label_col="pcr",
        baseline_blocks=["clinical", "tumor_size"],
        ranking_pool_blocks=["morph"],
        k_values=[1, 2],
        ranking_method="univariate_auc",
        baseline_arm_name="baseline_smoke",
        arm_name_prefix="tk",
        superset_blocks=["clinical", "tumor_size", "morph"],
    )
    assert arms[0]["name"] == "baseline_smoke"
    assert arms[0]["explicit_model_columns"] is None
    assert len(arms) == EXPECTED_TOTAL_ARMS
    assert ranked.shape[0] == EXPECTED_RANKED_MORPH_COLS
