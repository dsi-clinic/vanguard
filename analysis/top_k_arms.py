"""Build ablation arms that combine a baseline block set with top-K ranked features.

Issue #151: compare tumor-size (+ clinical) baselines to baseline plus a small
set of vessel features chosen by a screening ranker.

Default assumption documented in ``configs/top_features_eval.yaml``: ranking is
computed once on all labeled rows in the cohort (global screening). This is
fast and reproducible but is not leakage-safe for inferential claims; reserve
foldwise ranking for a stricter follow-up.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from analysis.feature_ranking import (
    RankingMethod,
    rank_features,
    top_k_features,
)
from features import (
    ANNOTATION_COLUMNS,
    feature_block_for_column,
    normalize_selected_features,
)

logger = logging.getLogger(__name__)


def modeling_columns_for_blocks(
    df: pd.DataFrame,
    *,
    blocks: list[str],
    label_col: str,
) -> list[str]:
    """List modeling column names belonging to ``blocks`` (excludes annotations)."""
    normalized = normalize_selected_features(blocks)
    if not normalized:
        return []
    selected_set = set(normalized)
    keep_like = set(ANNOTATION_COLUMNS) | {label_col}
    return [
        column
        for column in df.columns
        if column not in keep_like and feature_block_for_column(column) in selected_set
    ]


def numeric_ranking_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """Keep only numeric dtypes suitable for ``rank_features``."""
    out: list[str] = []
    for col in columns:
        if col not in df.columns:
            continue
        series = df[col]
        if pd.api.types.is_bool_dtype(series):
            continue
        if pd.api.types.is_numeric_dtype(series):
            out.append(col)
    return out


def build_topk_ablation_arms(
    ranked_df: pd.DataFrame,
    *,
    baseline_model_columns: list[str],
    k_values: list[int],
    arm_name_prefix: str,
    superset_blocks: list[str],
    ranking_method_label: str,
    shared_model_params_override: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return ablation arm dicts consumable by ``run_ablation_matrix``."""
    override_base: dict[str, Any] = dict(shared_model_params_override or {})
    arms: list[dict[str, Any]] = []

    for k in k_values:
        if k <= 0:
            continue
        picked = top_k_features(ranked_df, k=k)
        explicit = list(dict.fromkeys(list(baseline_model_columns) + picked))
        safe_method = ranking_method_label.replace(" ", "_")
        arms.append(
            {
                "name": f"{arm_name_prefix}_{safe_method}_k{k}",
                "selected_features": list(superset_blocks),
                "explicit_model_columns": explicit,
                "model_params_override": {**override_base},
            }
        )
    return arms


def run_global_feature_ranking(
    df: pd.DataFrame,
    *,
    label_col: str,
    pool_columns: list[str],
    method: RankingMethod,
    ranker_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Rank ``pool_columns`` using training labels in ``df`` (global screening)."""
    use_cols = [c for c in pool_columns if c in df.columns]
    missing = sorted(set(pool_columns) - set(use_cols))
    if missing:
        logger.warning("Dropping %d pool columns not in dataframe", len(missing))
    numeric = numeric_ranking_columns(df, use_cols)
    if not numeric:
        raise ValueError("No numeric columns in ranking pool.")
    y = df[label_col]
    X = df[numeric]
    return rank_features(
        X, y, feature_names=numeric, method=method, **(ranker_kwargs or {})
    )


def assemble_top_features_experiment_arms(
    full_labeled_df: pd.DataFrame,
    *,
    label_col: str,
    baseline_blocks: list[str],
    ranking_pool_blocks: list[str],
    k_values: list[int],
    ranking_method: RankingMethod,
    baseline_arm_name: str,
    arm_name_prefix: str,
    superset_blocks: list[str] | None = None,
    ranker_kwargs: dict[str, Any] | None = None,
    baseline_model_params_override: dict[str, Any] | None = None,
    topk_model_params_override: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Create baseline + top-K arms and the global ranking table."""
    base_blocks = normalize_selected_features(baseline_blocks)
    pool_blocks = normalize_selected_features(ranking_pool_blocks)
    if not base_blocks or not pool_blocks:
        raise ValueError("baseline_blocks and ranking_pool_blocks must be non-empty.")
    if superset_blocks is None:
        superset = sorted(set(base_blocks) | set(pool_blocks))
    else:
        superset = normalize_selected_features(superset_blocks)
        if set(base_blocks) - set(superset) or set(pool_blocks) - set(superset):
            raise ValueError("superset_blocks must include baseline and pool blocks.")

    baseline_model_columns = modeling_columns_for_blocks(
        full_labeled_df, blocks=base_blocks, label_col=label_col
    )
    pool_model_columns = modeling_columns_for_blocks(
        full_labeled_df, blocks=pool_blocks, label_col=label_col
    )

    ranked = run_global_feature_ranking(
        full_labeled_df,
        label_col=label_col,
        pool_columns=pool_model_columns,
        method=ranking_method,
        ranker_kwargs=ranker_kwargs,
    )

    baseline_override = dict(baseline_model_params_override or {})
    topk_override = dict(topk_model_params_override or {})

    arms: list[dict[str, Any]] = [
        {
            "name": baseline_arm_name,
            "selected_features": list(base_blocks),
            "explicit_model_columns": None,
            "model_params_override": baseline_override,
        }
    ]
    arms.extend(
        build_topk_ablation_arms(
            ranked,
            baseline_model_columns=baseline_model_columns,
            k_values=list(k_values),
            arm_name_prefix=arm_name_prefix,
            superset_blocks=superset,
            ranking_method_label=str(ranking_method),
            shared_model_params_override=topk_override,
        )
    )
    return arms, ranked
