"""Tabular adapter for LOCO (Leave-One-Covariate-Out) feature importance.

Unlike GNN/DeepSets, tabular already has a free (no-rebuild) column-exclusion
mechanism -- ``tabular.cohort.select_features``'s ``explicit_model_columns``
-- and an existing arm-based batch runner
(``modeling.ablation.run_ablation_matrix``) that does almost exactly what
LOCO needs: build the full dataset once, then retrain per named arm. So this
module's execution path intentionally diverges from
``evaluation.loco_gnn``/``evaluation.loco_deepsets``: instead of implementing
the ``LOCOAdapter`` per-covariate callback Protocol, it generates N+1 named
arms (one "full" baseline + one per left-out covariate) up front and
delegates the whole sweep to ``run_ablation_matrix`` in one call, then
reshapes its output (``ablation_summary.csv``/``ablation_fold_auc.csv``) into
the shared ``loco_summary.csv``/``loco_fold_deltas.csv`` schema so all three
families still land in one common table format. See
``docs/loco_feature_importance.md``.

Note: ``ablation_fold_auc.csv`` (and therefore ``loco_fold_deltas.csv`` here)
only carries the ``auc`` metric -- a limitation of the reused
``modeling.ablation`` infra, not something this module adds.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

from config import ConfigNode
from evaluation.loco import write_loco_readme
from features import ANNOTATION_COLUMNS, normalize_selected_features

_ARM_PREFIX = "loco_drop_"
_DEFAULT_BASELINE_ARM_NAME = "full_baseline"
_DEFAULT_MAX_BLOCK_COVARIATES = 50


def build_tabular_loco_arms(
    *,
    covariates: list[str],
    selected_blocks: list[str],
    baseline_arm_name: str = _DEFAULT_BASELINE_ARM_NAME,
) -> list[dict[str, Any]]:
    """N+1 ablation-arm dicts: one full baseline plus one per left-out covariate.

    Both the baseline and every LOCO arm draw ``explicit_model_columns`` from
    the identical, user-declared ``covariates`` pool (never "every column in
    the block") -- this is what keeps a tabular LOCO run tractable even
    though some blocks (e.g. kinematic) have hundreds of columns.
    """
    if not covariates:
        raise ValueError("covariates must be non-empty")
    arms: list[dict[str, Any]] = [
        {
            "name": baseline_arm_name,
            "selected_features": list(selected_blocks),
            "explicit_model_columns": list(covariates),
        }
    ]
    for covariate in covariates:
        arms.append(
            {
                "name": f"{_ARM_PREFIX}{covariate}",
                "selected_features": list(selected_blocks),
                "explicit_model_columns": [c for c in covariates if c != covariate],
            }
        )
    return arms


def _resolve_covariates(
    loco_cfg: ConfigNode, *, base_config: ConfigNode, outdir: Path
) -> list[str]:
    """Explicit covariate list, or an explicitly-requested single-block expansion.

    Never silently defaults to "every column in every selected block" -- the
    kinematic block alone can be ~800 columns, so unrestricted expansion
    could mean 800+ retrains per run by default.
    """
    explicit = loco_cfg.get("covariates")
    if explicit:
        covariates = [str(c) for c in explicit]
        if len(set(covariates)) != len(covariates):
            raise ValueError(f"loco.covariates has duplicates: {covariates}")
        return covariates

    block = loco_cfg.get("covariates_from_block")
    if not block:
        raise ValueError(
            "tabular LOCO requires either loco.covariates (an explicit column "
            "list) or loco.covariates_from_block (a single block name, "
            "capped by loco.max_covariates) -- never silently every column "
            "in every selected block. See docs/loco_feature_importance.md."
        )

    from modeling.ablation import _prepare_full_dataset
    from tabular.cohort import select_features

    full_df = _prepare_full_dataset(
        base_config, [{"selected_features": [str(block)]}], outdir
    )
    label_col = base_config.data_paths.label_column
    block_df = select_features(
        full_df, selected_blocks=[str(block)], label_col=label_col
    )
    covariates = [
        c for c in block_df.columns if c not in ANNOTATION_COLUMNS and c != label_col
    ]
    max_covariates = int(
        loco_cfg.get("max_covariates") or _DEFAULT_MAX_BLOCK_COVARIATES
    )
    if len(covariates) > max_covariates:
        logging.warning(
            "loco.covariates_from_block=%r expanded to %d columns; capping to "
            "the first %d (set loco.max_covariates to override).",
            block,
            len(covariates),
            max_covariates,
        )
        covariates = covariates[:max_covariates]
    return covariates


def _melt_ablation_summary_to_loco(
    summary_df: pd.DataFrame, *, baseline_arm_name: str
) -> pd.DataFrame:
    """Reshape ``ablation_summary.csv`` into the shared (covariate, metric) LOCO schema.

    Recovers the dropped covariate from each arm's ``loco_drop_<covariate>`` name.
    """
    columns = [
        "family",
        "node_mode",
        "feature_group",
        "covariate",
        "metric",
        "baseline_mean",
        "baseline_std",
        "loco_mean",
        "loco_std",
        "delta_mean",
        "n_splits",
    ]
    if summary_df.empty:
        return pd.DataFrame(columns=columns)

    metric_names = sorted(
        col[: -len("_mean")]
        for col in summary_df.columns
        if col.endswith("_mean") and f"{col[: -len('_mean')]}_std" in summary_df.columns
    )
    baseline_rows = summary_df.loc[summary_df["arm_name"] == baseline_arm_name]
    if baseline_rows.empty:
        raise ValueError(
            f"Baseline arm {baseline_arm_name!r} not found in ablation_summary"
        )
    baseline_row = baseline_rows.iloc[0]

    rows: list[dict[str, Any]] = []
    for _, row in summary_df.iterrows():
        arm_name = str(row["arm_name"])
        if arm_name == baseline_arm_name or not arm_name.startswith(_ARM_PREFIX):
            continue
        covariate = arm_name[len(_ARM_PREFIX) :]
        for metric in metric_names:
            base_mean = baseline_row.get(f"{metric}_mean")
            loco_mean = row.get(f"{metric}_mean")
            delta_mean = (
                loco_mean - base_mean
                if pd.notna(base_mean) and pd.notna(loco_mean)
                else None
            )
            rows.append(
                {
                    "family": "tabular",
                    "node_mode": None,
                    "feature_group": "column",
                    "covariate": covariate,
                    "metric": metric,
                    "baseline_mean": base_mean,
                    "baseline_std": baseline_row.get(f"{metric}_std"),
                    "loco_mean": loco_mean,
                    "loco_std": row.get(f"{metric}_std"),
                    "delta_mean": delta_mean,
                    "n_splits": row.get("n_splits"),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _melt_ablation_fold_to_loco(
    fold_df: pd.DataFrame, *, baseline_arm_name: str
) -> pd.DataFrame:
    """Reshape ``ablation_fold_auc.csv`` into the shared per-fold LOCO schema (auc only)."""
    columns = [
        "family",
        "node_mode",
        "feature_group",
        "covariate",
        "fold",
        "metric",
        "baseline_value",
        "loco_value",
        "delta",
    ]
    if fold_df.empty:
        return pd.DataFrame(columns=columns)

    baseline_fold = fold_df.loc[
        fold_df["arm_name"] == baseline_arm_name, ["fold", "auc"]
    ].rename(columns={"auc": "baseline_value"})

    rows: list[dict[str, Any]] = []
    for _, row in fold_df.iterrows():
        arm_name = str(row["arm_name"])
        if arm_name == baseline_arm_name or not arm_name.startswith(_ARM_PREFIX):
            continue
        covariate = arm_name[len(_ARM_PREFIX) :]
        base_match = baseline_fold.loc[baseline_fold["fold"] == row["fold"]]
        base_value = (
            float(base_match["baseline_value"].iloc[0])
            if not base_match.empty
            else None
        )
        loco_value = float(row["auc"]) if pd.notna(row["auc"]) else None
        rows.append(
            {
                "family": "tabular",
                "node_mode": None,
                "feature_group": "column",
                "covariate": covariate,
                "fold": int(row["fold"]),
                "metric": "auc",
                "baseline_value": base_value,
                "loco_value": loco_value,
                "delta": (
                    loco_value - base_value
                    if base_value is not None and loco_value is not None
                    else None
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def run_tabular_loco(config: ConfigNode, outdir: Path) -> None:
    """Config-driven LOCO sweep driver for the tabular track.

    Generates N+1 ablation arms (see ``build_tabular_loco_arms``) and
    delegates the whole sweep to ``modeling.ablation.run_ablation_matrix``
    (single full-dataset build, per-arm training, baseline deltas already
    computed there), then reshapes its output into ``loco_summary.csv`` /
    ``loco_fold_deltas.csv`` / ``README.md`` matching the GNN/DeepSets tracks.
    """
    from modeling.ablation import run_ablation_matrix

    outdir.mkdir(parents=True, exist_ok=True)
    loco_cfg = config.loco
    selected_blocks = normalize_selected_features(loco_cfg.get("selected_features"))
    if not selected_blocks:
        raise ValueError(
            "loco.selected_features (feature blocks) must be set for tabular LOCO."
        )

    covariates = _resolve_covariates(loco_cfg, base_config=config, outdir=outdir)
    baseline_arm_name = str(
        loco_cfg.get("baseline_arm_name") or _DEFAULT_BASELINE_ARM_NAME
    )

    arms = build_tabular_loco_arms(
        covariates=covariates,
        selected_blocks=selected_blocks,
        baseline_arm_name=baseline_arm_name,
    )
    arm_config = deepcopy(config)
    arm_config.ablation_arms = arms
    arm_config.baseline_arm_name = baseline_arm_name

    logging.info(
        "Tabular LOCO sweep: %d covariate(s) over blocks=%s",
        len(covariates),
        selected_blocks,
    )
    run_ablation_matrix(arm_config, outdir)

    ablation_summary = pd.read_csv(outdir / "ablation_summary.csv")
    summary_df = _melt_ablation_summary_to_loco(
        ablation_summary, baseline_arm_name=baseline_arm_name
    )
    summary_df.to_csv(outdir / "loco_summary.csv", index=False)

    fold_auc_path = outdir / "ablation_fold_auc.csv"
    fold_df = (
        _melt_ablation_fold_to_loco(
            pd.read_csv(fold_auc_path), baseline_arm_name=baseline_arm_name
        )
        if fold_auc_path.exists()
        else _melt_ablation_fold_to_loco(
            pd.DataFrame(), baseline_arm_name=baseline_arm_name
        )
    )
    fold_df.to_csv(outdir / "loco_fold_deltas.csv", index=False)

    write_loco_readme(
        outdir=outdir,
        family="tabular",
        node_mode=None,
        feature_group="column",
        covariates=covariates,
        summary_df=summary_df,
    )
    logging.info("Wrote tabular LOCO summary to %s", outdir / "loco_summary.csv")
