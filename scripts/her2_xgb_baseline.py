#!/usr/bin/env python3
"""Phase 0e XGBoost baseline on the HER2 intersection cohort.

Trains XGBoost on the n=68 HER2 \u2229 tabular intersection cohort using the
canonical fold map (``data/fold_map_her2.csv``). Hyperparameters mirror the
``vessel_all`` arm of ``configs/issue118_baseline_arms.yaml`` but nested
feature selection and inner CV tuning are dropped because n=68 is far too
small for either.

This produces the comparator OOF predictions for late fusion and the
``paired_wilcoxon_p_vs_xgb`` reference in the tracker. Note that this number
is **not** identical to the existing 0.676 in
``results/model_family_robustness_ispy2_subtype_summary.csv`` — that
comparator was produced by a full-cohort 5-fold XGB whose HER2-stratum AUC
was read post hoc. The HER2-only 3-fold rerun here is the methodologically
consistent comparator for HER2-only Deep Sets.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from features import ANNOTATION_COLUMNS, feature_block_for_column

MIN_CLASSES_FOR_AUC = 2
DEFAULT_THRESHOLD = 0.5
DEFAULT_FEATURE_BLOCKS = ("clinical", "tumor_size", "morph", "graph", "kinematic")
DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "gamma": 0.0,
    "random_state": 42,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_jobs": -1,
    "tree_method": "hist",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=Path(
            "experiments/independent_signal_q3_array/features_full_labeled.csv"
        ),
    )
    parser.add_argument("--fold-map", type=Path, default=Path("data/fold_map_her2.csv"))
    parser.add_argument(
        "--label-col",
        default="pcr",
        help="Label column in features CSV (default: pcr).",
    )
    parser.add_argument(
        "--feature-blocks",
        nargs="+",
        default=list(DEFAULT_FEATURE_BLOCKS),
        help="Feature block names to keep (any of clinical/tumor_size/morph/graph/kinematic).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("experiments/her2_phase0/xgb_vessel_all"),
    )
    return parser.parse_args()


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < MIN_CLASSES_FOR_AUC:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return float("nan")


def _safe_ap(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < MIN_CLASSES_FOR_AUC:
        return float("nan")
    try:
        return float(average_precision_score(y_true, y_prob))
    except ValueError:
        return float("nan")


def _select_feature_columns(
    df: pd.DataFrame, blocks: list[str], label_col: str
) -> list[str]:
    blocks_set = set(blocks)
    selected = [
        col
        for col in df.columns
        if col not in ANNOTATION_COLUMNS
        and col != label_col
        and feature_block_for_column(col) in blocks_set
    ]
    return selected


def run(
    features_path: Path,
    fold_map_path: Path,
    label_col: str,
    feature_blocks: list[str],
    outdir: Path,
) -> dict[str, Any]:
    from xgboost import XGBClassifier

    features = pd.read_csv(features_path)
    features["case_id"] = features["case_id"].astype(str)
    fold_map = pd.read_csv(fold_map_path)
    fold_map["case_id"] = fold_map["case_id"].astype(str)

    merged = fold_map.merge(features, on="case_id", how="left", validate="one_to_one")
    if merged[label_col].isna().any():
        missing = merged.loc[merged[label_col].isna(), "case_id"].tolist()[:5]
        raise ValueError(
            f"Fold map case_ids missing from features (label NaN): {missing}"
        )
    merged[label_col] = merged[label_col].astype(int)

    feature_cols = _select_feature_columns(features, feature_blocks, label_col)
    logging.info(
        "Using %d feature columns across blocks %s", len(feature_cols), feature_blocks
    )
    if not feature_cols:
        raise ValueError(
            f"No feature columns selected for blocks {feature_blocks}; "
            f"available blocks: {sorted({feature_block_for_column(c) for c in features.columns} - {None})}"
        )

    object_cols = [
        c for c in feature_cols if merged[c].dtype == "object" or str(merged[c].dtype) == "string"
    ]
    if object_cols:
        logging.info(
            "One-hot encoding %d object columns: %s", len(object_cols), object_cols
        )
        merged = pd.get_dummies(
            merged, columns=object_cols, prefix=object_cols, dummy_na=True
        )
        feature_cols = [c for c in feature_cols if c not in object_cols]
        new_oh_cols = [
            c
            for c in merged.columns
            if any(c.startswith(f"{oc}_") for oc in object_cols)
        ]
        feature_cols = feature_cols + new_oh_cols
        logging.info("Feature dim after one-hot: %d", len(feature_cols))

    rows: list[dict[str, object]] = []
    fold_aucs: dict[int, float] = {}
    fold_aps: dict[int, float] = {}

    sanitized = {
        c: c.replace("[", "_").replace("]", "_").replace("<", "lt").replace(">", "gt")
        for c in feature_cols
    }
    if any(k != v for k, v in sanitized.items()):
        logging.info(
            "Sanitizing %d feature names with reserved chars",
            sum(1 for k, v in sanitized.items() if k != v),
        )
    merged = merged.rename(columns=sanitized)
    feature_cols = [sanitized[c] for c in feature_cols]

    for fold_idx in sorted(merged["fold"].unique()):
        val_mask = merged["fold"] == fold_idx
        train_mask = ~val_mask
        x_train = merged.loc[train_mask, feature_cols]
        y_train = merged.loc[train_mask, label_col].astype(int).to_numpy()
        x_val = merged.loc[val_mask, feature_cols]
        y_val = merged.loc[val_mask, label_col].astype(int).to_numpy()

        clf = XGBClassifier(**DEFAULT_XGB_PARAMS)
        clf.fit(x_train, y_train)
        y_prob_val = clf.predict_proba(x_val)[:, 1]

        fold_aucs[int(fold_idx)] = _safe_auc(y_val, y_prob_val)
        fold_aps[int(fold_idx)] = _safe_ap(y_val, y_prob_val)

        subtype_lookup = (
            features.set_index("case_id")["tumor_subtype"]
            if "tumor_subtype" in features.columns
            else None
        )
        for case_id, y_t, y_p in zip(
            merged.loc[val_mask, "case_id"].astype(str).tolist(),
            y_val.tolist(),
            y_prob_val.tolist(),
            strict=True,
        ):
            subtype = (
                str(subtype_lookup.get(case_id, "")) if subtype_lookup is not None else ""
            )
            rows.append(
                {
                    "case_id": case_id,
                    "y_true": int(y_t),
                    "y_pred": int(y_p >= DEFAULT_THRESHOLD),
                    "y_prob": float(y_p),
                    "fold": int(fold_idx),
                    "stratum": subtype,
                }
            )

    preds_df = pd.DataFrame(rows)
    outdir.mkdir(parents=True, exist_ok=True)
    preds_df.to_csv(outdir / "predictions.csv", index=False)

    y_true_all = preds_df["y_true"].to_numpy()
    y_prob_all = preds_df["y_prob"].to_numpy()
    metrics = {
        "model": "xgb_her2_only_vessel_all",
        "n_cases": int(len(preds_df)),
        "n_folds": int(merged["fold"].nunique()),
        "n_features": int(len(feature_cols)),
        "feature_blocks": list(feature_blocks),
        "pooled_auc": _safe_auc(y_true_all, y_prob_all),
        "pooled_ap": _safe_ap(y_true_all, y_prob_all),
        "per_fold_auc": fold_aucs,
        "per_fold_ap": fold_aps,
        "mean_fold_auc": float(np.nanmean(list(fold_aucs.values()))),
        "std_fold_auc": float(np.nanstd(list(fold_aucs.values()))),
        "fold_map_path": str(fold_map_path),
        "features_path": str(features_path),
        "xgb_params": DEFAULT_XGB_PARAMS,
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    metrics = run(
        features_path=args.features,
        fold_map_path=args.fold_map,
        label_col=args.label_col,
        feature_blocks=args.feature_blocks,
        outdir=args.outdir,
    )
    print(f"Wrote predictions and metrics to {args.outdir}")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
