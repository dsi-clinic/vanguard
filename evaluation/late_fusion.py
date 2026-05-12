"""Late fusion of two out-of-fold prediction sets via fold-aligned LR-stack.

Given two ``predictions.csv`` files sharing ``case_id`` and ``fold`` columns
and produced under the same canonical fold map (e.g. ``data/fold_map_her2.csv``),
this module fits a small logistic-regression stacker on the OOF probabilities
using the same fold partitioning, so the stacker never sees a case's own
training-side prediction.

A mean-of-probabilities baseline is also reported. If the LR-stack beats the
mean-of-probs by more than a configurable threshold (default 0.02 AUC), the
stacker is likely fitting fold-specific noise; callers should fall back to the
mean-of-probs as the reportable number in that case.

The module **aborts at runtime** if either prediction file disagrees with the
fold map on a ``case_id``'s assigned fold, or if either file is missing cases
present in the other.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

MIN_CLASSES_FOR_AUC = 2
DEFAULT_LR_NOISE_THRESHOLD = 0.02


@dataclass
class FusionResult:
    """Container for late-fusion metrics on one cohort."""

    n_cases: int
    n_folds: int
    model_a_name: str
    model_b_name: str
    model_a_auc: float
    model_a_ap: float
    model_b_auc: float
    model_b_ap: float
    mean_of_probs_auc: float
    mean_of_probs_ap: float
    lr_stack_auc: float
    lr_stack_ap: float
    lr_stack_minus_mean_auc: float
    lr_likely_overfits: bool
    per_fold_lr_stack_auc: dict[int, float]
    per_fold_mean_of_probs_auc: dict[int, float]


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


def _load_predictions(path: Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"case_id", "y_true", "y_prob", "fold"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"{path}: missing required columns {sorted(missing)}; "
            f"got {list(df.columns)}"
        )
    df = df.loc[:, ["case_id", "y_true", "y_prob", "fold"]].copy()
    df["case_id"] = df["case_id"].astype(str)
    df["y_true"] = df["y_true"].astype(int)
    df["fold"] = df["fold"].astype(int)
    df = df.rename(columns={"y_prob": f"y_prob_{name}"})
    return df


def _assert_fold_map_alignment(
    df: pd.DataFrame, fold_map: pd.DataFrame, label: str
) -> None:
    merged = df.merge(
        fold_map[["case_id", "fold"]], on="case_id", how="left", suffixes=("", "_map")
    )
    missing_in_map = merged.loc[merged["fold_map"].isna(), "case_id"].tolist()
    if missing_in_map:
        raise ValueError(
            f"{label}: {len(missing_in_map)} case_ids absent from fold map; "
            f"first 5: {missing_in_map[:5]}"
        )
    mismatched = merged.loc[merged["fold"] != merged["fold_map"], "case_id"].tolist()
    if mismatched:
        raise ValueError(
            f"{label}: {len(mismatched)} case_ids disagree with fold map on "
            f"fold assignment; first 5: {mismatched[:5]}"
        )


def _filter_to_cohort(
    df: pd.DataFrame, cohort_case_ids: set[str] | None, label: str
) -> pd.DataFrame:
    if cohort_case_ids is None:
        return df
    before = len(df)
    out = df[df["case_id"].isin(cohort_case_ids)].copy()
    logging.info(
        "Filtered %s from %d to %d cases against cohort intersection",
        label,
        before,
        len(out),
    )
    return out


def fuse_oof_predictions(
    predictions_a_path: Path,
    predictions_b_path: Path,
    fold_map_path: Path,
    *,
    model_a_name: str = "model_a",
    model_b_name: str = "model_b",
    cohort_case_ids: set[str] | None = None,
    lr_noise_threshold: float = DEFAULT_LR_NOISE_THRESHOLD,
) -> FusionResult:
    """Run fold-aligned LR-stack + mean-of-probs fusion of two OOF prediction sets.

    Parameters
    ----------
    predictions_a_path, predictions_b_path
        CSVs with ``case_id, y_true, y_prob, fold`` (and any extras).
    fold_map_path
        CSV with ``case_id, fold`` (e.g. ``data/fold_map_her2.csv``). Both
        prediction files must agree with this map per case.
    model_a_name, model_b_name
        Labels used in the result and stack feature names.
    cohort_case_ids
        If provided, both inputs are filtered to this set first (used for
        the HER2 intersection cohort).
    lr_noise_threshold
        If ``lr_stack_auc - mean_of_probs_auc`` exceeds this value, the
        LR-stack is flagged as likely overfitting fold-specific noise.
    """
    fold_map = pd.read_csv(fold_map_path)
    if "case_id" not in fold_map.columns or "fold" not in fold_map.columns:
        raise ValueError(
            f"{fold_map_path}: fold map must contain 'case_id' and 'fold' columns"
        )
    fold_map["case_id"] = fold_map["case_id"].astype(str)
    fold_map["fold"] = fold_map["fold"].astype(int)

    df_a = _load_predictions(predictions_a_path, model_a_name)
    df_b = _load_predictions(predictions_b_path, model_b_name)

    df_a = _filter_to_cohort(df_a, cohort_case_ids, predictions_a_path.name)
    df_b = _filter_to_cohort(df_b, cohort_case_ids, predictions_b_path.name)
    if cohort_case_ids is not None:
        fold_map = fold_map[fold_map["case_id"].isin(cohort_case_ids)].copy()

    _assert_fold_map_alignment(df_a, fold_map, predictions_a_path.name)
    _assert_fold_map_alignment(df_b, fold_map, predictions_b_path.name)

    cases_a = set(df_a["case_id"])
    cases_b = set(df_b["case_id"])
    missing_in_b = cases_a - cases_b
    missing_in_a = cases_b - cases_a
    if missing_in_a or missing_in_b:
        raise ValueError(
            f"case_id sets disagree: "
            f"in A not B: {sorted(missing_in_b)[:5]} (n={len(missing_in_b)}); "
            f"in B not A: {sorted(missing_in_a)[:5]} (n={len(missing_in_a)})"
        )

    merged = df_a.merge(
        df_b[["case_id", f"y_prob_{model_b_name}"]],
        on="case_id",
        how="inner",
        validate="one_to_one",
    )

    y_true = merged["y_true"].to_numpy()
    a_probs = merged[f"y_prob_{model_a_name}"].to_numpy()
    b_probs = merged[f"y_prob_{model_b_name}"].to_numpy()
    folds = merged["fold"].to_numpy()

    mean_probs = (a_probs + b_probs) / 2.0
    mean_of_probs_auc = _safe_auc(y_true, mean_probs)
    mean_of_probs_ap = _safe_ap(y_true, mean_probs)

    lr_oof_probs = np.full_like(mean_probs, fill_value=np.nan, dtype=float)
    per_fold_lr_stack_auc: dict[int, float] = {}
    per_fold_mean_of_probs_auc: dict[int, float] = {}

    for fold_idx in sorted(np.unique(folds).tolist()):
        val_mask = folds == int(fold_idx)
        train_mask = ~val_mask
        x_train = np.column_stack([a_probs[train_mask], b_probs[train_mask]])
        y_train = y_true[train_mask]
        x_val = np.column_stack([a_probs[val_mask], b_probs[val_mask]])
        y_val = y_true[val_mask]

        if len(np.unique(y_train)) < MIN_CLASSES_FOR_AUC:
            lr_oof_probs[val_mask] = mean_probs[val_mask]
        else:
            clf = LogisticRegression(max_iter=1000, solver="lbfgs")
            clf.fit(x_train, y_train)
            lr_oof_probs[val_mask] = clf.predict_proba(x_val)[:, 1]

        per_fold_lr_stack_auc[int(fold_idx)] = _safe_auc(
            y_val, lr_oof_probs[val_mask]
        )
        per_fold_mean_of_probs_auc[int(fold_idx)] = _safe_auc(
            y_val, mean_probs[val_mask]
        )

    lr_stack_auc = _safe_auc(y_true, lr_oof_probs)
    lr_stack_ap = _safe_ap(y_true, lr_oof_probs)
    delta = lr_stack_auc - mean_of_probs_auc
    return FusionResult(
        n_cases=int(len(merged)),
        n_folds=int(merged["fold"].nunique()),
        model_a_name=model_a_name,
        model_b_name=model_b_name,
        model_a_auc=_safe_auc(y_true, a_probs),
        model_a_ap=_safe_ap(y_true, a_probs),
        model_b_auc=_safe_auc(y_true, b_probs),
        model_b_ap=_safe_ap(y_true, b_probs),
        mean_of_probs_auc=mean_of_probs_auc,
        mean_of_probs_ap=mean_of_probs_ap,
        lr_stack_auc=lr_stack_auc,
        lr_stack_ap=lr_stack_ap,
        lr_stack_minus_mean_auc=float(delta),
        lr_likely_overfits=bool(delta > lr_noise_threshold),
        per_fold_lr_stack_auc=per_fold_lr_stack_auc,
        per_fold_mean_of_probs_auc=per_fold_mean_of_probs_auc,
    )


def main() -> int:
    """CLI entry point for ad-hoc fusion runs."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-a", required=True, type=Path)
    parser.add_argument("--predictions-b", required=True, type=Path)
    parser.add_argument("--fold-map", required=True, type=Path)
    parser.add_argument("--model-a-name", default="model_a")
    parser.add_argument("--model-b-name", default="model_b")
    parser.add_argument(
        "--cohort-case-ids",
        type=Path,
        default=None,
        help="Optional CSV with case_id column to restrict fusion to.",
    )
    parser.add_argument(
        "--output-json", required=True, type=Path, help="Destination metrics JSON"
    )
    parser.add_argument(
        "--lr-noise-threshold",
        type=float,
        default=DEFAULT_LR_NOISE_THRESHOLD,
    )
    args = parser.parse_args()

    cohort = None
    if args.cohort_case_ids is not None:
        cohort = set(
            pd.read_csv(args.cohort_case_ids)["case_id"].astype(str).unique()
        )

    result = fuse_oof_predictions(
        predictions_a_path=args.predictions_a,
        predictions_b_path=args.predictions_b,
        fold_map_path=args.fold_map,
        model_a_name=args.model_a_name,
        model_b_name=args.model_b_name,
        cohort_case_ids=cohort,
        lr_noise_threshold=args.lr_noise_threshold,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    print(json.dumps(asdict(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
