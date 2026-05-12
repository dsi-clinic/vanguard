#!/usr/bin/env python3
"""Phase 0 log(N_points) one-feature logistic-regression baseline on HER2.

Question: does the existing Deep Sets HER2 AUC come from per-point structure
or just from tumor scale (point count)? This baseline fits a single-feature
LR on log(num_points + 1) using the canonical HER2 fold map, then reports
overall and per-fold AUC. If it reaches a DS Phase 3 winner's HER2 AUC, the
project is currently a tumor-scale detector.

Output: one ``predictions.csv`` in the standard schema plus a small
metrics JSON.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

MIN_CLASSES_FOR_AUC = 2
DEFAULT_THRESHOLD = 0.5


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/deepsets_ispy2_pointfeat_baseline/deepsets_manifest.csv"),
    )
    parser.add_argument("--fold-map", type=Path, default=Path("data/fold_map_her2.csv"))
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("experiments/her2_phase0/logn_lr"),
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


def run(manifest_path: Path, fold_map_path: Path, outdir: Path) -> dict:
    manifest = pd.read_csv(manifest_path)
    manifest["case_id"] = manifest["case_id"].astype(str)
    fold_map = pd.read_csv(fold_map_path)
    fold_map["case_id"] = fold_map["case_id"].astype(str)

    merged = fold_map.merge(
        manifest[["case_id", "num_points", "tumor_subtype"]],
        on="case_id",
        how="left",
        validate="one_to_one",
    )
    if merged["num_points"].isna().any():
        missing = merged.loc[merged["num_points"].isna(), "case_id"].tolist()[:5]
        raise ValueError(f"Fold map case_ids missing from manifest: {missing}")

    merged["log_num_points"] = np.log1p(merged["num_points"].astype(float))

    rows: list[dict[str, object]] = []
    fold_aucs: dict[int, float] = {}
    fold_aps: dict[int, float] = {}

    for fold_idx in sorted(merged["fold"].unique()):
        val_mask = merged["fold"] == fold_idx
        train_mask = ~val_mask
        x_train = merged.loc[train_mask, ["log_num_points"]].to_numpy()
        y_train = merged.loc[train_mask, "label"].astype(int).to_numpy()
        x_val = merged.loc[val_mask, ["log_num_points"]].to_numpy()
        y_val = merged.loc[val_mask, "label"].astype(int).to_numpy()

        scaler = StandardScaler().fit(x_train)
        x_train_s = scaler.transform(x_train)
        x_val_s = scaler.transform(x_val)

        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf.fit(x_train_s, y_train)
        y_prob_val = clf.predict_proba(x_val_s)[:, 1]

        fold_aucs[int(fold_idx)] = _safe_auc(y_val, y_prob_val)
        fold_aps[int(fold_idx)] = _safe_ap(y_val, y_prob_val)

        for case_id, y_t, y_p, subtype in zip(
            merged.loc[val_mask, "case_id"].astype(str).tolist(),
            y_val.tolist(),
            y_prob_val.tolist(),
            merged.loc[val_mask, "tumor_subtype"].astype(str).tolist(),
            strict=True,
        ):
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
        "model": "log_num_points_lr",
        "n_cases": int(len(preds_df)),
        "n_folds": int(merged["fold"].nunique()),
        "pooled_auc": _safe_auc(y_true_all, y_prob_all),
        "pooled_ap": _safe_ap(y_true_all, y_prob_all),
        "per_fold_auc": fold_aucs,
        "per_fold_ap": fold_aps,
        "mean_fold_auc": float(np.nanmean(list(fold_aucs.values()))),
        "std_fold_auc": float(np.nanstd(list(fold_aucs.values()))),
        "fold_map_path": str(fold_map_path),
        "manifest_path": str(manifest_path),
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    metrics = run(args.manifest, args.fold_map, args.outdir)
    print(f"Wrote predictions and metrics to {args.outdir}")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
