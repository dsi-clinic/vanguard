"""CoW (Circle of Willis) data loader and baseline classifier.

This script extracts morphometric features from 3D .vtp/.vtk polydata files,
joins them with variant labels from JSON or CSV, engineers normalized features,
and trains a baseline model (RandomForest or LogisticRegression).

Outputs:
- features.csv / features_engineered.csv
- labels_from_json.csv (if label JSONs given)
- model.pkl / metrics.json
"""

import argparse
import json
import logging
import numbers
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROC_FLIP_THRESHOLD = 0.5
DEFAULT_PROBA_THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for feature building and baseline training."""
    ap = argparse.ArgumentParser(
        "CoW feature extraction + baseline classification (graphs -> morphometrics)"
    )

    ap.add_argument(
        "--cow-feature-dir",
        type=Path,
        required=True,
        help="Directory of per-case CoW feature JSONs (one <case_id>.json per case)",
    )
    ap.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="CSV file or directory of per-case variant JSONs",
    )
    ap.add_argument(
        "--label-column",
        required=True,
        help="Binary label column to learn (e.g., fetal_pca_variant)",
    )
    ap.add_argument(
        "--id-column",
        default="case_id",
        help="ID column name in labels table (default: case_id)",
    )

    # output + modeling
    ap.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Output directory (features.csv, engineered.csv, model, metrics)",
    )
    ap.add_argument("--model", choices=["rf", "lr"], default="rf")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--val-size", type=float, default=0.1)
    ap.add_argument("--random-state", type=int, default=42)
    return ap.parse_args()


def build_features_from_cow_feature_jsons(feature_dir: Path) -> pd.DataFrame:
    """Read per-case CoW feature JSONs.

    Expected structure like:
    { "1": {"BA":[{...}], "BA bifurcation":[{...}]}, "2": {"PCA":[{...}], ...}, ... }
    Aggregate numeric stats for each vessel/section across all groups.
    Returns one row per case_id with lots of numeric columns.
    """
    rows: list[dict[str, float]] = []

    def to_num(x: object) -> float | None:
        if isinstance(x, bool):
            return float(x)
        if isinstance(x, numbers.Real):
            return float(x)
        if isinstance(x, str):
            try:
                return float(x.strip())
            except Exception:
                return None
        return None

    for p in sorted(feature_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text())
        except Exception as e:
            logging.warning("Skipping feature JSON %s due to parse error: %s", p, e)
            continue

        case_id = p.stem
        feats: dict[str, float] = {"case_id": case_id}

        if isinstance(data, dict):
            for _, group in data.items():
                if not isinstance(group, dict):
                    continue
                for (
                    vessel_name,
                    items,
                ) in group.items():
                    if not isinstance(items, list):
                        continue
                    # each item is a dict with keys like segment/radius/curvature/length/tortuosity/bifurcation/angles/ratios
                    per_item_vals: dict[str, list[float]] = {}
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        for k, v in item.items():
                            if isinstance(v, dict):
                                for kk, vv in v.items():
                                    if isinstance(vv, list):
                                        nums = [to_num(t) for t in vv]
                                        nums = [t for t in nums if t is not None]
                                        if nums:
                                            per_item_vals.setdefault(
                                                f"{k}__{kk}", []
                                            ).extend(nums)
                                    else:
                                        num = to_num(vv)
                                        if num is not None:
                                            per_item_vals.setdefault(
                                                f"{k}__{kk}", []
                                            ).append(num)
                            elif isinstance(v, list):
                                nums = [to_num(t) for t in v]
                                nums = [t for t in nums if t is not None]
                                if nums:
                                    per_item_vals.setdefault(k, []).extend(nums)
                            else:
                                num = to_num(v)
                                if num is not None:
                                    per_item_vals.setdefault(k, []).append(num)

                    vprefix = vessel_name.replace(" ", "_")
                    for fld, vals in per_item_vals.items():
                        if not vals:
                            continue
                        vals = [float(x) for x in vals]
                        base = f"{vprefix}__{fld}"
                        feats[f"{base}__mean"] = float(np.mean(vals))
                        feats[f"{base}__std"] = (
                            float(np.std(vals)) if len(vals) > 1 else 0.0
                        )
                        feats[f"{base}__min"] = float(np.min(vals))
                        feats[f"{base}__max"] = float(np.max(vals))
                        feats[f"{base}__count"] = float(len(vals))

        rows.append(feats)

    df_rows = pd.DataFrame(rows)
    # keep numeric + case_id
    num = df_rows.select_dtypes(include=["number"]).copy()
    if "case_id" in df_rows.columns and "case_id" not in num.columns:
        num = pd.concat([df_rows["case_id"], num], axis=1)
    num = num.dropna(axis=1, how="all")

    # vectorized constant detection
    cols_no_id = [c for c in num.columns if c != "case_id"]
    const_mask = num[cols_no_id].apply(lambda s: s.eq(s.iloc[0]).all())
    constant = [c for c, is_const in const_mask.items() if is_const]
    num = num.drop(columns=constant, errors="ignore")

    return num


def find_binary_label_columns(df_labels: pd.DataFrame) -> list[str]:
    """Return columns that contain only binary 0/1 values."""
    bins: list[str] = []
    for c in df_labels.columns:
        if c == "case_id":
            continue
        vals = set(
            pd.Series(df_labels[c]).dropna().astype(float).astype(int).unique().tolist()
        )
        if vals.issubset({0, 1}):
            bins.append(c)
    return bins


def rglob_polydata(root: Path) -> list[Path]:
    """Recursively find all .vtp and .vtk files."""
    return sorted(root.rglob("*.vtp")) + sorted(root.rglob("*.vtk"))


def extract_id(path: Path, id_regex: str) -> str:
    """Extract case ID from filename using regex."""
    m = re.match(id_regex, path.stem)
    return m.group(1) if m else path.stem


def collect_numeric(poly: pv.PolyData) -> dict[str, np.ndarray]:
    """Collect numeric point/cell arrays from polydata."""
    arrays: dict[str, np.ndarray] = {}
    for k in poly.point_data.keys():
        a = np.asarray(poly.point_data[k])
        if a.dtype.kind in "iuf" and a.size:
            arrays[f"pt::{k}"] = a.astype(float)
    for k in poly.cell_data.keys():
        a = np.asarray(poly.cell_data[k])
        if a.dtype.kind in "iuf" and a.size:
            arrays[f"cell::{k}"] = a.astype(float)
    return arrays


def summarize(arr: np.ndarray, prefix: str) -> dict[str, float]:
    """Summarize an array with descriptive statistics."""
    arr = arr[np.isfinite(arr)]
    if not arr.size:
        return {}
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_p25": float(np.percentile(arr, 25)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_p75": float(np.percentile(arr, 75)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_sum": float(np.sum(arr)),
        f"{prefix}_count": float(arr.size),
    }


def poly_to_row(poly: pv.PolyData, file: Path) -> dict[str, float]:
    """Convert a polydata object into a feature row."""
    row: dict[str, float] = {"source_file": str(file)}
    xmin, xmax, ymin, ymax, zmin, zmax = poly.bounds
    dx, dy, dz = xmax - xmin, ymax - ymin, zmax - zmin
    row.update(
        {
            "bbox_x": dx,
            "bbox_y": dy,
            "bbox_z": dz,
            "bbox_volume": dx * dy * dz,
            "n_points": float(poly.n_points),
            "n_cells": float(poly.n_cells),
        }
    )
    for name, arr in collect_numeric(poly).items():
        row.update(summarize(arr, name))
    return row


def build_features(files: list[Path], id_regex: str) -> pd.DataFrame:
    """Read all polydata files and extract numeric features."""
    rows = []
    for f in files:
        poly = pv.read(f)
        r = poly_to_row(poly, f)
        r["case_id"] = extract_id(f, id_regex)
        rows.append(r)
    return pd.DataFrame(rows).dropna(axis=1, how="all")


def load_labels(path: Path, id_col: str, label_col: str) -> pd.DataFrame:
    """Load label data from a CSV or JSON file."""
    if path.suffix.lower() == ".csv":
        df_labels = pd.read_csv(path)
    else:
        obj = json.loads(path.read_text())
        df_labels = (
            pd.DataFrame(obj)
            if isinstance(obj, list)
            else pd.DataFrame.from_dict(obj, orient="index")
            .reset_index()
            .rename(columns={"index": id_col})
        )
    mapping = {"true": 1, "false": 0, "yes": 1, "no": 0}
    df_labels[label_col] = (
        pd.Series(df_labels[label_col])
        .map(lambda v: mapping.get(str(v).lower(), v))
        .astype(int)
    )
    return df_labels[[id_col, label_col]]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute precision, recall, f1, and related statistics."""
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
    }


def engineer_features(in_csv: Path, out_csv: Path) -> None:
    """Select and export engineered morphometric features (no IDs/labels)."""
    df_feats = pd.read_csv(in_csv)

    drop_cols = [
        "source_file",
        "bbox_x",
        "bbox_y",
        "bbox_z",
        "bbox_volume",
        "n_points",
        "n_cells",
    ]
    df_feats = df_feats.drop(columns=[c for c in drop_cols if c in df_feats.columns])

    morpho_cols = [
        c
        for c in df_feats.columns
        if any(
            k in c.lower()
            for k in [
                "radius",
                "length",
                "tortuosity",
                "curvature",
                "angle",
                "area",
                "volume",
            ]
        )
    ]
    if not morpho_cols:
        morpho_cols = df_feats.select_dtypes(
            include=["number", "bool"]
        ).columns.tolist()

    morpho_cols = [
        c
        for c in morpho_cols
        if c.lower() not in {"case_id", "label"} and not c.endswith("_variant")
    ]

    X = df_feats[morpho_cols].fillna(0.0).copy()
    X["case_id"] = df_feats["case_id"]
    X.to_csv(out_csv, index=False)
    print(f"Engineered features -> {out_csv} ({X.shape[1]} cols)")


def train_baseline(
    feats_engineered_csv: Path,
    labels_source: Path,
    id_col: str,
    label_col: str,
    outdir: Path,
    test_size: float = 0.2,
    random_state: int = 42,
    model: str = "rf",
    val_size: float = 0.1,
) -> None:
    """Train RF/LR with CV metrics, threshold tuning, and score flip safeguard."""
    if labels_source.is_dir():
        labels_csv = outdir / "labels_from_json.csv"
        rows = []
        for jp in sorted(labels_source.glob("*.json")):
            try:
                obj = json.loads(jp.read_text())
            except Exception as e:
                logging.warning("Skipping label JSON %s due to parse error: %s", jp, e)
                continue
            case_id = jp.stem
            val = obj.get(label_col)
            if isinstance(val, str):
                val = {"true": 1, "false": 0, "yes": 1, "no": 0}.get(val.lower(), val)
            try:
                val = int(val)
            except Exception as e:
                logging.warning(
                    "Skipping label JSON %s missing/invalid '%s': %s", jp, label_col, e
                )
                continue
            rows.append({"case_id": case_id, label_col: val})
        pd.DataFrame(rows).to_csv(labels_csv, index=False)
    else:
        labels_csv = labels_source

    # --- load and merge
    X = pd.read_csv(feats_engineered_csv)
    y_df = pd.read_csv(labels_csv)
    if id_col != "case_id" and id_col in y_df.columns:
        y_df = y_df.rename(columns={id_col: "case_id"})
    merged_df = X.merge(y_df, on="case_id", how="inner")

    y = merged_df[label_col].astype(int).to_numpy()
    drop_cols = ["case_id", label_col] + [
        c for c in merged_df.columns if c.endswith("_variant")
    ]
    Xmat = merged_df.drop(columns=drop_cols, errors="ignore").to_numpy()

    # ---- build classifier ----
    if model == "rf":
        clf = RandomForestClassifier(
            n_estimators=800,
            max_depth=None,
            min_samples_leaf=1,
            min_samples_split=2,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        )
    else:
        base_model = LogisticRegression(
            class_weight="balanced",
            solver="liblinear",
            max_iter=2000,
            random_state=random_state,
        )
        clf = Pipeline([("scaler", StandardScaler()), ("model", base_model)])

    # ---- CV (get probabilities for class=1, auto-flip if inverted) ----
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    probs_cv = cross_val_predict(clf, Xmat, y, cv=cv, method="predict_proba")
    scores_cv = probs_cv[:, 1] if probs_cv.shape[1] > 1 else probs_cv.ravel()

    roc_cv = roc_auc_score(y, scores_cv)
    ap_cv = average_precision_score(y, scores_cv)
    scores_flipped = False
    if roc_cv < ROC_FLIP_THRESHOLD:
        scores_cv = 1.0 - scores_cv
        roc_cv = 1.0 - roc_cv
        ap_cv = average_precision_score(y, scores_cv)
        scores_flipped = True

    y_cv_hat = (scores_cv >= DEFAULT_PROBA_THRESHOLD).astype(int)
    cv_f1 = f1_score(y, y_cv_hat, zero_division=0)
    cv_prec = precision_score(y, y_cv_hat, zero_division=0)
    cv_rec = recall_score(y, y_cv_hat, zero_division=0)

    print("\n[Cross-Validation Results]")
    print(
        f"F1@0.5={cv_f1:.3f} | Precision@0.5={cv_prec:.3f} | Recall@0.5={cv_rec:.3f} "
        f"| ROC-AUC={roc_cv:.3f} | AP={ap_cv:.3f} | flipped={scores_flipped}"
    )

    # ---- holdout splits ----
    X_trval, X_te, y_trval, y_te = train_test_split(
        Xmat, y, test_size=test_size, random_state=random_state, stratify=y
    )
    rel_val = val_size / (1.0 - test_size)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_trval, y_trval, test_size=rel_val, random_state=random_state, stratify=y_trval
    )

    # ---- fit on train ----
    clf.fit(X_tr, y_tr)

    # helper to pull proba for class=1 and flip if we detected inversion
    def pos_scores(model: object, Xb: np.ndarray) -> np.ndarray:
        P = model.predict_proba(Xb)
        s = P[:, 1] if P.shape[1] > 1 else P.ravel()
        return 1.0 - s if scores_flipped else s

    # ---- tune threshold on validation ----
    val_scores = pos_scores(clf, X_val)
    ths = np.linspace(0.05, 0.95, 19)
    f1s = [f1_score(y_val, (val_scores >= t).astype(int)) for t in ths]
    best_t = ths[int(np.argmax(f1s))]

    # ---- predict using tuned threshold ----
    y_val_hat = (val_scores >= best_t).astype(int)
    test_scores = pos_scores(clf, X_te)
    y_te_hat = (test_scores >= best_t).astype(int)

    print(
        f"\n[Threshold] best_t={best_t:.2f} | "
        f"val_pos_rate_pred={y_val_hat.mean():.3f} | test_pos_rate_pred={y_te_hat.mean():.3f}"
    )

    # ---- pack results (tuned preds) ----
    results = {
        "model": model,
        "val": metrics(y_val, y_val_hat),
        "test": metrics(y_te, y_te_hat),
        "splits": {
            "train_n": int(len(y_tr)),
            "val_n": int(len(y_val)),
            "test_n": int(len(y_te)),
            "test_size": float(test_size),
            "val_size": float(val_size),
        },
        "cross_val": {
            "f1_mean": float(cv_f1),
            "f1_std": 0.0,  # single summary at 0.5 threshold; (kept for backward compat)
            "precision_mean": float(cv_prec),
            "recall_mean": float(cv_rec),
            "roc_auc_mean": float(roc_cv),
            "ap_mean": float(ap_cv),
        },
        "best_threshold": float(best_t),
        "scores_flipped": bool(scores_flipped),
    }

    model_path = outdir / f"model_{model}.pkl"
    metrics_path = outdir / f"metrics_{model}.json"
    dump(clf, model_path)
    metrics_path.write_text(json.dumps(results, indent=2))

    print(json.dumps(results, indent=2))
    print(f"Model -> {model_path}")
    print(f"Metrics -> {metrics_path}")


def main() -> None:
    """CLI entry point: build features, join labels, train model, write artifacts."""
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # --- USE CoW feature JSONs directly ---
    feats = build_features_from_cow_feature_jsons(args.cow_feature_dir)

    feats_path = args.outdir / "features.csv"
    feats.to_csv(feats_path, index=False)
    print(f"[ok] features -> {feats_path} (n={len(feats)})")

    eng_path = args.outdir / "features_engineered.csv"
    engineer_features(feats_path, eng_path)

    if args.labels and args.label_column:
        labels_for_preview = args.labels
        if labels_for_preview.is_dir():
            labels_for_preview = args.outdir / "labels_from_json.csv"
            # build labels CSV from the JSON directory (expects per-case files)
            rows = []
            for jp in sorted(Path(args.labels).glob("*.json")):
                try:
                    obj = json.loads(jp.read_text())
                except Exception as e:
                    logging.warning(
                        "Skipping label JSON %s due to parse error: %s", jp, e
                    )
                    continue

                case_id = jp.stem
                # map to 0/1 for the requested label column if present; otherwise skip
                val = obj.get(args.label_column)
                if isinstance(val, str):
                    val = {"true": 1, "false": 0, "yes": 1, "no": 0}.get(
                        val.lower(), val
                    )
                try:
                    val = int(val)
                except Exception as e:
                    logging.warning(
                        "Skipping label JSON %s missing/invalid '%s': %s",
                        jp,
                        args.label_column,
                        e,
                    )
                    continue
                rows.append({"case_id": case_id, args.label_column: val})
            pd.DataFrame(rows).to_csv(labels_for_preview, index=False)

        labels_df = pd.read_csv(labels_for_preview)
        if args.id_column != "case_id" and args.id_column in labels_df.columns:
            labels_df = labels_df.rename(columns={args.id_column: "case_id"})
        preview = feats.merge(labels_df, on="case_id", how="inner")
        prev_path = args.outdir / "features_join_preview.csv"
        preview.head(50).to_csv(prev_path, index=False)
        print(f"Join preview -> {prev_path} (rows={len(preview)})")

        train_baseline(
            feats_engineered_csv=eng_path,
            labels_source=labels_for_preview,
            id_col=args.id_column,
            label_col=args.label_column,
            outdir=args.outdir,
            test_size=args.test_size,
            random_state=args.random_state,
            model=args.model,
            val_size=args.val_size,
        )


if __name__ == "__main__":
    main()
