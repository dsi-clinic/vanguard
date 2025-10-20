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
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser("CoW data loading (graphs -> morphometrics) + labels")
    ap.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Folder with .vtp/.vtk (searched recursively)",
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Output folder (features.csv, preview)",
    )
    ap.add_argument(
        "--labels",
        type=Path,
        required=False,
        help="Optional: CSV labels file OR directory of per-case JSONs",
    )
    ap.add_argument(
        "--id-column", default="case_id", help="ID column name in labels file"
    )
    ap.add_argument(
        "--label-column",
        required=False,
        help="Binary label column (e.g., fetal_pca_variant)",
    )
    ap.add_argument(
        "--id-regex",
        default=r"^([A-Za-z0-9_-]+)",
        help="Regex to extract ID from filename stem",
    )
    ap.add_argument(
        "--model",
        choices=["rf", "lr"],
        default="rf",
        help="Baseline model: RandomForest (rf) or LogisticRegression (lr)",
    )
    ap.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    ap.add_argument("--val-size", type=float, default=0.1, help="Validation fraction")
    ap.add_argument("--random-state", type=int, default=42, help="Random seed")
    return ap.parse_args()


def build_labels_from_cow_variant_json(
    variants_dir: Path, out_csv: Path
) -> pd.DataFrame:
    """Flatten CoW variant JSONs into a single labels CSV."""
    rows = []
    for p in sorted(Path(variants_dir).glob("*.json")):
        obj = json.loads(p.read_text())
        flat = {"case_id": p.stem}
        for group, sub in obj.items():
            if isinstance(sub, dict):
                for k, v in sub.items():
                    col = re.sub(r"[^A-Za-z0-9_]", "_", f"{group}_{k}")
                    if isinstance(v, bool | np.bool_):  # UP038 fix
                        flat[col] = int(bool(v))
                    else:
                        flat[col] = v
        rows.append(flat)

    df_labels = pd.DataFrame(rows).fillna(0)
    if "fetal_L_PCA" in df_labels.columns and "fetal_R_PCA" in df_labels.columns:
        df_labels["fetal_pca_variant"] = (
            (df_labels["fetal_L_PCA"] == 1) | (df_labels["fetal_R_PCA"] == 1)
        ).astype(int)

    df_labels.to_csv(out_csv, index=False)
    print(
        f"Labels table -> {out_csv} (rows={len(df_labels)}, cols={len(df_labels.columns)})"
    )
    return df_labels


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
    """Clean and standardize raw features."""
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
            for k in ["radius", "length", "tortuosity", "curvature", "degree", "labels"]
        )
    ]
    feats = df_feats[morpho_cols].copy()
    feats["case_id"] = df_feats["case_id"]
    feats = feats.fillna(0.0)
    X = feats.drop(columns=["case_id"])
    X = (X - X.mean()) / (X.std() + 1e-8)
    X["case_id"] = feats["case_id"]
    X.to_csv(out_csv, index=False)
    print(f"Engineered features -> {out_csv} ({X.shape[1]} cols)")


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate

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
    """Train baseline RandomForest or LogisticRegression model with scaling and CV."""
    if labels_source.is_dir():
        labels_csv = outdir / "labels_from_json.csv"
        build_labels_from_cow_variant_json(labels_source, labels_csv)
    else:
        labels_csv = labels_source

    X = pd.read_csv(feats_engineered_csv)
    y_df = pd.read_csv(labels_csv)

    if id_col != "case_id" and id_col in y_df.columns:
        y_df = y_df.rename(columns={id_col: "case_id"})

    merged_df = X.merge(y_df, on="case_id", how="inner")
    y = merged_df[label_col].astype(int).to_numpy()
    Xmat = merged_df.drop(columns=["case_id", label_col]).to_numpy()

    if model == "rf":
        base_model = RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced_subsample",
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

    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("model", base_model),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_results = cross_validate(
        clf,
        Xmat,
        y,
        cv=cv,
        scoring=["precision", "recall", "f1"],
        n_jobs=-1,
    )

    print("\n[Cross-Validation Results]")
    print(
        f"F1 mean={cv_results['test_f1'].mean():.3f} "
        f"(±{cv_results['test_f1'].std():.3f}) | "
        f"Precision={cv_results['test_precision'].mean():.3f} | "
        f"Recall={cv_results['test_recall'].mean():.3f}"
    )

    # --- Train/val/test split (for final model reporting) ---
    X_trval, X_te, y_trval, y_te = train_test_split(
        Xmat, y, test_size=test_size, random_state=random_state, stratify=y
    )
    rel_val = val_size / (1.0 - test_size)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_trval, y_trval, test_size=rel_val, random_state=random_state, stratify=y_trval
    )

    clf.fit(X_tr, y_tr)
    y_val_hat = clf.predict(X_val)
    y_te_hat = clf.predict(X_te)

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
            "f1_mean": float(cv_results["test_f1"].mean()),
            "f1_std": float(cv_results["test_f1"].std()),
            "precision_mean": float(cv_results["test_precision"].mean()),
            "recall_mean": float(cv_results["test_recall"].mean()),
        },
    }

    dump(clf, outdir / "model.pkl")
    (outdir / "metrics.json").write_text(json.dumps(results, indent=2))

    # --- Print summary ---
    print(json.dumps(results, indent=2))
    print(f"Model -> {outdir/'model.pkl'}")
    print(f"Metrics -> {outdir/'metrics.json'}")

    # --- Confusion matrices ---
    from sklearn.metrics import confusion_matrix
    print("\n[Confusion Matrix] Validation:")
    print(confusion_matrix(y_val, y_val_hat))
    print("\n[Confusion Matrix] Test:")
    print(confusion_matrix(y_te, y_te_hat))


def main() -> None:
    """Pipeline entrypoint: build features, labels, and train baseline model."""
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    files = rglob_polydata(args.data_dir)
    print(f"Found {len(files)} polydata files under {args.data_dir}")
    feats = build_features(files, args.id_regex)

    feats_path = args.outdir / "features.csv"
    feats.to_csv(feats_path, index=False)
    print(f"[ok] features -> {feats_path} (n={len(feats)})")

    eng_path = args.outdir / "features_engineered.csv"
    engineer_features(feats_path, eng_path)

    if args.labels and args.label_column:
        labels_for_preview = args.labels
        if labels_for_preview.is_dir():
            labels_for_preview = args.outdir / "labels_from_json.csv"
            build_labels_from_cow_variant_json(args.labels, labels_for_preview)

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
