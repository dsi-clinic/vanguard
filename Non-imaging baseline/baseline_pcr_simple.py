"""Baseline pCR (0/1) prediction with minimal metadata features.

Outputs (written to --output):
- metrics.json: {"auc_train": float, "auc_test": float, "n_features": int,
                 "n_train": int, "n_test": int}
- predictions.csv: columns [patient_id, split, y_true, y_pred_score]
- roc_test.png: ROC curve plot (on held-out test)
- model.pkl: saved logistic regression model

Usage:
  python baseline_pcr_simple.py \
    --json-dir /path/to/jsons \
    --split-csv splits_v1.csv \
    --output outdir
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ------------------ Feature extractors ------------------


def get_patient_id(path: Path, js: dict[str, Any]) -> str:
    """Return patient_id from JSON, falling back to the filename stem."""
    return js.get("patient_id", path.stem)


def get_age(js: dict[str, Any]) -> float | None:
    """Return patient age as float, or None if missing/unparseable."""
    age = js.get("clinical_data", {}).get("age", None)
    try:
        return float(age) if age not in (None, "") else None
    except Exception:
        return None

def get_subtype(js: dict[str, Any]) -> str:
    subtype = js.get("primary_lesion", {}).get("tumor_subtype", "")
    s = str(subtype).strip().lower()
    if s in {"", "nan", "null", "unknown"}:
        return "unknown"
    return s


def get_label_optional(js: dict[str, Any]) -> int | None:
    """Return pCR label (0/1) if present, else None."""
    lab = js.get("primary_lesion", {}).get("pcr", None)
    if lab in (None, ""):
        return None
    try:
        return int(lab)
    except Exception:
        return None


def get_bbox_volume(js: dict[str, Any]) -> float | None:
    """Return 3D bbox volume if all coordinates present and valid; else None."""
    bc = js.get("primary_lesion", {}).get("breast_coordinates", {})
    try:
        x_min, x_max = float(bc.get("x_min")), float(bc.get("x_max"))
        y_min, y_max = float(bc.get("y_min")), float(bc.get("y_max"))
        z_min, z_max = float(bc.get("z_min")), float(bc.get("z_max"))
        dx, dy, dz = x_max - x_min, y_max - y_min, z_max - z_min
        vol = dx * dy * dz
        return vol if (dx > 0 and dy > 0 and dz > 0) else None
    except Exception:
        return None


# ------------------ Data loading ------------------


def load_dataset(json_dir: Path, split_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/test DataFrames.

    The CSV includes columns: patient_id, split. If there is no explicit
    'test' split, the 'val' rows are promoted to 'test' (for reporting only).
    """
    splits = pd.read_csv(split_csv, comment="#").dropna(how="all")
    if not {"patient_id", "split"}.issubset(set(splits.columns)):
        raise ValueError("split CSV must have columns: patient_id, split")

    def map_split(s: str) -> str:
        s = str(s).strip().lower()
        if s == "train":
            return "train"
        if s == "val":
            return "val"
        if s == "test":
            return "test"
        return "test"

    splits["split"] = splits["split"].map(map_split)
    split_map = dict(zip(splits["patient_id"].astype(str), splits["split"]))

    rows: list[dict[str, Any]] = []
    for p in sorted(Path(json_dir).glob("*.json")):
        js = json.loads(Path(p).read_text())
        pid = get_patient_id(p, js)
        if pid not in split_map:
            raise KeyError(f"{pid} missing in split CSV")
        rows.append(
            {
                "patient_id": pid,
                "split": split_map[pid],
                "age": get_age(js),
                "tumor_subtype": get_subtype(js),
                "bbox_volume": get_bbox_volume(js),
                "y": get_label_optional(js),
            }
        )

    df_all = pd.DataFrame(rows)

    # Use val as test when no explicit test split is present.
    if not (df_all["split"] == "test").any():
        warnings.warn(
            "No explicit TEST split found — using VAL as TEST for now.",
            stacklevel=2,
        )
        df_all.loc[df_all["split"] == "val", "split"] = "test"

    df_train = df_all[df_all["split"] == "train"].copy()
    df_test = df_all[df_all["split"] == "test"].copy()
    return df_train, df_test


# ------------------ Model ------------------


def build_pipeline(C: float = 1.0, max_iter: int = 1000) -> Pipeline:
    """Build preprocessing + logistic regression pipeline."""
    numeric_features = ["age", "bbox_volume"]
    categorical_features = ["tumor_subtype"]

    num = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    pre = ColumnTransformer(
        [
            ("num", num, numeric_features),
            ("cat", cat, categorical_features),
        ]
    )
    clf = LogisticRegression(C=C, max_iter=max_iter, penalty="l2", solver="lbfgs")

    return Pipeline(
        [
            ("preprocess", pre),
            ("clf", clf),
        ]
    )


# ------------------ Train / Test ------------------


def train_and_test(
    df_train: pd.DataFrame, df_test: pd.DataFrame, outdir: Path, C: float = 1.0, max_iter: int = 1000
) -> dict[str, Any]:
    """Fit on TRAIN, evaluate on TEST, and write artifacts to --output."""
    outdir.mkdir(parents=True, exist_ok=True)
    pipe = build_pipeline(C=C, max_iter=max_iter)

    df_train_lab = df_train.dropna(subset=["y"]).copy()
    df_test_lab = df_test.dropna(subset=["y"]).copy()

    X_train = df_train_lab[["age", "bbox_volume", "tumor_subtype"]]
    y_train = df_train_lab["y"].astype(int).to_numpy()
    pipe.fit(X_train, y_train)

    # AUCs
    train_scores = pipe.predict_proba(X_train)[:, 1]
    auc_train = float(roc_auc_score(y_train, train_scores))

    if len(df_test_lab) != 0:
        X_test = df_test_lab[["age", "bbox_volume", "tumor_subtype"]]
        y_test = df_test_lab["y"].astype(int).to_numpy()
        test_scores = pipe.predict_proba(X_test)[:, 1]
        auc_test = float(roc_auc_score(y_test, test_scores))
    else:
        auc_test = np.nan

    # Predictions (for all patients, even unlabeled)
    preds: list[pd.DataFrame] = []
    for block in [df_train, df_test]:
        preds.append(
            pd.DataFrame(
                {
                    "patient_id": block["patient_id"].to_numpy(),
                    "split": block["split"].to_numpy(),
                    "y_true": block["y"].to_numpy(),
                    "y_pred_score": pipe.predict_proba(
                        block[["age", "bbox_volume", "tumor_subtype"]]
                    )[:, 1],
                }
            )
        )
    pd.concat(preds, ignore_index=True).to_csv(outdir / "predictions.csv", index=False)

    # ROC curve on TEST
    if len(df_test_lab) != 0:
        RocCurveDisplay.from_predictions(y_test, test_scores)
        plt.title("ROC (TEST)")
        plt.tight_layout()
        plt.savefig(outdir / "roc_test.png", dpi=200)
        plt.close()
    else:
        print("[WARN] No labeled TEST samples; skipping ROC plot.")

    # Save model + metrics
    joblib.dump(pipe, outdir / "model.pkl")

    n_feat = (
        clone(pipe.named_steps["preprocess"])
        .fit(df_train_lab[["age", "bbox_volume", "tumor_subtype"]])
        .transform(df_train_lab[["age", "bbox_volume", "tumor_subtype"]])
        .shape[1]
    )

    metrics: dict[str, Any] = {
        "auc_train": auc_train,
        "auc_test": auc_test,
        "n_features": int(n_feat),
        "n_train": int(df_train_lab.shape[0]),
        "n_test": int(df_test_lab.shape[0]),
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


# ------------------ CLI ------------------


def main() -> None:
    """CLI entrypoint."""
    ap = argparse.ArgumentParser(description="Minimal pCR baseline (fixed schema).")
    ap.add_argument("--json-dir", required=True, type=Path)
    ap.add_argument("--split-csv", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--max-iter", type=int, default=1000)
    args = ap.parse_args()

    df_train, df_test = load_dataset(args.json_dir, args.split_csv)
    print(f"Loaded {len(df_train)} train and {len(df_test)} test samples.")

    metrics = train_and_test(df_train, df_test, args.output, C=args.C, max_iter=args.max_iter)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()