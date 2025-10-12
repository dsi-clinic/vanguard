"""
Baseline pCR (0/1) prediction with minimal metadata features

Outputs (written to --output):
  - metrics.json: {"auc_train": float, "auc_test": float, "n_features": int, "n_train": int, "n_test": int}
  - predictions.csv: columns [patient_id, split, y_true, y_pred_score]
  - roc_test.png: ROC curve plot
  - model.pkl: saved logistic regression model

Usage:
  python baseline_pcr_simple.py
    --json-dir /path/to/jsons
    --split-csv splits_v1.csv
    --output outdir
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
from sklearn import clone

def get_patient_id(path: Path, js: Dict[str, Any]) -> str:
    return js.get("patient_id", path.stem)

def get_age(js) -> Optional[float]:
    '''
    Gets age, returns None if missing
    '''
    age = js.get("clinical_data", {}).get("age", None)
    try:
        return float(age) if age not in (None, "") else None
    except Exception:
        return None

def get_subtype(js: Dict[str, Any]) -> str:
    '''
    Gets tumor subtype, returns unknown if missing
    '''
    subtype = js.get("primary_lesion", {}).get("tumor_subtype", "")
    s = str(subtype).strip().lower()
    return s if s else "unknown"   # In the README: missing subtype = "unknown"

def get_label_optional(js: Dict[str, Any]) -> Optional[int]:
    """
    Extracts the pathologic complete response (pCR) label from the JSON file.

    Returns:
        int (0 or 1): if a valid pCR label exists.
        None: if the label is missing, blank, or malformed.

    This function is "optional"; it allows unlabeled patients to remain
    in the dataset so they can still receive predictions (y_pred_score), even though
    they are excluded from model training and AUC computation.
    """
    # Try to get pCR field
    lab = js.get("primary_lesion", {}).get("pcr", None)

    # If missing or blank, return None
    if lab in (None, ""):
        return None

    try:
        # Convert "0" or "1" (string or int) into integer form
        return int(lab)
    except Exception:
        # If conversion fails (e.g. "NA" or malformed), treat as unlabeled
        return None


def get_bbox_volume(js: Dict[str, Any]) -> Optional[float]:
    bc = js.get("primary_lesion", {}).get("breast_coordinates", {})
    try:
        x_min = float(bc.get("x_min"))
        x_max = float(bc.get("x_max"))
        y_min = float(bc.get("y_min"))
        y_max = float(bc.get("y_max"))
        z_min = float(bc.get("z_min"))
        z_max = float(bc.get("z_max"))
        dx, dy, dz = x_max - x_min, y_max - y_min, z_max - z_min
        vol = dx * dy * dz
        return vol if (dx > 0 and dy > 0 and dz > 0) else None
    except Exception:
        return None
  
# DATA LOADING

def load_dataset(json_dir: Path, split_csv: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
  splits = pd.read_csv(split_csv, comment="#").dropna(how="all") #ignores any comments, snippet of csv I saw had comment in it
  if not {"patient_id", "split"}.issubset(set(splits.columns)):
    raise ValueError("split CSV must have columns: patient_id, split")
  
  def get_splits(s: str) -> str: #getting splits from the split column in csv
      s = str(s).strip().lower()
      if s == "train":
          return "train"
      if s == "val":
          return "val"
      return "test"

  splits["split"] = splits["split"].map(get_splits)
  split_map = dict(zip(splits["patient_id"].astype(str), splits["split"]))

  rows = []
  for p in sorted(Path(json_dir).glob("*.json")):
      js = json.loads(Path(p).read_text())

      pid = get_patient_id(p, js)
      y = get_label_optional(js) 
      age = get_age(js)
      subtype = get_subtype(js)
      bbox_volume = get_bbox_volume(js)

      rows.append({
            "patient_id": pid,
            "split": split_map[pid],
            "age": age,
            "tumor_subtype": subtype,
            "bbox_volume": bbox_volume,
            "y": y,
        })
      
  df = pd.DataFrame(rows)

  df_train = df[df["split"] == "train"].copy()
  df_val   = df[df["split"] == "val"].copy()

  df_eval = df_val

  return df_train, df_eval

# Model

def build_pipeline(C: float = 1.0, max_iter: int = 1000) -> Pipeline:
    numeric_features = ["age", "bbox_volume"]
    categorical_features = ["tumor_subtype"]

    num = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),  # fill missing age/volume
    ("scaler", StandardScaler()),
    ])


    cat = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore")), #OneHot handles if unknown
    ])

    pre = ColumnTransformer([
        ("num", num, numeric_features),
        ("cat", cat, categorical_features),
    ])

    return Pipeline([
        ("preprocess", pre),
        ("clf", LogisticRegression(C=C, max_iter=max_iter, penalty="l2", solver="lbfgs")), #optimization algorithm
    ])

def train_and_eval(df_train, df_eval, outdir: Path, C=1.0, max_iter=1000):
    outdir.mkdir(parents=True, exist_ok=True)
    pipe = build_pipeline(C=C, max_iter=max_iter)

    df_train_lab = df_train.dropna(subset=["y"]).copy()
    df_eval_lab  = df_eval.dropna(subset=["y"]).copy()

    # Fit on labeled train only
    X_train = df_train_lab[["age", "bbox_volume", "tumor_subtype"]]
    y_train = df_train_lab["y"].astype(int).values
    pipe.fit(X_train, y_train)

    # Train AUC
    train_scores = pipe.predict_proba(X_train)[:, 1]
    auc_train = float(roc_auc_score(y_train, train_scores))

    # Held-out AUC
    if len(df_eval_lab):
        X_eval = df_eval_lab[["age", "bbox_volume", "tumor_subtype"]]
        y_eval = df_eval_lab["y"].astype(int).values
        eval_scores = pipe.predict_proba(X_eval)[:, 1]
        auc_eval = float(roc_auc_score(y_eval, eval_scores))
    else:
        auc_eval = float("nan")

    # Predictions for labeled + unlabeled
    pred_train = pd.DataFrame({
        "patient_id": df_train["patient_id"].values,
        "split": df_train["split"].values,
        "y_true": df_train["y"].values,  # may contain NaN
        "y_pred_score": pipe.predict_proba(df_train[["age","bbox_volume","tumor_subtype"]])[:, 1],
    })
    pred_eval = pd.DataFrame({
        "patient_id": df_eval["patient_id"].values,
        "split": df_eval["split"].values,
        "y_true": df_eval["y"].values,  # may contain NaN
        "y_pred_score": pipe.predict_proba(df_eval[["age","bbox_volume","tumor_subtype"]])[:, 1],
    })
    pred_df = pd.concat([pred_train, pred_eval], ignore_index=True)
    pred_df.to_csv(outdir / "predictions.csv", index=False)

    # ROC curve
    if len(df_eval_lab):
        RocCurveDisplay.from_predictions(y_eval, eval_scores)
        plt.title("ROC (held-out)")
        plt.tight_layout()
        plt.savefig(outdir / "roc_test.png", dpi=200)
        plt.close()
    else:
        print("[WARN] No labeled samples in eval split; skipping ROC plot.")

    # Save model
    joblib.dump(pipe, outdir / "model.pkl")

    # Count transformed features
    n_feat = clone(pipe.named_steps["preprocess"]).fit(
        df_train_lab[["age","bbox_volume","tumor_subtype"]]
    ).transform(
        df_train_lab[["age","bbox_volume","tumor_subtype"]]
    ).shape[1]

    metrics = {
        "auc_train": auc_train,
        "auc_test": auc_eval,
        "n_features": int(n_feat),
        "n_train": int(df_train_lab.shape[0]),       # labeled train
        "n_test": int(df_eval_lab.shape[0]),         # labeled eval
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Minimal pCR baseline (fixed schema).")
    ap.add_argument("--json-dir", required=True, type=Path)
    ap.add_argument("--split-csv", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--max-iter", type=int, default=1000)
    args = ap.parse_args()

    df_train, df_eval = load_dataset(args.json_dir, args.split_csv)

    print(f"Loaded {len(df_train)} train and {len(df_eval)} held-out samples.")

    metrics = train_and_eval(df_train, df_eval, args.output, C=args.C, max_iter=args.max_iter)
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()





