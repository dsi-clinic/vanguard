"""Feature importance for the baseline pCR model.

Loads a trained model pipeline (model.pkl), lines up logistic-regression
coefficients with the *transformed* feature names, computes permutation
importance on a validation set (or test if no val), and writes a CSV and plot.

Outputs (to --output):
- feature_importance.csv  (feature_name, coef, abs_coef, permutation_importance)
- feature_importance.png  (top features by |coef|)

Usage:
  python feature_importance.py \
    --model /path/to/output/model.pkl \
    --json-dir /path/to/jsons \
    --split-csv /path/to/splits_v1.csv \
    --output /path/to/output_dir \
    [--top-k 20] [--n-repeats 20]
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

FEATURES = ["age", "bbox_volume", "tumor_subtype"]

def get_patient_id(path: Path, js: dict[str, Any]) -> str:
    return js.get("patient_id", path.stem)


def get_age(js: dict[str, Any]) -> float | None:
    age = js.get("clinical_data", {}).get("age", None)
    try:
        return float(age) if age not in (None, "") else None
    except Exception:
        return None


def get_subtype(js: dict[str, Any]) -> str:
    raw = js.get("primary_lesion", {}).get("tumor_subtype", "")
    s = str(raw).strip().lower()
    if s == "nan":
        return "unknown"
    return s


def get_label_optional(js: dict[str, Any]) -> int | None:
    lab = js.get("primary_lesion", {}).get("pcr", None)
    if lab in (None, ""):
        return None
    try:
        return int(lab)
    except Exception:
        return None


def get_bbox_volume(js: dict[str, Any]) -> float | None:
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


def load_splits(json_dir: Path, split_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return train and held-out (val) DataFrames."""
    splits = pd.read_csv(split_csv, comment="#").dropna(how="all")
    if not {"patient_id", "split"}.issubset(splits.columns):
        raise ValueError("split CSV must have columns: patient_id, split")

    splits["split"] = splits["split"].str.strip().str.lower()
    split_map = dict(zip(splits["patient_id"].astype(str), splits["split"]))

    rows: list[dict[str, Any]] = []
    for p in sorted(Path(json_dir).glob("*.json")):
        js = json.loads(Path(p).read_text())
        pid = get_patient_id(p, js)
        if pid not in split_map:
            continue
        rows.append(
            {
                "patient_id": pid,
                "split": split_map[pid],
                "age": get_age(js),
                "bbox_volume": get_bbox_volume(js),
                "tumor_subtype": get_subtype(js),
                "y": get_label_optional(js),
            }
        )
    df_all = pd.DataFrame(rows)
    df_train = df_all[df_all["split"] == "train"].copy()
    df_val = df_all[df_all["split"] == "val"].copy()
    return df_train, df_val


def get_transformed_feature_names(preprocessor) -> np.ndarray:
    """Recover transformed feature names in the fitted ColumnTransformer."""
    try:
        return preprocessor.get_feature_names_out()
    except Exception:
        num_names = ["age", "bbox_volume"]
        
        cat_enc = preprocessor.named_transformers_["cat"]["onehot"]
        cat_names = cat_enc.get_feature_names_out(["tumor_subtype"])
        return np.concatenate([num_names, cat_names])


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute feature importance for baseline model.")
    ap.add_argument("--model", required=True, type=Path, help="Path to trained model.pkl")
    ap.add_argument("--json-dir", required=True, type=Path)
    ap.add_argument("--split-csv", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--n-repeats", type=int, default=20)
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # Load trained pipeline
    pipe = joblib.load(args.model)
    pre = pipe.named_steps["preprocess"]
    clf = pipe.named_steps["clf"]

    # Get transformed feature names & coefficients
    feat_names = get_transformed_feature_names(pre)
    coef = clf.coef_.ravel()
    if coef.shape[0] != feat_names.shape[0]:
        raise RuntimeError(f"n_coefs={coef.shape[0]} != n_features={feat_names.shape[0]}")

    # Load data (train/val only) and choose validation for permutation
    _, df_val = load_splits(args.json_dir, args.split_csv)
    df_perm = df_val.dropna(subset=["y"]).copy()
    split_used = "val"

    # Permutation importance on VAL; if no labeled val, skip and fill NaNs
    perm_scores = np.full_like(coef, fill_value=np.nan, dtype=float)
    X_raw = df_perm[FEATURES]
    y = df_perm["y"].astype(int).to_numpy()
    # compute in transformed space to align with coefficients
    X_trans = pre.transform(X_raw)
    perm = permutation_importance(
        estimator=clf,
        X=X_trans,
        y=y,
        scoring="roc_auc",
        n_repeats=args.n_repeats,
        random_state=args.random_state,
        n_jobs=-1,
    )
    perm_scores = perm.importances_mean

    # Build table, save CSV
    out_df = (
        pd.DataFrame(
            {
                "feature_name": feat_names,
                "coef": coef,
                "abs_coef": np.abs(coef),
                "permutation_importance": perm_scores,
            }
        )
        .sort_values("abs_coef", ascending=False)
        .reset_index(drop=True)
    )
    out_csv = args.output / "feature_importance.csv"
    out_df.to_csv(out_csv, index=False)

    # Plot top-k by |coef|
    top = out_df.head(args.top_k)
    plt.figure(figsize=(8, max(3, 0.35 * len(top))))
    plt.barh(top["feature_name"][::-1], top["abs_coef"][::-1])
    plt.xlabel("|Coefficient|")
    plt.title("Top features by absolute coefficient")
    plt.tight_layout()
    plt.savefig(args.output / "feature_importance.png", dpi=200)
    plt.close()

    print(
        f"Wrote {out_csv.name} and feature_importance.png to {args.output} "
        f"(permutation set = {split_used})"
    )


if __name__ == "__main__":
    main()