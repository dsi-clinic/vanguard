#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
radiomics_train.py

Stage 2 of the pipeline: take already-extracted radiomics features (CSV)
and train a classifier.

Inputs:
- features_train.csv (from radiomics_extract.py)
- features_test.csv  (from radiomics_extract.py)
- labels.csv         (global labels file with patient_id,pcr[,subtype])

What it does:
1. read train/test feature CSVs
2. read labels.csv, align labels to the feature indices
3. optionally append a numeric subtype code (if --include-subtype and labels.csv has 'subtype')
4. sanitize to numeric-only, drop all-NaN and zero-variance columns
5. train the chosen classifier (logistic or RF)
6. write metrics.json, predictions.csv, and diagnostic plots
"""

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import joblib


def load_features(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    return df


def sanitize_numeric(df: pd.DataFrame, tag: str):
    raw = df.shape
    num = df.select_dtypes(include=[np.number]).copy()
    # drop all-NaN
    all_nan = num.columns[num.isna().all()].tolist()
    num = num.drop(columns=all_nan, errors="ignore")
    # drop zero-var
    nunique = num.nunique(dropna=True)
    zero_var = nunique[nunique <= 1].index.tolist()
    num = num.drop(columns=zero_var, errors="ignore")
    print(f"[DEBUG] {tag}: raw={raw} -> numeric={num.shape} (all-NaN={len(all_nan)}, zero-var={len(zero_var)})")
    return num


def plot_roc(y_true, y_prob, outpath: Path):
    if len(np.unique(y_true)) < 2:
        return
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_pr(y_true, y_prob, outpath: Path):
    if len(np.unique(y_true)) < 2:
        return
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(rec, prec)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_calib(y_true, y_prob, outpath: Path):
    if len(np.unique(y_true)) < 2:
        return
    prob_true, prob_pred = calibration_curve(y_true, y_prob,
                                             n_bins=min(10, len(y_true)),
                                             strategy="quantile")
    plt.figure()
    plt.plot(prob_pred, prob_true, "o-")
    plt.plot([0, 1], [0, 1], "--")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def normalize_rf_max_features(val):
    """
    argparse gives us a string sometimes; make it acceptable to RF.
    """
    if val is None:
        return None
    # already one of the allowed strings
    if val in ("sqrt", "log2"):
        return val
    # try int
    try:
        iv = int(val)
        return iv
    except Exception:
        pass
    # try float
    try:
        fv = float(val)
        return fv
    except Exception:
        pass
    # last resort
    return None


def main():
    ap = argparse.ArgumentParser(description="Train classifier on already-extracted radiomics CSVs.")
    ap.add_argument("--train-features", required=True)
    ap.add_argument("--test-features", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--output", required=True)

    ap.add_argument("--classifier", choices=["logistic", "rf"], default="logistic")

    # logistic
    ap.add_argument("--logreg-C", type=float, default=1.0)

    # RF
    ap.add_argument("--rf-n-estimators", type=int, default=300)
    ap.add_argument("--rf-max-depth", type=int, default=None)
    ap.add_argument("--rf-min-samples-leaf", type=int, default=1)
    ap.add_argument("--rf-min-samples-split", type=int, default=2)
    ap.add_argument("--rf-max-features", default="sqrt")
    ap.add_argument("--rf-ccp-alpha", type=float, default=0.0)

    ap.add_argument("--include-subtype", action="store_true",
                    help="append labels.csv:tumor_subtype as numeric feature")
    args = ap.parse_args()

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # load
    # ------------------------------------------------------------------
    Xtr_raw = load_features(args.train_features)
    Xte_raw = load_features(args.test_features)
    print(f"[DEBUG] Xtr_raw shape: {Xtr_raw.shape}, Xte_raw shape: {Xte_raw.shape}")

    labels = pd.read_csv(args.labels)
    labels = labels.set_index("patient_id")

    # y
    ytr = labels.loc[Xtr_raw.index, "pcr"].astype(int).values
    yte = labels.loc[Xte_raw.index, "pcr"].astype(int).values

    # subtype (optional)
    if args.include_subtype:
        if "subtype" in labels.columns:
            subtype_map = {v: i for i, v in enumerate(sorted(labels["subtype"].dropna().unique()))}
            tr_sub = labels.loc[Xtr_raw.index, "subtype"].map(subtype_map)
            te_sub = labels.loc[Xte_raw.index, "subtype"].map(subtype_map)
            Xtr_raw = Xtr_raw.assign(subtype_code=tr_sub)
            Xte_raw = Xte_raw.assign(subtype_code=te_sub)
            print("[DEBUG] appended subtype column 'subtype' → 'subtype_code'")
        else:
            print("[WARN] --include-subtype set but 'subtype' not in labels; skipping.")

    # ------------------------------------------------------------------
    # sanitize numeric
    # ------------------------------------------------------------------
    Xtr = sanitize_numeric(Xtr_raw, "train")
    Xte = sanitize_numeric(Xte_raw, "test")
    # align
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=np.nan)

    print(f"[DEBUG] Xtr shape after sanitize: {Xtr.shape}")
    print(f"[DEBUG] Xte shape after sanitize: {Xte.shape}")

    # OPTIONAL variance filter
    # TOP_K = 200
    # if Xtr.shape[1] > TOP_K:
    #     var = Xtr.var(axis=0).sort_values(ascending=False)
    #     keep = var.index[:TOP_K]
    #     Xtr = Xtr[keep]
    #     Xte = Xte[keep]
    #     print(f"[DEBUG] variance filter: kept {TOP_K} / {len(var)}")

    # ------------------------------------------------------------------
    # model
    # ------------------------------------------------------------------
    if args.classifier == "logistic":
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l2",
                C=args.logreg_C,
                max_iter=2000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=42,
            ))
        ])
        clf_type = "logistic"
    else:
        rf_max_feat = normalize_rf_max_features(args.rf_max_features)
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=args.rf_n_estimators,
                max_depth=args.rf_max_depth,
                min_samples_leaf=args.rf_min_samples_leaf,
                min_samples_split=args.rf_min_samples_split,
                max_features=rf_max_feat,
                ccp_alpha=args.rf_ccp_alpha,
                n_jobs=-1,
                class_weight="balanced",
                random_state=42,
            ))
        ])
        clf_type = "random_forest"

    pipe.fit(Xtr, ytr)

    clf_step = pipe.named_steps["clf"]
    pos_idx = int(np.where(clf_step.classes_ == 1)[0][0])

    p_tr = pipe.predict_proba(Xtr)[:, pos_idx]
    p_te = pipe.predict_proba(Xte)[:, pos_idx]

    auc_tr = float(roc_auc_score(ytr, p_tr)) if len(np.unique(ytr)) > 1 else float("nan")
    auc_te = float(roc_auc_score(yte, p_te)) if len(np.unique(yte)) > 1 else float("nan")

    fpr, tpr, thr = roc_curve(ytr, p_tr)
    thr_opt = float(thr[np.argmax(tpr - fpr)]) if len(thr) else 0.5
    ypred_te = (p_te >= thr_opt).astype(int)

    cm = confusion_matrix(yte, ypred_te, labels=[0, 1])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    else:
        tn = fp = fn = tp = sens = spec = None

    # save preds
    pd.DataFrame({
        "patient_id": Xte.index,
        "y_true": yte,
        "y_prob": p_te,
        "y_pred": ypred_te,
    }).to_csv(outdir / "predictions.csv", index=False)

    # plots
    plot_roc(yte, p_te, outdir / "roc_test.png")
    plot_pr(yte, p_te, outdir / "pr_curve.png")
    try:
        plot_calib(yte, p_te, outdir / "calibration_curve.png")
        calib_status = "ok"
    except Exception:
        calib_status = "none"

    metrics = {
        "auc_train": auc_tr,
        "auc_test": auc_te,
        "n_features_used": int(Xtr.shape[1]),
        "classifier_type": clf_type,
        "class_order": clf_step.classes_.tolist(),
        "threshold_train_youdenJ": thr_opt,
        "sensitivity_test": sens,
        "specificity_test": spec,
        "tn_fp_fn_tp_test": [
            int(x) if x is not None else None for x in [tn, fp, fn, tp]
        ],
        "calibration": calib_status,
    }
    if not (auc_te > 0.5):
        metrics["commentary"] = (
            "AUC_test ≤ 0.5. Consider different thresholding, "
            "stronger RF regularization, or feature filtering."
        )

    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump(pipe, outdir / "model.pkl")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
