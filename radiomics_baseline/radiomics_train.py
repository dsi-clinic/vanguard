#!/usr/bin/env python3
"""Train a classifier on already-extracted radiomics features.

Stage 2: Train a classifier on already-extracted radiomics features.

Inputs
------
- --train-features : CSV from radiomics_extract.py (rows = patients, cols = features)
- --test-features  : CSV from radiomics_extract.py
- --labels         : CSV with at least columns: patient_id,pcr[,subtype]
- --output         : output directory to write metrics, plots, and model

What this script does
---------------------
1) Load train/test feature CSVs and align with labels.
2) (Optional) Append a numeric subtype code as an extra feature.
3) Sanitize to numeric-only; drop all-NaN and zero-variance columns.
4) (Optional) Correlation prune (train-only), then align test columns.
5) (Optional) SelectKBest K features (train-only), then align test columns.
6) Train the chosen classifier (logistic or RF), optionally via GridSearchCV.
7) Save metrics.json, predictions.csv, ROC/PR/Calibration plots, and model.pkl.

Example Usage
---------------------
python radiomics_train.py \
  --train-features experiments/extract_peri5_multiphase/features_train_final.csv \
  --test-features  experiments/extract_peri5_multiphase/features_test_final.csv \
  --labels         labels.csv \
  --output         outputs/elasticnet_corr0.9_k50_cv5 \
  --classifier     logistic \
  --logreg-penalty elasticnet \
  --logreg-l1-ratio 0.5 \
  --corr-threshold 0.9 \
  --k-best         50 \
  --grid-search \
  --cv-folds       5 \
  --include-subtype
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MIN_CLASS_COUNT = 2
CONF_MATRIX_SIZE = 4
AUC_BASELINE = 0.5


# ---------------------------
# I/O helpers
# ---------------------------
def load_features(path: str) -> pd.DataFrame:
    """Load feature CSV with patient IDs in the index."""
    return pd.read_csv(path, index_col=0)


def load_labels(labels_csv: str) -> pd.DataFrame:
    """Load labels CSV and ensure it contains a 'pcr' column."""
    lab = pd.read_csv(labels_csv).set_index("patient_id")
    if "pcr" not in lab.columns:
        msg = "labels.csv must contain column 'pcr'"
        raise ValueError(msg)
    return lab


# ---------------------------
# Sanitization & selection
# ---------------------------
def sanitize_numeric(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    """Keep only numeric columns, dropping all-NaN and zero-variance features."""
    raw = df.shape
    num = df.select_dtypes(include=[np.number]).copy()
    # drop all-NaN
    all_nan = num.columns[num.isna().all()].tolist()
    num = num.drop(columns=all_nan, errors="ignore")
    # drop zero-var
    nunique = num.nunique(dropna=True)
    zero_var = nunique[nunique <= 1].index.tolist()
    num = num.drop(columns=zero_var, errors="ignore")
    print(
        f"[DEBUG] {tag}: raw={raw} -> numeric={num.shape} "
        f"(all-NaN={len(all_nan)}, zero-var={len(zero_var)})",
    )
    return num


def corr_prune(X: pd.DataFrame, thr: float) -> tuple[pd.DataFrame, list[str]]:
    """Drop one of any pair of features with |rho| >= thr (train-only)."""
    if thr <= 0 or thr >= 1 or X.shape[1] < MIN_CLASS_COUNT:
        return X, []
    corr = X.corr(method="pearson").abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [c for c in upper.columns if (upper[c] >= thr).any()]
    X_kept = X.drop(columns=drop_cols, errors="ignore")
    print(
        f"[DEBUG] corr-prune @ {thr:.2f}: "
        f"dropped={len(drop_cols)}, kept={X_kept.shape[1]}",
    )
    return X_kept, drop_cols


def apply_kbest(
    Xtr: pd.DataFrame,
    ytr: np.ndarray,
    Xte: pd.DataFrame,
    k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select top-k features on TRAIN only using ANOVA F-test, handling NaNs via median imputation."""
    if not k or k >= Xtr.shape[1]:
        return Xtr, Xte

    # For feature scoring, fill NaNs in TRAIN with column medians
    Xtr_for_kbest = Xtr.copy()
    if Xtr_for_kbest.isna().to_numpy().any():
        col_medians = Xtr_for_kbest.median(axis=0)
        Xtr_for_kbest = Xtr_for_kbest.fillna(col_medians)
        print(
            "[DEBUG] apply_kbest: filled NaNs in Xtr with column "
            "medians for k-best selection.",
        )

    sel = SelectKBest(score_func=f_classif, k=k)
    sel.fit(Xtr_for_kbest, ytr)

    keep_cols = Xtr.columns[sel.get_support()]
    Xtr_k = Xtr[keep_cols]
    Xte_k = Xte.reindex(columns=keep_cols, fill_value=np.nan)
    print(f"[DEBUG] k-best={k}: Xtr -> {Xtr_k.shape}, Xte -> {Xte_k.shape}")
    return Xtr_k, Xte_k


# ---------------------------
# Plotting
# ---------------------------
def plot_roc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    outpath: Path,
) -> None:
    """Plot ROC curve and save to disk."""
    if len(np.unique(y_true)) < MIN_CLASS_COUNT:
        return
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.legend()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_pr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    outpath: Path,
) -> None:
    """Plot precision-recall curve and save to disk."""
    if len(np.unique(y_true)) < MIN_CLASS_COUNT:
        return
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_calib(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    outpath: Path,
) -> None:
    """Plot calibration curve and save to disk."""
    if len(np.unique(y_true)) < MIN_CLASS_COUNT:
        return
    prob_true, prob_pred = calibration_curve(
        y_true,
        y_prob,
        n_bins=min(10, len(y_true)),
        strategy="quantile",
    )
    plt.figure()
    plt.plot(prob_pred, prob_true, "o-")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("Predicted")
    plt.ylabel("Observed")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# ---------------------------
# Model builders
# ---------------------------
def normalize_rf_max_features(
    val: int | float | str | None,
) -> int | float | str | None:
    """Parse --rf-max-features into a valid value."""
    if val is None:
        return None
    if isinstance(val, int | float):
        return val
    if val in ("sqrt", "log2"):
        return val
    if isinstance(val, str):
        try:
            return int(val)
        except (TypeError, ValueError):
            try:
                return float(val)
            except (TypeError, ValueError):
                return None
    return None


def build_estimator(
    args: argparse.Namespace,
) -> RandomForestClassifier | LogisticRegression | GridSearchCV:
    """Return a classifier or a GridSearchCV wrapping the classifier."""
    if args.classifier == "logistic":
        solver = "saga" if args.logreg_penalty in ("l1", "elasticnet") else "lbfgs"
        base = LogisticRegression(
            penalty=args.logreg_penalty,
            l1_ratio=(
                args.logreg_l1_ratio if args.logreg_penalty == "elasticnet" else None
            ),
            C=args.logreg_C,
            solver=solver,
            max_iter=4000,
            class_weight="balanced",
            random_state=42,
        )
        if args.grid_search:
            param_grid: dict[str, list[float]] = {"C": [0.05, 0.1, 0.2, 0.5, 1.0]}
            if args.logreg_penalty == "elasticnet":
                param_grid["l1_ratio"] = [0.1, 0.3, 0.5, 0.7]
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            return GridSearchCV(
                base,
                param_grid,
                scoring="roc_auc",
                cv=cv,
                n_jobs=-1,
                verbose=1,
            )
        return base

    # Random Forest
    base = RandomForestClassifier(
        n_estimators=args.rf_n_estimators,
        max_depth=args.rf_max_depth,
        min_samples_leaf=args.rf_min_samples_leaf,
        min_samples_split=args.rf_min_samples_split,
        max_features=normalize_rf_max_features(args.rf_max_features),
        ccp_alpha=args.rf_ccp_alpha,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )
    if args.grid_search:
        param_grid_rf: dict[str, list[float | int | str]] = {
            "n_estimators": [300, 400, 500],
            "max_depth": [6, 8, 10],
            "min_samples_leaf": [5, 10, 20],
            "max_features": [0.2, 0.3, 0.5, "sqrt"],
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        return GridSearchCV(
            base,
            param_grid_rf,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            verbose=1,
        )
    return base


# ---------------------------
# Main
# ---------------------------
def main() -> None:
    """Parse arguments, train the model, and save predictions/metrics."""
    ap = argparse.ArgumentParser(
        description="Train classifier on already-extracted radiomics CSVs.",
    )
    ap.add_argument("--train-features", required=True)
    ap.add_argument("--test-features", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--output", required=True)

    ap.add_argument("--classifier", choices=["logistic", "rf"], default="logistic")

    # Logistic options
    ap.add_argument("--logreg-C", type=float, default=1.0)
    ap.add_argument(
        "--logreg-penalty",
        choices=["l2", "l1", "elasticnet"],
        default="l2",
    )
    ap.add_argument("--logreg-l1-ratio", type=float, default=0.0)

    # RF options
    ap.add_argument("--rf-n-estimators", type=int, default=300)
    ap.add_argument("--rf-max-depth", type=int, default=None)
    ap.add_argument("--rf-min-samples-leaf", type=int, default=1)
    ap.add_argument("--rf-min-samples-split", type=int, default=2)
    ap.add_argument("--rf-max-features", default="sqrt")
    ap.add_argument("--rf-ccp-alpha", type=float, default=0.0)

    # Feature handling
    ap.add_argument(
        "--include-subtype",
        action="store_true",
        help=(
            "Append labels subtype as numeric feature "
            "(column 'subtype' or 'tumor_subtype')."
        ),
    )
    ap.add_argument(
        "--corr-threshold",
        type=float,
        default=0.0,
        help="If >0, drop one of any pair with |rho| >= threshold on TRAIN only.",
    )
    ap.add_argument(
        "--k-best",
        type=int,
        default=0,
        help="If >0, keep top-K features by ANOVA F-test on TRAIN only.",
    )
    ap.add_argument(
        "--grid-search",
        action="store_true",
        help="If set, run a small GridSearchCV on the classifier using train only.",
    )
    ap.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        help=(
            "If > 1, run Stratified K-fold CV on the training set and "
            "save cross-validated metrics."
        ),
    )

    args = ap.parse_args()
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------------- Load data ----------------
    Xtr_raw = load_features(args.train_features)
    Xte_raw = load_features(args.test_features)
    print(f"[DEBUG] Xtr_raw shape: {Xtr_raw.shape}, Xte_raw shape: {Xte_raw.shape}")

    labels = load_labels(args.labels)

    # Make sure indices line up and y are extracted
    ytr = labels.loc[Xtr_raw.index, "pcr"].astype(int).to_numpy()
    yte = labels.loc[Xte_raw.index, "pcr"].astype(int).to_numpy()

    # Identify subtype column if present
    subtype_col: str | None = None
    for candidate in ("subtype", "tumor_subtype"):
        if candidate in labels.columns:
            subtype_col = candidate
            break

    # Optional: subtype as a numeric feature
    if args.include_subtype:
        if subtype_col is None:
            print(
                "[WARN] --include-subtype set but no 'subtype' or 'tumor_subtype' "
                "in labels; skipping.",
            )
        else:
            cats = labels[subtype_col].dropna().unique()
            # stable mapping: sort for reproducibility
            code_map = {cat: i for i, cat in enumerate(sorted(cats))}
            Xtr_raw = Xtr_raw.assign(
                subtype_code=labels.loc[Xtr_raw.index, subtype_col].map(code_map),
            )
            Xte_raw = Xte_raw.assign(
                subtype_code=labels.loc[Xte_raw.index, subtype_col].map(code_map),
            )
            print(f"[DEBUG] appended subtype column '{subtype_col}' → 'subtype_code'")

    # ---------------- Sanitize ----------------
    Xtr = sanitize_numeric(Xtr_raw, "train")
    Xte = sanitize_numeric(Xte_raw, "test")
    # column-align test to train
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=np.nan)

    # ---------------- Corr prune (train only) ----------------
    Xtr, _dropped = corr_prune(Xtr, args.corr_threshold)
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=np.nan)

    # ---------------- K-best (train only) ----------------
    Xtr, Xte = apply_kbest(Xtr, ytr, Xte, args.k_best)

    print(f"[DEBUG] final train/test shapes: {Xtr.shape} / {Xte.shape}")

    # ---------------- Build pipeline ----------------
    steps: list[tuple[str, object]] = [("impute", SimpleImputer(strategy="median"))]
    if args.classifier == "logistic":
        steps.append(("scale", StandardScaler()))
    steps.append(("clf", build_estimator(args)))
    pipe = Pipeline(steps)

    # ---------------- Train ----------------
    pipe.fit(Xtr, ytr)
    clf_step = pipe.named_steps["clf"]

    # predict_proba and classes_ should be available (GridSearchCV delegates)
    classes_ = clf_step.classes_
    pos_idx = int(np.where(classes_ == 1)[0][0])

    p_tr = pipe.predict_proba(Xtr)[:, pos_idx]
    p_te = pipe.predict_proba(Xte)[:, pos_idx]

    auc_tr = (
        float(roc_auc_score(ytr, p_tr))
        if len(np.unique(ytr)) > MIN_CLASS_COUNT - 1
        else float("nan")
    )
    auc_te = (
        float(roc_auc_score(yte, p_te))
        if len(np.unique(yte)) > MIN_CLASS_COUNT - 1
        else float("nan")
    )

    fpr, tpr, thr = roc_curve(ytr, p_tr)
    thr_opt = float(thr[np.argmax(tpr - fpr)]) if len(thr) else AUC_BASELINE
    ypred_te = (p_te >= thr_opt).astype(int)

    cm = confusion_matrix(yte, ypred_te, labels=[0, 1])
    if cm.size == CONF_MATRIX_SIZE:
        tn, fp, fn, tp = cm.ravel()
        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    else:
        tn = fp = fn = tp = sens = spec = None

    # ---------------- Optional K-fold CV on training set ----------------
    auc_cv = float("nan")
    auc_cv_by_subtype: dict[str, float] | None = None
    if args.cv_folds and args.cv_folds > 1:
        cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
        # cross-validated train predictions for the positive class (label=1)
        p_cv = cross_val_predict(
            pipe,
            Xtr,
            ytr,
            cv=cv,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]
        auc_cv = (
            float(roc_auc_score(ytr, p_cv))
            if len(np.unique(ytr)) > MIN_CLASS_COUNT - 1
            else float("nan")
        )

        # per-subtype CV AUC, if subtype labels exist
        if subtype_col is not None:
            sub_tr = labels.loc[Xtr.index, subtype_col]
            auc_cv_by_subtype = {}
            for sub_val in sorted(sub_tr.dropna().unique()):
                mask = sub_tr == sub_val
                mask_array = mask.to_numpy()
                y_true_sub = ytr[mask_array]
                y_prob_sub = p_cv[mask_array]
                if len(np.unique(y_true_sub)) < MIN_CLASS_COUNT:
                    auc_sub = float("nan")
                else:
                    auc_sub = float(roc_auc_score(y_true_sub, y_prob_sub))
                auc_cv_by_subtype[str(sub_val)] = auc_sub

        # save CV predictions
        pd.DataFrame(
            {
                "patient_id": Xtr.index,
                "y_true": ytr,
                "y_prob_cv": p_cv,
            },
        ).to_csv(outdir / "predictions_cv_train.csv", index=False)

    # ---------------- AUC by subtype on test set ----------------
    auc_test_by_subtype: dict[str, float] | None = None
    if subtype_col is not None:
        sub_te = labels.loc[Xte.index, subtype_col]
        auc_test_by_subtype = {}
        for sub_val in sorted(sub_te.dropna().unique()):
            mask = sub_te == sub_val
            mask_array = mask.to_numpy()
            y_true_sub = yte[mask_array]
            y_prob_sub = p_te[mask_array]
            if len(np.unique(y_true_sub)) < MIN_CLASS_COUNT:
                auc_sub_test = float("nan")
            else:
                auc_sub_test = float(roc_auc_score(y_true_sub, y_prob_sub))
            auc_test_by_subtype[str(sub_val)] = auc_sub_test

    # ---------------- Save outputs ----------------
    # predictions
    pd.DataFrame(
        {
            "patient_id": Xte.index,
            "y_true": yte,
            "y_prob": p_te,
            "y_pred": ypred_te,
        },
    ).to_csv(outdir / "predictions.csv", index=False)

    # plots
    plot_roc(yte, p_te, outdir / "roc_test.png")
    plot_pr(yte, p_te, outdir / "pr_curve.png")
    try:
        plot_calib(yte, p_te, outdir / "calibration_curve.png")
        calib_status = "ok"
    except Exception:  # noqa: BLE001
        calib_status = "none"

    # metrics
    metrics: dict[str, object] = {
        "auc_train": auc_tr,
        "auc_test": auc_te,
        "auc_train_cv": auc_cv,
        "auc_train_cv_by_subtype": auc_cv_by_subtype,
        "auc_test_by_subtype": auc_test_by_subtype,
        "n_features_used": int(Xtr.shape[1]),
        "classifier_type": (
            "logistic" if args.classifier == "logistic" else "random_forest"
        ),
        "class_order": classes_.tolist(),
        "threshold_train_youdenJ": thr_opt,
        "sensitivity_test": sens,
        "specificity_test": spec,
        "tn_fp_fn_tp_test": [
            int(x) if x is not None else None for x in [tn, fp, fn, tp]
        ],
        "calibration": calib_status,
        "corr_threshold": args.corr_threshold,
        "k_best": int(args.k_best),
        "grid_search": bool(args.grid_search),
        "cv_folds": int(args.cv_folds),
    }

    if not (auc_te > AUC_BASELINE):
        metrics["commentary"] = (
            "AUC_test ≤ 0.5. Try stronger regularization (Elastic-Net), "
            "correlation pruning, K-best, or DCE kinetic deltas."
        )

    metrics_path = outdir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump(pipe, outdir / "model.pkl")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
    