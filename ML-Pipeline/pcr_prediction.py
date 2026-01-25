"""MAMA-MIA data loader and baseline classifier (JSON-only).

This script extracts morphometric features from per-case JSON files,
joins them with variant labels from JSON or CSV, engineers normalized features,
and trains a baseline model (RandomForest or LogisticRegression).

Outputs:
- features.csv / features_engineered.csv
- labels_from_json.csv (if label JSONs given)
- model.pkl / metrics.json
"""

import yaml
import json
import logging
import math
import numbers
import shutil
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from feature_factor import get_clinical_features

ROC_FLIP_THRESHOLD = 0.5
DEFAULT_PROBA_THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(
        "JSON feature extraction + baseline classification (JSON morphometrics)"
    )
    ap.add_argument(
        "--feature-dir",
        type=Path,
        required=True,
        help="Directory of per-case MAMA-MIA Data feature JSONs (one <case_id>.json per case)",
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
    ap.add_argument(
        "--random-baseline",
        action="store_true",
        help="Run a random baseline on the same splits for comparison.",
    )
    ap.add_argument(
        "--bootstrap-n",
        type=int,
        default=0,
        help="If >0, bootstrap test metrics with N resamples to get 95% CIs.",
    )
    ap.add_argument(
        "--plots",
        action="store_true",
        help="Save ROC/PR curves and confusion matrix PNGs.",
    )
    ap.add_argument(
        "--save-intermediate-checks",
        action="store_true",
        help="Emit feature sanity ranges to CSV for pipeline verification.",
    )
    ap.add_argument(
        "--delong",
        action="store_true",
        help="Run DeLong test to compare model AUC vs random baseline.",
    )
    ap.add_argument(
        "--ensemble-runs",
        type=int,
        default=0,
        help="If >0, repeat train/val/test with different seeds to get AUC/AP distribution.",
    )
    ap.add_argument(
        "--ensemble-hist",
        action="store_true",
        help="Save histogram PNGs of ensemble AUC/AP if --ensemble-runs>0.",
    )
    ap.add_argument("--model", choices=["rf", "lr"], default="rf")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--val-size", type=float, default=0.1)
    ap.add_argument("--random-state", type=int, default=42)
    return ap.parse_args()


def build_modular_features(config):
    feature_dir = Path(config['data_paths']['feature_dir'])
    rows = []
    
    for p in sorted(feature_dir.glob("*.json")):
        with open(p, 'r') as f:
            data = json.load(f)
            
        case_id = data.get("patient_id")
        feats = {"case_id": case_id}
        
        if config['feature_toggles']['use_clinical']:
            c_data = data.get("clinical_data", {})
            feats["age"] = c_data.get("age")
            feats["pcr_label"] = data.get("primary_lesion", {}).get("pcr")
            feats["subtype"] = data.get("primary_lesion", {}).get("tumor_subtype")
            
        if config['feature_toggles']['use_vascular']:
            pass 
            
        rows.append(feats)
    return pd.DataFrame(rows)

def find_binary_label_columns(df_labels: pd.DataFrame) -> list[str]:
    """Identify columns in a DataFrame that contain only binary (0/1) values."""
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


def to_int_label(val: object) -> int:
    """Map fetal dict/bool/str/int to {0,1}. For dicts: 1 if ANY side True."""
    if isinstance(val, dict):
        return int(any(bool(v) for v in val.values()))
    if isinstance(val, bool | np.bool_):
        return int(val)
    s = str(val).strip().lower()
    if s in {"true", "yes", "1"}:
        return 1
    if s in {"false", "no", "0"}:
        return 0
    return int(val)


def load_labels(path: Path, id_col: str, label_col: str) -> pd.DataFrame:
    """Load labels from CSV or JSON and normalize to integer {0, 1}."""
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
    """Calculate standard binary classification metrics."""
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
    }


def engineer_features(in_csv: Path, out_csv: Path) -> None:
    """Clean and select relevant morphometric features from raw extraction CSV."""
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


def bootstrap_ci(
    y_true: np.ndarray | list,
    scores: np.ndarray | list,
    preds_threshold: float,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 123,
) -> dict[str, tuple[float, float]]:
    """Compute bootstrap confidence intervals for AUC, AP, and F1."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    n = len(y_true)
    aucs, aps, f1s = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        sc = scores[idx]
        pr = (sc >= preds_threshold).astype(int)
        try:
            aucs.append(roc_auc_score(yt, sc))
        except ValueError:
            continue
        aps.append(average_precision_score(yt, sc))
        f1s.append(f1_score(yt, pr, zero_division=0))

    def _ci(arr: list[float]) -> tuple[float, float]:
        arr_np = np.asarray(arr)
        lo = np.percentile(arr_np, 2.5)
        hi = np.percentile(arr_np, 97.5)
        return float(lo), float(hi)

    return {"auc_ci": _ci(aucs), "ap_ci": _ci(aps), "f1_ci": _ci(f1s)}


# ---- Minimal DeLong (binary) ----
def compute_midrank(x: np.ndarray) -> np.ndarray:
    """Compute midranks for DeLong test efficiency."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def fast_delong(
    y_true: np.ndarray | list, y_scores: np.ndarray | list
) -> tuple[float, float]:
    """Compute AUC and variance using the fast DeLong method."""
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    pos = y_scores[y_true == 1]
    neg = y_scores[y_true == 0]
    m, n = len(pos), len(neg)
    all_scores = np.hstack([pos, neg])
    tx = compute_midrank(pos)
    ty = compute_midrank(neg)
    tz = compute_midrank(all_scores)
    auc = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
    v01 = (tz[:m] - tx) / n
    v10 = 1 - (tz[m:] - ty) / m
    s01 = v01.var(ddof=1)
    s10 = v10.var(ddof=1)
    var = s01 / m + s10 / n
    return auc, var


def delong_test(
    y_true: np.ndarray | list,
    scores1: np.ndarray | list,
    scores2: np.ndarray | list,
) -> tuple[float, float]:
    """Compare two ROC curves on the same data using DeLong's test."""
    auc1, var1 = fast_delong(y_true, scores1)
    auc2, var2 = fast_delong(y_true, scores2)
    diff = auc1 - auc2
    se = np.sqrt(var1 + var2)
    if se == 0:
        return diff, float("nan")

    z = diff / se
    p = math.erfc(abs(z) / np.sqrt(2))
    return diff, float(p)


def choose_threshold(
    scores: np.ndarray,
    y_true: np.ndarray,
    *,
    beta: float = 0.5,  # < 1 => emphasize precision; > 1 => emphasize recall
    pos_rate_lo: float = 0.05,
    pos_rate_hi: float = 0.80,
    min_tn: int = 1,  # require at least 1 true negative on the tuning split
) -> tuple[float, dict]:
    """Sweep candidate thresholds and pick the one that maximizes F_beta.

    Uses weighted harmonic mean of precision and recall under simple sanity
    constraints so we don't end up predicting everything positive (or nothing).

    Returns (best_threshold, info_dict).
    """
    scores = np.asarray(scores).astype(float)
    y_true = np.asarray(y_true).astype(int)

    # Candidate thresholds: combine a grid with unique score knots for finer control
    grid = np.linspace(0.02, 0.98, 49)
    knots = np.unique(np.round(scores, 4))
    ths = np.unique(np.clip(np.concatenate([grid, knots]), 0.0, 1.0))

    best_t = 0.5
    best_val = -1.0
    best_stats = None

    # Avoid division warnings
    eps = 1e-12

    for t in ths:
        y_hat = (scores >= t).astype(int)
        pos_rate = y_hat.mean()

        # confusion counts
        tp = int(((y_hat == 1) & (y_true == 1)).sum())
        fp = int(((y_hat == 1) & (y_true == 0)).sum())
        tn = int(((y_hat == 0) & (y_true == 0)).sum())
        fn = int(((y_hat == 0) & (y_true == 1)).sum())

        # constraints to avoid degenerate thresholds
        if not (pos_rate_lo <= pos_rate <= pos_rate_hi):
            continue
        if tn < min_tn:
            continue

        prec = tp / (tp + fp + eps)
        rec = tp / (tp + fn + eps)

        # F_beta with stability
        fbeta = (1 + beta**2) * (prec * rec) / (beta**2 * prec + rec + eps)

        if fbeta > best_val:
            best_val = fbeta
            best_t = t
            best_stats = {
                "pos_rate": float(pos_rate),
                "precision": float(prec),
                "recall": float(rec),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "F_beta": float(fbeta),
                "beta": float(beta),
            }

    # Fallback: if constraints filtered everything, use median score
    if best_stats is None:
        best_t = float(np.median(scores))
        y_hat = (scores >= best_t).astype(int)

        best_stats = {
            "pos_rate": float(y_hat.mean()),
            "precision": float(precision_score(y_true, y_hat, zero_division=0)),
            "recall": float(recall_score(y_true, y_hat, zero_division=0)),
            "F_beta": float(fbeta_score(y_true, y_hat, beta=beta, zero_division=0)),
            "beta": float(beta),
        }

    return best_t, best_stats


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
    run_random_baseline: bool = False,
    bootstrap_n: int = 0,
    make_plots: bool = False,
    run_delong: bool = False,
) -> None:
    """Train the model, evaluate on test split, and optionally plot results."""
    # --- labels (directory of JSONs -> CSV
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
            try:
                rows.append({"case_id": case_id, label_col: to_int_label(val)})
            except Exception as e:
                logging.warning(
                    "Skipping label JSON %s missing/invalid '%s': %s", jp, label_col, e
                )
                continue
    else:
        labels_csv = labels_source

    # load & merge
    X = pd.read_csv(feats_engineered_csv)
    y_df = pd.read_csv(labels_csv)
    if id_col != "case_id" and id_col in y_df.columns:
        y_df = y_df.rename(columns={id_col: "case_id"})
    merged_df = X.merge(y_df, on="case_id", how="inner")

    before = len(merged_df)
    merged_df = merged_df[merged_df[label_col].notna()].copy()
    after = len(merged_df)
    print(
        f"[train_baseline] Dropped {before - after} rows with missing {label_col} (kept {after})"
    )

    y = merged_df[label_col].astype(int).to_numpy()
    drop_cols = ["case_id", label_col] + [
        c for c in merged_df.columns if c.endswith("_variant")
    ]
    Xmat = merged_df.drop(columns=drop_cols, errors="ignore").to_numpy()

    # classifier
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

    # 5-fold CV to get out-of-fold scores; auto-flip if inverted
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

    # holdout split (train/val/test)
    X_trval, X_te, y_trval, y_te = train_test_split(
        Xmat, y, test_size=test_size, random_state=random_state, stratify=y
    )
    rel_val = val_size / (1.0 - test_size)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_trval, y_trval, test_size=rel_val, random_state=random_state, stratify=y_trval
    )

    # fit
    clf.fit(X_tr, y_tr)

    def pos_scores(model: object, Xb: np.ndarray) -> np.ndarray:
        P = model.predict_proba(Xb)
        s = P[:, 1] if P.shape[1] > 1 else P.ravel()
        return 1.0 - s if scores_flipped else s

    # tuned threshold (precision-tilted + constraints)
    val_scores = pos_scores(clf, X_val)
    best_t, tinfo = choose_threshold(
        val_scores,
        y_val,
        beta=0.4,  # for better precision
        pos_rate_lo=0.05,
        pos_rate_hi=0.60,  # stricter upper bound
        min_tn=1,
    )

    print(f"[Threshold] chosen={best_t:.3f} | info={tinfo}")

    # predictions with tuned threshold
    y_val_hat = (val_scores >= best_t).astype(int)
    test_scores = pos_scores(clf, X_te)
    y_te_hat = (test_scores >= best_t).astype(int)

    # scalar metrics
    model_auc = float(roc_auc_score(y_te, test_scores))
    model_ap = float(average_precision_score(y_te, test_scores))

    # random baseline
    baseline = None
    rand_scores = None
    if run_random_baseline:
        rng = np.random.default_rng(random_state)
        rand_scores = rng.random(len(y_te))
        rand_preds = (rand_scores >= DEFAULT_PROBA_THRESHOLD).astype(int)
        baseline = {
            "roc_auc": float(roc_auc_score(y_te, rand_scores)),
            "ap": float(average_precision_score(y_te, rand_scores)),
            "precision@0.5": float(precision_score(y_te, rand_preds, zero_division=0)),
            "recall@0.5": float(recall_score(y_te, rand_preds, zero_division=0)),
            "f1@0.5": float(f1_score(y_te, rand_preds, zero_division=0)),
        }

    # bootstrap CIs at tuned threshold
    ci = None
    if bootstrap_n and bootstrap_n > 0:
        ci = bootstrap_ci(
            y_te,
            test_scores,
            preds_threshold=best_t,
            n_boot=bootstrap_n,
            alpha=0.05,
            seed=random_state,
        )

    # DeLong test vs random
    delong = None
    if run_delong:
        if rand_scores is None:
            rng = np.random.default_rng(random_state)
            rand_scores = rng.random(len(y_te))
        diff, p = delong_test(y_te, test_scores, rand_scores)
        delong = {"auc_diff": float(diff), "p_value": float(p)}

    # plots
    if make_plots:
        fpr_m, tpr_m, _ = roc_curve(y_te, test_scores)
        plt.figure()
        plt.plot(fpr_m, tpr_m, label=f"Model (AUC={model_auc:.2f})")
        if rand_scores is not None:
            fpr_r, tpr_r, _ = roc_curve(y_te, rand_scores)
            plt.plot(fpr_r, tpr_r, label="Random")
        plt.plot([0, 1], [0, 1], "--", lw=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "roc_comparison.png")
        plt.close()

        prec_m, rec_m, _ = precision_recall_curve(y_te, test_scores)
        plt.figure()
        plt.plot(rec_m, prec_m, label=f"Model (AP={model_ap:.2f})")
        if rand_scores is not None:
            prec_r, rec_r, _ = precision_recall_curve(y_te, rand_scores)
            plt.plot(rec_r, prec_r, label="Random")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "pr_comparison.png")
        plt.close()

        cm = confusion_matrix(y_te, y_te_hat)
        plot_confusion_matrix_clean(cm, outdir)
        print(
            "[ok] Plots: roc_comparison.png, pr_comparison.png, confusion_matrix_labeled.png"
        )

    print(f"[Split sizes] train={len(y_tr)} | val={len(y_val)} | test={len(y_te)}")
    print(
        f"[Threshold summary] best_t={best_t:.3f} | "
        f"val_pos_rate={y_val_hat.mean():.3f} | test_pos_rate={(test_scores>=best_t).mean():.3f}"
    )

    if model == "rf" and hasattr(clf, "feature_importances_"):
        X_cols = merged_df.drop(columns=drop_cols, errors="ignore").columns
        imp_df = pd.DataFrame(
            {"feature": X_cols, "importance": clf.feature_importances_}
        ).sort_values("importance", ascending=False)
        imp_path = outdir / "feature_importance.csv"
        imp_df.to_csv(imp_path, index=False)
        print(f"[ok] Feature importances -> {imp_path}")

        # Top 20 bar plot
        plt.figure(figsize=(6, 4))
        plt.barh(imp_df.head(20)["feature"][::-1], imp_df.head(20)["importance"][::-1])
        plt.xlabel("Importance")
        plt.title("Top 20 RF Feature Importances")
        plt.tight_layout()
        plt.savefig(outdir / "feature_importance_top20.png")
        plt.close()

    # results
    results = {
        "model": model,
        "scores_flipped": bool(scores_flipped),
        "best_threshold": float(best_t),
        "splits": {
            "train_n": int(len(y_tr)),
            "val_n": int(len(y_val)),
            "test_n": int(len(y_te)),
            "test_size": float(test_size),
            "val_size": float(val_size),
        },
        "cross_val": {
            "f1_mean": float(cv_f1),
            "precision_mean": float(cv_prec),
            "recall_mean": float(cv_rec),
            "roc_auc_mean": float(roc_cv),
            "ap_mean": float(ap_cv),
        },
        "val": metrics(y_val, y_val_hat),
        "test": {
            **metrics(y_te, (test_scores >= best_t).astype(int)),
            "roc_auc": model_auc,
            "ap": model_ap,
        },
    }
    if ci is not None:
        results["test"].update(ci)
    if baseline is not None:
        results["baseline_random"] = baseline
    if delong is not None:
        results["delong_vs_random"] = delong

    model_path = outdir / f"model_{model}.pkl"
    metrics_path = outdir / f"metrics_{model}.json"
    dump(clf, model_path)
    metrics_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"Model -> {model_path}")
    print(f"Metrics -> {metrics_path}")


def plot_confusion_matrix_clean(
    cm: np.ndarray, outdir: Path, title: str = "Confusion Matrix @ tuned threshold"
) -> None:
    """Clean, readable confusion matrix for binary classification.

    cm: 2x2 numpy array or list-of-lists in [ [TN, FP], [FN, TP] ] format
    outdir: directory to save the figure
    """
    fig, ax = plt.subplots(figsize=(5, 4))

    # Show matrix
    im = ax.imshow(cm, cmap="viridis")

    # Annotate counts in each cell
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, cm[i, j], ha="center", va="center", color="white", fontsize=12
            )

    # Set binary tick positions
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    # Set tick labels (TEXT, not numbers)
    ax.set_xticklabels(
        ["Predicted: not pCR", "Predicted: pCR"], rotation=20, ha="right"
    )
    ax.set_yticklabels(["True: not pCR", "True: pCR"])

    # Title
    ax.set_title(title)

    # Colorbar
    fig.colorbar(im, ax=ax)

    # Save and show
    plt.tight_layout()
    fig.savefig(f"{outdir}/confusion_matrix_clean.png", dpi=220)
    plt.close(fig)


def run_ensemble(
    feats_engineered_csv: Path,
    labels_source: Path,
    id_col: str,
    label_col: str,
    outdir: Path,
    base_random_state: int,
    model: str,
    test_size: float,
    val_size: float,
    runs: int,
    make_plots: bool,
) -> None:
    """Run model training multiple times with different seeds to estimate stability."""
    aucs, aps = [], []
    rng = np.random.default_rng(base_random_state)
    seeds = rng.integers(0, 2**31 - 1, size=runs).tolist()
    for i, seed in enumerate(seeds, 1):
        print(f"[Ensemble] run {i}/{runs} (seed={seed})")
        X = pd.read_csv(feats_engineered_csv)
        y_df = pd.read_csv(
            labels_source
            if not labels_source.is_dir()
            else outdir / "labels_from_json.csv"
        )
        if id_col != "case_id" and id_col in y_df.columns:
            y_df = y_df.rename(columns={id_col: "case_id"})
        merged = X.merge(y_df, on="case_id", how="inner")
        y = merged[label_col].astype(int).to_numpy()
        drop_cols = ["case_id", label_col] + [
            c for c in merged.columns if c.endswith("_variant")
        ]
        Xmat = merged.drop(columns=drop_cols, errors="ignore").to_numpy()

        if model == "rf":
            clf = RandomForestClassifier(
                n_estimators=800,
                max_depth=None,
                min_samples_leaf=1,
                min_samples_split=2,
                max_features="sqrt",
                class_weight="balanced",
                n_jobs=-1,
                random_state=seed,
            )
        else:
            base_model = LogisticRegression(
                class_weight="balanced",
                solver="liblinear",
                max_iter=2000,
                random_state=seed,
            )
            clf = Pipeline([("scaler", StandardScaler()), ("model", base_model)])

        X_trval, X_te, y_trval, y_te = train_test_split(
            Xmat, y, test_size=test_size, random_state=seed, stratify=y
        )
        rel_val = val_size / (1.0 - test_size)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_trval, y_trval, test_size=rel_val, random_state=seed, stratify=y_trval
        )
        clf.fit(X_tr, y_tr)

        P_te = clf.predict_proba(X_te)[:, 1]
        aucs.append(float(roc_auc_score(y_te, P_te)))
        aps.append(float(average_precision_score(y_te, P_te)))

    dist = {
        "runs": int(runs),
        "auc": {
            "mean": float(np.mean(aucs)),
            "std": float(np.std(aucs)),
            "min": float(np.min(aucs)),
            "p25": float(np.percentile(aucs, 25)),
            "median": float(np.median(aucs)),
            "p75": float(np.percentile(aucs, 75)),
            "max": float(np.max(aucs)),
            "values": aucs,
        },
        "ap": {
            "mean": float(np.mean(aps)),
            "std": float(np.std(aps)),
            "min": float(np.min(aps)),
            "p25": float(np.percentile(aps, 25)),
            "median": float(np.median(aps)),
            "p75": float(np.percentile(aps, 75)),
            "max": float(np.max(aps)),
            "values": aps,
        },
    }
    (outdir / "ensemble_metrics.json").write_text(json.dumps(dist, indent=2))

    print(
        f"[Ensemble Summary] AUC mean={np.mean(aucs):.3f} ± {np.std(aucs):.3f}, "
        f"AP mean={np.mean(aps):.3f} ± {np.std(aps):.3f}"
    )

    if make_plots:
        plt.figure()
        plt.hist(aucs, bins=12)
        plt.xlabel("Test ROC-AUC")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(outdir / "ensemble_auc_hist.png")
        plt.close()
        plt.figure()
        plt.hist(aps, bins=12)
        plt.xlabel("Test AP")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(outdir / "ensemble_ap_hist.png")
        plt.close()

    print("[ok] Ensemble -> ensemble_metrics.json")

def build_modular_features(config):
    """Refactored loader for model framework and clinical features."""
    feature_dir = Path(config['data_paths']['feature_dir'])
    rows = []
    
    for p in sorted(feature_dir.glob("*.json")):
        with open(p, 'r') as f:
            data = json.load(f)
            
        case_id = data.get("patient_id")
        feats = {"case_id": case_id}
        
        # Clinical Extraction Logic
        if config['feature_toggles']['use_clinical']:
            c_data = data.get("clinical_data", {})
            feats["age"] = c_data.get("age")
            feats["pcr_label"] = data.get("primary_lesion", {}).get("pcr")
            feats["subtype"] = data.get("primary_lesion", {}).get("tumor_subtype")
            
        # Vascular Extraction
        if config['feature_toggles']['use_vascular']:
            pass 
            
        rows.append(feats)
    return pd.DataFrame(rows)

def main() -> None:
    """For args, config, output dir, feature building, training."""
    config_path = "ML-Pipeline/config_pcr.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Output Directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(config['experiment_setup']['base_outdir']) / f"{config['experiment_setup']['name']}_{timestamp}"
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Save copy of config
    shutil.copy(config_path, outdir / "config_used.yaml")

    # Feature Building
    feats = build_modular_features(config)

    # Integrate Clinical Features from Excel/CSV
    if config['feature_toggles']['use_clinical']:
        clinical_df = get_clinical_features(config)
        feats = feats.merge(clinical_data, left_on='case_id', right_on='patient_id', how='inner')
        logging.info(f"Integrated clinical features. New shape: {feats.shape}")

    feats_path = outdir / "features.csv"
    feats.to_csv(feats_path, index=False)
    
    eng_path = outdir / "features_engineered.csv"
    engineer_features(feats_path, eng_path)

    train_baseline(
        feats_engineered_csv=eng_path,
        labels_source=Path(config['data_paths']['labels_csv']),
        id_col=config['data_paths']['id_column'],
        label_col=config['data_paths']['label_column'],
        outdir=outdir,
        test_size=config['model_params']['test_size'],
        random_state=config['model_params']['random_state'],
        model=config['model_params']['model'],
        val_size=config['model_params']['val_size'],
        bootstrap_n=config['model_params']['bootstrap_n']
    )

if __name__ == "__main__":
    main()