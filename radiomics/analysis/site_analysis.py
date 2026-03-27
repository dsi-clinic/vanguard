#!/usr/bin/env python3
"""Site-level A/B analysis for radiomics experiments.

Two complementary analyses:

1. **Per-site evaluation** — Train on the existing mixed train split,
   evaluate on the test split, then break down metrics by clinical site.
2. **Leave-one-site-out (LOSO)** — For each site, train on all other sites
   and test on the held-out site to measure cross-site generalisation.

The script takes pre-extracted feature CSVs (output of ``radiomics_extract.py``)
so it can be reused with any extraction configuration.

Usage
-----
    python site_analysis.py \
        --features-dir outputs/shared_extraction \
        --labels labels.csv \
        --splits splits_train_test_ready.csv \
        --output outputs/site_analysis \
        --classifier logistic \
        --corr-threshold 0.9 \
        --k-best 50 \
        --grid-search
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Reuse helpers from radiomics_train (imported at function level so the
# module can also be tested in isolation with mocks if needed).
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

from radiomics_train import (  # noqa: E402
    CorrelationPruner,
    MRMRSelector,
    align_numeric_to_reference,
    append_categorical_feature,
    build_estimator,
    load_features,
    load_labels,
    sanitize_numeric,
)

MIN_CLASS_COUNT = 2
CONF_MATRIX_SIZE = 4
DUPLICATE_PREVIEW_LIMIT = 5


# ---------------------------------------------------------------------------
# Site helpers
# ---------------------------------------------------------------------------


def extract_site(pid: str) -> str:
    """Extract clinical site prefix from a patient ID.

    Examples: ``DUKE_001`` → ``DUKE``, ``ISPY2_0042`` → ``ISPY2``.
    """
    m = re.match(r"^([A-Za-z]+\d*)", pid)
    return m.group(1) if m else "UNKNOWN"


def assign_sites(index: pd.Index) -> pd.Series:
    """Return a Series mapping each patient ID to its site."""
    return pd.Series(
        [extract_site(str(pid)) for pid in index], index=index, name="site"
    )


def assign_sites_from_labels(labels: pd.DataFrame, index: pd.Index) -> pd.Series:
    """Return site labels aligned to *index*, preferring labels['site'] when present."""
    if "site" in labels.columns:
        sites = labels.loc[index, "site"].astype(str).copy()
        missing_mask = (
            sites.isna() | (sites.str.len() == 0) | (sites.str.lower() == "nan")
        )
        if missing_mask.any():
            fallback = assign_sites(index[missing_mask.to_numpy()])
            sites.loc[missing_mask] = fallback
        sites.name = "site"
        return sites
    return assign_sites(index)


# ---------------------------------------------------------------------------
# Train / evaluate one split
# ---------------------------------------------------------------------------


def _build_pipeline(args: argparse.Namespace) -> Pipeline:
    """Build the sklearn Pipeline.

    (imputer -> [corr_prune] -> [kbest/mrmr] -> [scale] -> classifier).

    Feature selection steps are included inside the pipeline so they are
    re-fitted on only the training rows of each LOSO fold, matching the
    leakage-prevention approach in radiomics_train.py.
    """
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    steps: list[tuple[str, object]] = [("impute", SimpleImputer(strategy="median"))]
    if args.corr_threshold > 0:
        steps.append(("corr_prune", CorrelationPruner(threshold=args.corr_threshold)))
    if args.k_best > 0:
        if args.feature_selection == "mrmr":
            steps.append(("mrmr", MRMRSelector(k=args.k_best)))
        else:
            steps.append(("kbest", SelectKBest(score_func=f_classif, k=args.k_best)))
    if args.classifier == "logistic":
        steps.append(("scale", StandardScaler()))
    steps.append(("clf", build_estimator(args)))
    return Pipeline(steps)


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Return AUC or NaN if fewer than 2 classes are present."""
    if len(np.unique(y_true)) < MIN_CLASS_COUNT:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def _safe_sens_spec(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[float, float]:
    """Return (sensitivity, specificity) or (NaN, NaN)."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.size == CONF_MATRIX_SIZE:
        tn, fp, fn, tp = cm.ravel()
        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
        return sens, spec
    return float("nan"), float("nan")


def train_and_evaluate(
    Xtr: pd.DataFrame,
    ytr: np.ndarray,
    Xte: pd.DataFrame,
    yte: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, float]:
    """Train on (Xtr, ytr), predict probabilities on Xte.

    Applies sanitisation, correlation pruning, and k-best selection
    (all fitted on train only, then aligned to test).

    Returns (y_prob_test, threshold).
    """
    # Sanitise
    Xtr_s = sanitize_numeric(Xtr, "train")
    Xte_s = align_numeric_to_reference(Xte, Xtr_s.columns.tolist(), "test")

    # Build and fit (feature selection lives inside the pipeline)
    pipe = _build_pipeline(args)
    pipe.fit(Xtr_s, ytr)

    classes_ = pipe.named_steps["clf"].classes_
    pos_idx = int(np.where(classes_ == 1)[0][0])
    y_prob = pipe.predict_proba(Xte_s)[:, pos_idx]

    # Optimal threshold (Youden's J on train)
    p_tr = pipe.predict_proba(Xtr_s)[:, pos_idx]
    fpr, tpr, thr = roc_curve(ytr, p_tr)
    threshold = float(thr[np.argmax(tpr - fpr)]) if len(thr) else 0.5

    return y_prob, threshold


# ---------------------------------------------------------------------------
# Analysis 1: Per-site evaluation on existing train/test split
# ---------------------------------------------------------------------------


def per_site_analysis(
    Xtr: pd.DataFrame,
    ytr: np.ndarray,
    Xte: pd.DataFrame,
    yte: np.ndarray,
    sites_test: pd.Series,
    args: argparse.Namespace,
) -> dict:
    """Train on full mixed train split, evaluate per-site on test."""
    y_prob, threshold = train_and_evaluate(Xtr, ytr, Xte, yte, args)
    y_pred = (y_prob >= threshold).astype(int)

    results: dict[str, dict] = {}
    for site in sorted(sites_test.unique()):
        mask = (sites_test == site).to_numpy()
        yt = yte[mask]
        yp = y_prob[mask]
        ypd = y_pred[mask]
        sens, spec = _safe_sens_spec(yt, ypd)
        results[site] = {
            "n": int(mask.sum()),
            "n_pos": int(yt.sum()),
            "auc": _safe_auc(yt, yp),
            "sensitivity": sens,
            "specificity": spec,
        }

    results["_overall"] = {
        "n": len(yte),
        "n_pos": int(yte.sum()),
        "auc": _safe_auc(yte, y_prob),
        "sensitivity": _safe_sens_spec(yte, y_pred)[0],
        "specificity": _safe_sens_spec(yte, y_pred)[1],
        "threshold": threshold,
    }

    return {"per_site": results, "y_prob": y_prob, "y_pred": y_pred}


# ---------------------------------------------------------------------------
# Analysis 2: Leave-one-site-out
# ---------------------------------------------------------------------------


def loso_analysis(
    X_all: pd.DataFrame,
    y_all: np.ndarray,
    sites_all: pd.Series,
    args: argparse.Namespace,
) -> dict:
    """Leave-one-site-out: for each site, train on the rest, test on that site."""
    unique_sites = sorted(sites_all.unique())
    results: dict[str, dict] = {}
    predictions: list[pd.DataFrame] = []

    for held_out in unique_sites:
        mask_test = (sites_all == held_out).to_numpy()
        mask_train = ~mask_test

        Xtr = X_all.iloc[mask_train]
        ytr = y_all[mask_train]
        Xte = X_all.iloc[mask_test]
        yte = y_all[mask_test]

        print(
            f"[LOSO] Held out: {held_out}"
            f" (n_train={mask_train.sum()},"
            f" n_test={mask_test.sum()})"
        )

        y_prob, threshold = train_and_evaluate(Xtr, ytr, Xte, yte, args)
        y_pred = (y_prob >= threshold).astype(int)

        sens, spec = _safe_sens_spec(yte, y_pred)
        results[held_out] = {
            "n_train": int(mask_train.sum()),
            "n_test": int(mask_test.sum()),
            "n_pos_test": int(yte.sum()),
            "auc": _safe_auc(yte, y_prob),
            "sensitivity": sens,
            "specificity": spec,
            "threshold": threshold,
        }

        predictions.append(
            pd.DataFrame(
                {
                    "case_id": Xte.index,
                    "site": held_out,
                    "y_true": yte,
                    "y_prob": y_prob,
                    "y_pred": y_pred,
                }
            )
        )

    preds_df = pd.concat(predictions, ignore_index=True)
    return {"loso": results, "predictions": preds_df}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_roc_multi(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    groups: pd.Series,
    title: str,
    outpath: Path,
) -> None:
    """Overlaid ROC curves, one per group (site), plus overall."""
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = plt.cm.Set1.colors  # noqa: N806

    for i, site in enumerate(sorted(groups.unique())):
        mask = (groups == site).to_numpy()
        yt = y_true[mask]
        yp = y_prob[mask]
        if len(np.unique(yt)) < MIN_CLASS_COUNT:
            continue
        fpr, tpr, _ = roc_curve(yt, yp)
        auc = roc_auc_score(yt, yp)
        ax.plot(
            fpr,
            tpr,
            color=colors[i % len(colors)],
            label=f"{site} (AUC={auc:.3f}, n={mask.sum()})",
        )

    # Overall
    if len(np.unique(y_true)) >= MIN_CLASS_COUNT:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(
            fpr,
            tpr,
            "k--",
            linewidth=2,
            label=f"Overall (AUC={auc:.3f}, n={len(y_true)})",
        )

    ax.plot([0, 1], [0, 1], ":", color="grey")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for site-level analysis."""
    ap = argparse.ArgumentParser(
        description="Site-level A/B analysis on pre-extracted radiomics features.",
    )
    ap.add_argument(
        "--features-dir",
        required=True,
        help="Directory with features_train_final.csv and features_test_final.csv.",
    )
    ap.add_argument("--labels", required=True)
    ap.add_argument(
        "--splits", required=True, help="CSV with case_id and split columns."
    )
    ap.add_argument("--output", required=True)

    # Classifier args (mirror radiomics_train.py)
    ap.add_argument(
        "--classifier", choices=["logistic", "rf", "xgb"], default="logistic"
    )
    ap.add_argument("--logreg-C", type=float, default=1.0)
    ap.add_argument(
        "--logreg-penalty", choices=["l2", "l1", "elasticnet"], default="l2"
    )
    ap.add_argument("--logreg-l1-ratio", type=float, default=0.0)
    ap.add_argument("--rf-n-estimators", type=int, default=300)
    ap.add_argument("--rf-max-depth", type=int, default=None)
    ap.add_argument("--rf-min-samples-leaf", type=int, default=1)
    ap.add_argument("--rf-min-samples-split", type=int, default=2)
    ap.add_argument("--rf-max-features", default="sqrt")
    ap.add_argument("--rf-ccp-alpha", type=float, default=0.0)
    ap.add_argument("--xgb-n-estimators", type=int, default=300)
    ap.add_argument("--xgb-max-depth", type=int, default=4)
    ap.add_argument("--xgb-learning-rate", type=float, default=0.05)
    ap.add_argument("--xgb-subsample", type=float, default=0.8)
    ap.add_argument("--xgb-colsample-bytree", type=float, default=0.8)
    ap.add_argument("--xgb-reg-lambda", type=float, default=1.0)
    ap.add_argument("--xgb-reg-alpha", type=float, default=0.0)
    ap.add_argument("--xgb-scale-pos-weight", type=float, default=1.0)

    # Feature selection
    ap.add_argument("--corr-threshold", type=float, default=0.0)
    ap.add_argument("--k-best", type=int, default=0)
    ap.add_argument(
        "--feature-selection",
        choices=["kbest", "mrmr"],
        default="kbest",
        help=(
            "Filter method after optional correlation"
            " pruning (only active when --k-best > 0)."
        ),
    )
    ap.add_argument("--grid-search", action="store_true")
    ap.add_argument("--include-subtype", action="store_true")
    ap.add_argument(
        "--categorical-encoding",
        choices=["onehot", "ordinal"],
        default="onehot",
        help=(
            "Encoding for --include-subtype. "
            "'onehot' (default) avoids imposing ordinal structure."
        ),
    )

    args = ap.parse_args()
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    feat_dir = Path(args.features_dir)

    # ---- Load data ----
    Xtr_raw = load_features(str(feat_dir / "features_train_final.csv"))
    Xte_raw = load_features(str(feat_dir / "features_test_final.csv"))
    labels = load_labels(args.labels)
    splits = pd.read_csv(args.splits).copy()
    if "case_id" not in splits.columns or "split" not in splits.columns:
        msg = f"{args.splits} must contain columns: case_id, split"
        raise ValueError(msg)
    splits["case_id"] = splits["case_id"].astype(str)
    dup = splits["case_id"][splits["case_id"].duplicated()].unique()
    if len(dup) > 0:
        preview = ", ".join(map(str, dup[:5]))
        msg = (
            f"{args.splits} has duplicate case_id values (n={len(dup)}): "
            f"{preview}{' ...' if len(dup) > DUPLICATE_PREVIEW_LIMIT else ''}"
        )
        raise ValueError(msg)
    split_map = splits.set_index("case_id")["split"]

    missing_tr = Xtr_raw.index.difference(split_map.index)
    missing_te = Xte_raw.index.difference(split_map.index)
    if len(missing_tr) > 0 or len(missing_te) > 0:
        msg = (
            "features contain patient IDs missing from splits file: "
            f"missing_train={len(missing_tr)}, missing_test={len(missing_te)}"
        )
        raise ValueError(msg)

    bad_tr = Xtr_raw.index[split_map.loc[Xtr_raw.index].to_numpy() != "train"]
    bad_te = Xte_raw.index[split_map.loc[Xte_raw.index].to_numpy() != "test"]
    if len(bad_tr) > 0 or len(bad_te) > 0:
        msg = (
            "feature files do not match split assignments in splits.csv: "
            f"train_mismatch={len(bad_tr)}, test_mismatch={len(bad_te)}"
        )
        raise ValueError(msg)

    ytr = labels.loc[Xtr_raw.index, "pcr"].astype(int).to_numpy()
    yte = labels.loc[Xte_raw.index, "pcr"].astype(int).to_numpy()

    subtype_col: str | None = None
    for candidate in ("subtype", "tumor_subtype"):
        if candidate in labels.columns:
            subtype_col = candidate
            break

    if args.include_subtype:
        if subtype_col is None:
            print(
                "[WARN] --include-subtype set but no 'subtype' or 'tumor_subtype' "
                "in labels; skipping.",
            )
        else:
            Xtr_raw, Xte_raw, added = append_categorical_feature(
                Xtr_raw,
                Xte_raw,
                labels,
                column=subtype_col,
                prefix="subtype",
                encoding=args.categorical_encoding,
            )
            print(
                f"[DEBUG] appended subtype features from '{subtype_col}' "
                f"using {args.categorical_encoding}: +{len(added)} columns",
            )

    sites_train = assign_sites_from_labels(labels, Xtr_raw.index)
    sites_test = assign_sites_from_labels(labels, Xte_raw.index)

    print(f"[SITE] Sites found — train: {dict(sites_train.value_counts())}")
    print(f"[SITE] Sites found — test:  {dict(sites_test.value_counts())}")

    # ================================================================
    # Analysis 1: Per-site evaluation on existing split
    # ================================================================
    print("\n" + "=" * 60)
    print("Analysis 1: Per-site evaluation (existing train/test split)")
    print("=" * 60)

    ps_result = per_site_analysis(Xtr_raw, ytr, Xte_raw, yte, sites_test, args)

    with (outdir / "per_site_metrics.json").open("w") as f:
        json.dump(ps_result["per_site"], f, indent=2)

    plot_roc_multi(
        yte,
        ps_result["y_prob"],
        sites_test,
        "Per-Site ROC (existing split)",
        outdir / "roc_per_site.png",
    )

    print("\nPer-site test results:")
    for site, m in ps_result["per_site"].items():
        print(
            f"  {site:>10s}: AUC={m['auc']:.3f}  sens={m['sensitivity']:.3f}  "
            f"spec={m['specificity']:.3f}  n={m['n']}"
        )

    # ================================================================
    # Analysis 2: Leave-one-site-out
    # ================================================================
    print("\n" + "=" * 60)
    print("Analysis 2: Leave-one-site-out (LOSO)")
    print("=" * 60)

    # Combine train + test for LOSO
    X_all = pd.concat([Xtr_raw, Xte_raw])
    y_all = labels.loc[X_all.index, "pcr"].astype(int).to_numpy()
    sites_all = assign_sites_from_labels(labels, X_all.index)

    loso_result = loso_analysis(X_all, y_all, sites_all, args)

    with (outdir / "loso_metrics.json").open("w") as f:
        json.dump(loso_result["loso"], f, indent=2)

    loso_result["predictions"].to_csv(outdir / "predictions_loso.csv", index=False)

    # LOSO ROC plot
    preds_df = loso_result["predictions"]
    plot_roc_multi(
        preds_df["y_true"].to_numpy(),
        preds_df["y_prob"].to_numpy(),
        preds_df["site"],
        "Leave-One-Site-Out ROC",
        outdir / "roc_loso.png",
    )

    print("\nLOSO results:")
    for site, m in loso_result["loso"].items():
        print(
            f"  {site:>10s}: AUC={m['auc']:.3f}  sens={m['sensitivity']:.3f}  "
            f"spec={m['specificity']:.3f}  n_test={m['n_test']}"
        )

    # ================================================================
    # Combined summary CSV
    # ================================================================
    rows = []
    for site, m in ps_result["per_site"].items():
        if site == "_overall":
            continue
        rows.append(
            {
                "analysis": "per_site",
                "site": site,
                "n_test": m["n"],
                "n_pos_test": m["n_pos"],
                "auc": m["auc"],
                "sensitivity": m["sensitivity"],
                "specificity": m["specificity"],
            }
        )
    for site, m in loso_result["loso"].items():
        rows.append(
            {
                "analysis": "loso",
                "site": site,
                "n_train": m["n_train"],
                "n_test": m["n_test"],
                "n_pos_test": m["n_pos_test"],
                "auc": m["auc"],
                "sensitivity": m["sensitivity"],
                "specificity": m["specificity"],
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "summary.csv", index=False)
    print(f"\n[SITE] Summary written to {outdir / 'summary.csv'}")
    print(f"[SITE] Done. Outputs in {outdir}")


if __name__ == "__main__":
    main()
