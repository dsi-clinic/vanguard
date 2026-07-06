#!/usr/bin/env python3
"""Signal-preservation test for ISPY2 graph features: native vs resampled.

The batch-detectability test showed resampling removes a chunk of scanner
variance. This test asks the complementary question: does resampling also weaken
the features' association with the *outcome we actually care about* (pCR)?

If resampling mostly scrubbed harmless scanner noise, the pCR signal should
survive. If resampling's smoothing blurred the real vessel structure, the pCR
signal should drop -- which would explain "prettier masks, worse models".

We quantify pCR signal two ways, on the SAME matched cases and SAME graph
features, native vs resampled:

  1. Univariate: for each feature on its own, how well does it separate pCR from
     non-pCR? We use |AUC - 0.5| (distance from chance; direction-agnostic) and,
     as a nonlinear check, mutual information. Then we compare native vs
     resampled feature-by-feature (paired Wilcoxon signed-rank).
  2. Multivariate: a 5-fold cross-validated classifier using ALL graph features
     to predict pCR, native vs resampled. This mirrors the real modeling: it is
     the graph arm's actual pCR predictive power.

Scope (fixed by request): ISPY2 only, GRAPH arm only, matched cases.

Usage::

    python scripts/signal_preservation_ispy2.py \
        --native   .../tabular/all_datasets_run/features_full_labeled.csv \
        --resampled .../independent_signal_resampled_ispy2/features_full_labeled.csv \
        --outdir   /ess/scratch/scratch1/t-9sbose/ispy2_signal_preservation
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

DATASET = "ISPY2"
ID_COL = "case_id"
PCR_COL = "pcr"
N_SPLITS = 5
MIN_DISTINCT_FOR_AUC = 2  # a feature needs >=2 distinct values to carry signal


def load_ispy2(path: Path, prefixes: tuple[str, ...]) -> pd.DataFrame:
    """Load an ISPY2 feature file, keeping case_id, pcr, and selected arms."""
    feats = pd.read_csv(path, low_memory=False)
    if "dataset" in feats.columns:
        feats = feats[feats["dataset"] == DATASET]
    feat_cols = sorted(c for c in feats.columns if c.startswith(prefixes))
    keep = (
        [ID_COL, PCR_COL, *feat_cols]
        if PCR_COL in feats.columns
        else [ID_COL, *feat_cols]
    )
    return feats[keep].copy()


def univariate_abs_auc(x_imputed: np.ndarray, y: np.ndarray) -> np.ndarray:
    """|AUC - 0.5| of each column predicting y on its own (0 = no signal)."""
    out = np.empty(x_imputed.shape[1])
    for j in range(x_imputed.shape[1]):
        col = x_imputed[:, j]
        if np.unique(col).size < MIN_DISTINCT_FOR_AUC:  # constant = no signal
            out[j] = 0.0
        else:
            out[j] = abs(roc_auc_score(y, col) - 0.5)
    return out


def multivariate_auc(
    x: np.ndarray,
    y: np.ndarray,
    model: str,
    folds: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> float:
    """5-fold CV pCR AUC using all graph features together."""
    clf = (
        LogisticRegression(max_iter=2000, class_weight="balanced")
        if model == "logreg"
        else RandomForestClassifier(
            n_estimators=400, class_weight="balanced", random_state=seed, n_jobs=-1
        )
    )
    pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", clf),
        ]
    )
    proba = cross_val_predict(pipe, x, y, cv=folds, method="predict_proba")
    return float(roc_auc_score(y, proba[:, 1]))


def main() -> None:
    """Run native-vs-resampled pCR signal-preservation and write outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--resampled", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--feature-prefixes",
        default="graph_",
        help="comma-separated feature-arm prefixes, e.g. 'morph_,kinematic_'",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    prefixes = tuple(p for p in args.feature_prefixes.split(",") if p)
    log.info("Feature arms: %s", prefixes)

    native = load_ispy2(args.native, prefixes)
    resampled = load_ispy2(args.resampled, prefixes)
    feat_cols = sorted(
        {c for c in native.columns if c.startswith(prefixes)}
        & {c for c in resampled.columns if c.startswith(prefixes)}
    )
    shared_ids = sorted(set(native[ID_COL]) & set(resampled[ID_COL]))
    log.info(
        "Matched cases: %d | shared features: %d",
        len(shared_ids),
        len(feat_cols),
    )

    native = native.set_index(ID_COL).loc[shared_ids]
    resampled = resampled.set_index(ID_COL).loc[shared_ids]
    # pCR label comes from the native table (canonical); confirm resampled agrees.
    y = native[PCR_COL].to_numpy().astype(int)
    if PCR_COL in resampled.columns:
        assert np.array_equal(
            y, resampled[PCR_COL].to_numpy().astype(int)
        ), "pCR labels disagree between native and resampled tables"
    log.info("pCR rate: %.3f (%d/%d)", y.mean(), int(y.sum()), len(y))

    x_nat = native[feat_cols].to_numpy()
    x_res = resampled[feat_cols].to_numpy()

    # --- 1. Univariate per-feature pCR signal ---------------------------------
    imputer = SimpleImputer(strategy="median")
    nat_absauc = univariate_abs_auc(imputer.fit_transform(x_nat), y)
    res_absauc = univariate_abs_auc(imputer.fit_transform(x_res), y)
    nat_mi = mutual_info_classif(
        imputer.fit_transform(x_nat), y, random_state=args.seed
    )
    res_mi = mutual_info_classif(
        imputer.fit_transform(x_res), y, random_state=args.seed
    )

    per_feat = pd.DataFrame(
        {
            "feature": feat_cols,
            "native_abs_auc": nat_absauc,
            "resampled_abs_auc": res_absauc,
            "abs_auc_change": res_absauc - nat_absauc,  # <0 = resampling weakened it
            "native_mi": nat_mi,
            "resampled_mi": res_mi,
        }
    ).sort_values("abs_auc_change")
    per_feat.to_csv(args.outdir / "per_feature_signal.csv", index=False)

    n_weakened = int((per_feat["abs_auc_change"] < 0).sum())
    # Paired test: did |AUC-0.5| systematically change native -> resampled?
    w_stat, w_p = stats.wilcoxon(nat_absauc, res_absauc)

    # --- 2. Multivariate CV pCR AUC (all graph features together) -------------
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=args.seed)
    folds = list(skf.split(np.zeros(len(y)), y))
    multi = []
    for model in ("logreg", "rf"):
        auc_nat = multivariate_auc(x_nat, y, model, folds, args.seed)
        auc_res = multivariate_auc(x_res, y, model, folds, args.seed)
        multi.append(
            {
                "model": model,
                "native_auc": auc_nat,
                "resampled_auc": auc_res,
                "auc_change": auc_res - auc_nat,
            }
        )
        log.info(
            "  multivariate %-6s native=%.3f resampled=%.3f (Δ=%+.3f)",
            model,
            auc_nat,
            auc_res,
            auc_res - auc_nat,
        )
    multi = pd.DataFrame(multi)
    multi.to_csv(args.outdir / "multivariate_pcr_auc.csv", index=False)

    arm = "+".join(p.rstrip("_") for p in prefixes)
    _plot_univariate(per_feat, args.outdir / "univariate_signal_scatter.png")
    _plot_multivariate(multi, args.outdir / "multivariate_pcr_auc.png", arm)
    _write_summary(
        per_feat,
        multi,
        nat_absauc,
        res_absauc,
        n_weakened,
        w_p,
        len(shared_ids),
        len(feat_cols),
        y,
        arm,
        args.outdir / "SUMMARY.txt",
    )

    log.info(
        "\n=== per-feature (most weakened first) ===\n%s",
        per_feat.head(10).to_string(index=False),
    )
    log.info(
        "\nmean |AUC-0.5|: native=%.4f resampled=%.4f | features weakened: %d/%d"
        " | Wilcoxon p=%.3g",
        nat_absauc.mean(),
        res_absauc.mean(),
        n_weakened,
        len(feat_cols),
        w_p,
    )
    log.info("\nAll outputs written to: %s", args.outdir)


def _plot_univariate(per_feat: pd.DataFrame, path: Path) -> None:
    """Scatter: resampled vs native per-feature |AUC-0.5| (below line = lost)."""
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(
        per_feat["native_abs_auc"],
        per_feat["resampled_abs_auc"],
        s=28,
        alpha=0.75,
        color="steelblue",
    )
    hi = (
        float(
            max(per_feat["native_abs_auc"].max(), per_feat["resampled_abs_auc"].max())
        )
        * 1.1
    )
    ax.plot([0, hi], [0, hi], ls="--", color="0.5", label="no change")
    ax.set_xlabel("native |AUC - 0.5|  (per-feature pCR signal)")
    ax.set_ylabel("resampled |AUC - 0.5|")
    ax.set_title(
        "Per-feature pCR signal: native vs resampled\n"
        "(points below the line = signal lost to resampling)"
    )
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_multivariate(multi: pd.DataFrame, path: Path, arm: str) -> None:
    """Bars: multivariate CV pCR AUC, native vs resampled, per model."""
    x = np.arange(len(multi))
    width = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.bar(x - width / 2, multi["native_auc"], width, label="native", color="steelblue")
    ax.bar(
        x + width / 2,
        multi["resampled_auc"],
        width,
        label="resampled",
        color="darkorange",
    )
    ax.axhline(0.5, ls="--", color="0.5", label="chance (AUC 0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(multi["model"])
    ax.set_ylabel(f"5-fold CV pCR AUC (all {arm} features)")
    ax.set_ylim(
        0.4, max(0.75, float(multi[["native_auc", "resampled_auc"]].max().max()) + 0.05)
    )
    ax.set_title(f"{arm}-arm pCR prediction: native vs resampled")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_summary(
    per_feat: pd.DataFrame,
    multi: pd.DataFrame,
    nat_absauc: np.ndarray,
    res_absauc: np.ndarray,
    n_weakened: int,
    w_p: float,
    n_cases: int,
    n_feats: int,
    y: np.ndarray,
    arm: str,
    path: Path,
) -> None:
    """Human-readable summary of the signal-preservation comparison."""
    with path.open("w") as fh:
        w = lambda s: fh.write(s + "\n")  # noqa: E731 - tiny local writer
        w(f"ISPY2 signal preservation — native vs resampled {arm} features")
        w("=" * 62)
        w(
            f"Matched cases: {n_cases} | {arm} features: {n_feats} | "
            f"pCR rate {y.mean():.3f}"
        )
        w("")
        w("Univariate per-feature pCR signal (|AUC - 0.5|, higher = more signal):")
        w(f"  mean native    = {nat_absauc.mean():.4f}")
        w(f"  mean resampled = {res_absauc.mean():.4f}")
        w(f"  features weakened by resampling: {n_weakened}/{n_feats}")
        w(f"  paired Wilcoxon (native vs resampled): p = {w_p:.3g}")
        w("")
        w("Multivariate 5-fold CV pCR AUC (all graph features together):")
        w(multi.to_string(index=False))
        w("")
        w("Most-weakened features (native -> resampled |AUC-0.5|):")
        w(
            per_feat.head(8)[
                ["feature", "native_abs_auc", "resampled_abs_auc", "abs_auc_change"]
            ].to_string(index=False)
        )
    log.info("\n%s", path.read_text())


if __name__ == "__main__":
    main()
