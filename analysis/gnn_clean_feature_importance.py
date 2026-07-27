"""Which of the conditioned (floored) kinetic features carry the pCR signal?

Reads a cache's graph_qc.csv (per-case feature summaries) + the labels file's
predefined folds and reports, on the SAME pooled 179 cases the GNN sees:
  1. Univariate held-out AUC for each of the 7 node features (per-case mean),
     using predefined-fold CV -- ranks individual feature signal.
  2. A tabular reference: logistic regression on the 7 per-case means under the
     same folds -- how close does a plain graph-summary model get to the GNN's
     ~0.61? (If it matches, the graph structure isn't adding much.)
  3. Permutation importance of that tabular model -- multivariate ranking.

Pooled over all cases (UChicago sub-families are one institution, not cohorts).
Reads small CSVs only; head-node safe.

Usage:
    python -m analysis.gnn_clean_feature_importance \
        --graph-qc .../gnn_cache_voxel_richfeat_floored/processed/graph_qc.csv \
        --labels .../uchicago_labels_folded.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "peak_time",
    "peak_enhancement",
    "time_to_enhancement",
    "washin_slope",
    "washout_slope",
    "auc_positive",
    "radius",
]
_CHANCE = 0.5


def _oof_auc_logreg(x: np.ndarray, y: np.ndarray, folds: np.ndarray) -> float:
    """Predefined-fold OOF AUC for logistic regression on columns of ``x``."""
    oof = np.zeros(len(y), dtype=float)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(x[tr], y[tr])
        oof[te] = model.predict_proba(x[te])[:, 1]
    return roc_auc_score(y, oof)


def main() -> None:
    """Print univariate, tabular-reference, and permutation-importance tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-qc", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    args = parser.parse_args()

    qc = pd.read_csv(args.graph_qc)
    labels = pd.read_csv(args.labels)[["case_id", "fold"]]
    merged = qc.merge(labels, on="case_id", how="inner")
    y = merged["pcr"].to_numpy()
    folds = merged["fold"].to_numpy()
    print(
        f"{len(merged)} cases, pCR rate {y.mean():.3f}, {len(np.unique(folds))} folds\n"
    )

    print("=== 1. Univariate OOF AUC per feature (per-case mean, predefined folds) ===")
    rows = []
    for feat in FEATURES:
        col = f"{feat}_mean"
        auc = _oof_auc_logreg(merged[[col]].to_numpy(), y, folds)
        # orient so AUC >= 0.5 and report the direction
        rows.append(
            (feat, max(auc, 1 - auc), "higher->pCR" if auc >= _CHANCE else "lower->pCR")
        )
    for feat, auc, direction in sorted(rows, key=lambda r: -r[1]):
        print(f"  {feat:20s} AUC={auc:.3f}  ({direction})")

    print("\n=== 2. Tabular reference: logreg on all 7 means (predefined folds) ===")
    x = merged[[f"{f}_mean" for f in FEATURES]].to_numpy()
    auc_all = _oof_auc_logreg(x, y, folds)
    print(f"  OOF AUC = {auc_all:.3f}   (GNN baseline on same cache/folds ~0.61)")

    print("\n=== 3. Permutation importance (logreg, refit on full data) ===")
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(x, y)
    imp = permutation_importance(
        model, x, y, scoring="roc_auc", n_repeats=50, random_state=0
    )
    order = np.argsort(-imp.importances_mean)
    for i in order:
        print(
            f"  {FEATURES[i]:20s} Δauc={imp.importances_mean[i]:+.4f} "
            f"± {imp.importances_std[i]:.4f}"
        )


if __name__ == "__main__":
    main()
