"""Phase 1 gate: does the vessel-GNN beat a tabular bar by a CI-separated margin?

Puts every contender on one footing -- **pooled OOF AUC over all 179 UChicago
cases** (the plan's metric convention) with a **case-resampling bootstrap 95%
CI** (Phase 0.3) -- and then runs the actual gate test: a **paired** bootstrap of
the AUC *difference* between the GNN and the strongest tabular model, resampling
the same cases jointly so the two share sampling noise. Overlapping marginal CIs
would be an over-conservative proxy; the paired dAUC CI is the correct read of
"beats by a CI-separated margin."

Contenders:
  - GNN baseline / tier1: per-case OOF ``y_prob`` from each seed's
    ``predictions.csv`` in the floored tier sweep, averaged across seeds.
  - Tabular 7-means: logistic regression on the 7 per-case node-feature means
    (reproduces the ~0.614 documented bar), predefined-fold OOF.
  - Tabular 38-feat: logistic regression + XGBoost on the full vessel-summary
    set (mean/std/q10/q50/q90 + counts) from tabular.gnn_feature_baseline.

Reads only small CSVs (no graph-cache load) -- head-node safe.

Usage:
    python -m analysis.gnn_tabular_gate \
        --gnn-sweep-root /gpfs/.../experiments/gnn_uchicago_tier_sweep_floored \
        --graph-qc /ess/.../gnn_cache_voxel_richfeat_floored/processed/graph_qc.csv \
        --tabular-features experiments/uchicago_tabular_bar_floored/tabular_baseline_features.csv \
        --out-dir experiments/uchicago_tabular_bar_floored
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

_SEEDS = (42, 142, 242)
_GNN_SEED_DIRS = (0, 1, 2, 3, 4)
_N_BOOTSTRAP = 2000
_CI_PCT = (2.5, 97.5)
_MEANS7 = [
    "peak_time_mean",
    "peak_enhancement_mean",
    "time_to_enhancement_mean",
    "washin_slope_mean",
    "washout_slope_mean",
    "auc_positive_mean",
    "radius_mean",
]


def _oof_logreg(
    x: np.ndarray, y: np.ndarray, folds: np.ndarray, seed: int
) -> np.ndarray:
    """Predefined-fold out-of-fold probabilities for logistic regression."""
    oof = np.full(len(y), np.nan)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        model = make_pipeline(
            SimpleImputer(strategy="mean"),
            StandardScaler(),
            LogisticRegression(max_iter=1000, random_state=seed),
        )
        model.fit(x[tr], y[tr])
        oof[te] = model.predict_proba(x[te])[:, 1]
    return oof


def _oof_xgb(x: np.ndarray, y: np.ndarray, folds: np.ndarray, seed: int) -> np.ndarray:
    """Predefined-fold out-of-fold probabilities for XGBoost."""
    oof = np.full(len(y), np.nan)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        model = make_pipeline(
            SimpleImputer(strategy="mean"),
            XGBClassifier(
                n_estimators=100, max_depth=3, random_state=seed, eval_metric="logloss"
            ),
        )
        model.fit(x[tr], y[tr])
        oof[te] = model.predict_proba(x[te])[:, 1]
    return oof


def _seed_mean_oof(
    oof_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, int], np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
) -> np.ndarray:
    """Average a model's OOF probabilities across seeds to damp init noise."""
    return np.mean([oof_fn(x, y, folds, s) for s in _SEEDS], axis=0)


def _gnn_pooled_oof(sweep_root: Path, arm: str, case_order: list[str]) -> np.ndarray:
    """Per-case OOF prob for a GNN arm, averaged over the 5 seed runs.

    Each seed dir holds one ``predictions.csv`` with one held-out row per case
    (predefined folds). Averaging the per-case ``y_prob`` across seeds mirrors the
    tabular seed-averaging so the two are compared on the same construction.
    """
    per_seed = []
    for s in _GNN_SEED_DIRS:
        matches = sorted(sweep_root.glob(f"{arm}_seed{s}/**/predictions.csv"))
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one predictions.csv for {arm}_seed{s}, "
                f"found {len(matches)}: {matches}"
            )
        preds = pd.read_csv(matches[0]).set_index("case_id")
        per_seed.append(preds.loc[case_order, "y_prob"].to_numpy())
    return np.mean(per_seed, axis=0)


def _boot_index_matrix(n: int, seed: int = 0) -> np.ndarray:
    """One shared set of case-resampling draws so every CI uses the same resamples."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(_N_BOOTSTRAP, n))


def _auc_ci(
    y: np.ndarray, prob: np.ndarray, boot: np.ndarray
) -> tuple[float, float, float]:
    """Point AUC on full data + percentile bootstrap CI over the shared draws."""
    point = float(roc_auc_score(y, prob))
    aucs = np.array([roc_auc_score(y[idx], prob[idx]) for idx in boot])
    lo, hi = np.percentile(aucs, _CI_PCT)
    return point, float(lo), float(hi)


def _paired_delta_ci(
    y: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray, boot: np.ndarray
) -> dict[str, float]:
    """Paired bootstrap of AUC(a) - AUC(b): same resample scores both models.

    ``prob_gt_zero`` is the bootstrap fraction with a positive delta -- a one-sided
    "how often does A win" read; the gate is cleared only if the CI excludes 0.
    """
    point = float(roc_auc_score(y, prob_a) - roc_auc_score(y, prob_b))
    deltas = np.array(
        [roc_auc_score(y[i], prob_a[i]) - roc_auc_score(y[i], prob_b[i]) for i in boot]
    )
    lo, hi = np.percentile(deltas, _CI_PCT)
    return {
        "delta_auc": point,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "frac_positive": float(np.mean(deltas > 0)),
    }


def main() -> None:
    """Score every contender, print/save the AUC-CI table + paired gate + plot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gnn-sweep-root", type=Path, required=True)
    parser.add_argument("--graph-qc", type=Path, required=True)
    parser.add_argument("--tabular-features", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--arms", type=str, default="baseline,tier1")
    args = parser.parse_args()

    feats = pd.read_csv(args.tabular_features)
    case_order = feats["case_id"].tolist()
    y = feats["pcr"].to_numpy()
    folds = feats["fold"].to_numpy()
    feat_cols = [
        c for c in feats.columns if c not in ("case_id", "pcr", "dataset", "fold")
    ]

    qc = pd.read_csv(args.graph_qc).set_index("case_id").loc[case_order]
    x_means7 = qc[_MEANS7].to_numpy()
    x_full = feats[feat_cols].to_numpy()

    boot = _boot_index_matrix(len(y))

    contenders: dict[str, np.ndarray] = {}
    for arm in args.arms.split(","):
        contenders[f"gnn_{arm}"] = _gnn_pooled_oof(args.gnn_sweep_root, arm, case_order)
    contenders["tabular_7means_logreg"] = _seed_mean_oof(
        _oof_logreg, x_means7, y, folds
    )
    contenders["tabular_38feat_logreg"] = _seed_mean_oof(_oof_logreg, x_full, y, folds)
    contenders["tabular_38feat_xgboost"] = _seed_mean_oof(_oof_xgb, x_full, y, folds)

    rows = []
    for name, prob in contenders.items():
        point, lo, hi = _auc_ci(y, prob, boot)
        rows.append({"model": name, "pooled_oof_auc": point, "ci_lo": lo, "ci_hi": hi})
    table = pd.DataFrame(rows).sort_values("pooled_oof_auc", ascending=False)

    print("=== Pooled OOF AUC (all 179 cases) with bootstrap 95% CI ===")
    for _, r in table.iterrows():
        print(
            f"  {r['model']:26s} {r['pooled_oof_auc']:.3f}  [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
        )

    # Gate: strongest GNN arm vs strongest tabular model, paired.
    gnn_names = [n for n in contenders if n.startswith("gnn_")]
    tab_names = [n for n in contenders if n.startswith("tabular_")]
    best_gnn = max(gnn_names, key=lambda n: roc_auc_score(y, contenders[n]))
    best_tab = max(tab_names, key=lambda n: roc_auc_score(y, contenders[n]))
    delta = _paired_delta_ci(y, contenders[best_gnn], contenders[best_tab], boot)

    print()
    print(f"=== GATE: paired dAUC ({best_gnn} - {best_tab}) ===")
    print(
        f"  dAUC = {delta['delta_auc']:+.3f}  95% CI [{delta['ci_lo']:+.3f}, {delta['ci_hi']:+.3f}]"
        f"  (P[dAUC>0] = {delta['frac_positive']:.2f})"
    )
    gate_pass = delta["ci_lo"] > 0.0
    print(
        f"  GATE {'PASSED' if gate_pass else 'NOT PASSED'}: "
        f"CI {'excludes' if gate_pass else 'includes'} 0."
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / "gate_auc_ci.csv", index=False)
    pd.DataFrame(
        [
            {
                "best_gnn": best_gnn,
                "best_tabular": best_tab,
                **delta,
                "gate_passed": gate_pass,
            }
        ]
    ).to_csv(args.out_dir / "gate_paired_delta.csv", index=False)

    _forest_plot(table, delta, best_gnn, best_tab, args.out_dir / "gate_forest.png")
    print(
        f"\nWrote {args.out_dir}/gate_auc_ci.csv, gate_paired_delta.csv, gate_forest.png"
    )


def _forest_plot(
    table: pd.DataFrame,
    delta: dict[str, float],
    best_gnn: str,
    best_tab: str,
    out_path: Path,
) -> None:
    """Forest plot of each model's pooled-OOF AUC with its 95% CI."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = table.iloc[::-1].reset_index(drop=True)  # highest AUC at top
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(t) + 1.2))
    ys = np.arange(len(t))
    is_gnn = t["model"].str.startswith("gnn_").to_numpy()
    # errorbar's ecolor takes one color, so plot the GNN and tabular rows as two
    # calls (blue vs grey) rather than a per-row color array.
    for mask, color, label in (
        (is_gnn, "#1f77b4", "GNN"),
        (~is_gnn, "#7f7f7f", "tabular"),
    ):
        if not mask.any():
            continue
        sub, ysub = t[mask], ys[mask]
        ax.errorbar(
            sub["pooled_oof_auc"],
            ysub,
            xerr=[
                sub["pooled_oof_auc"] - sub["ci_lo"],
                sub["ci_hi"] - sub["pooled_oof_auc"],
            ],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=2,
            capsize=4,
            markersize=6,
            zorder=3,
            label=label,
        )
    ax.axvline(0.5, color="k", ls=":", lw=1, label="chance")
    ax.set_yticks(ys)
    ax.set_yticklabels(t["model"])
    ax.set_xlabel("Pooled OOF AUC (95% bootstrap CI)")
    ax.set_title(
        f"UChicago pCR: GNN vs tabular bar (N=179)\n"
        f"gate dAUC({best_gnn.replace('gnn_','')} - {best_tab.replace('tabular_','')}) "
        f"= {delta['delta_auc']:+.3f} [{delta['ci_lo']:+.3f}, {delta['ci_hi']:+.3f}]"
    )
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
