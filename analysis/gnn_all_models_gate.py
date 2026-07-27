"""All-models summary: every UChicago pCR model on one pooled-OOF AUC + CI plot.

Puts the tabular baselines and every GNN arm (2-feature and 7-feature voxel
baseline, tier0/tier1 hygiene ladder, and the segment/junction graph modes) on
the same footing -- pooled OOF AUC over all 179 cases with a shared-draw
bootstrap 95% CI -- and renders one grouped forest plot with error bars. Reuses
the pooled-OOF/CI helpers from analysis.gnn_tabular_gate. Reads only small CSVs.

Usage:
    python -m analysis.gnn_all_models_gate \
        --tier-sweep-root /gpfs/.../experiments/gnn_uchicago_tier_sweep_floored \
        --graph-mode-root /gpfs/.../experiments/gnn_uchicago_graph_mode_floored \
        --twofeat-root /gpfs/.../experiments/gnn_uchicago_2feat_floored \
        --graph-qc /ess/.../gnn_cache_voxel_richfeat_floored/processed/graph_qc.csv \
        --tabular-features experiments/uchicago_tabular_bar_floored/tabular_baseline_features.csv \
        --out-dir experiments/uchicago_all_models
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.gnn_tabular_gate import (
    _MEANS7,
    _auc_ci,
    _boot_index_matrix,
    _gnn_pooled_oof,
    _oof_logreg,
    _oof_xgb,
    _seed_mean_oof,
)


def main() -> None:
    """Score every model, print the AUC-CI table, render the grouped forest plot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier-sweep-root", type=Path, required=True)
    parser.add_argument("--graph-mode-root", type=Path, required=True)
    parser.add_argument("--twofeat-root", type=Path, required=True)
    parser.add_argument("--graph-qc", type=Path, required=True)
    parser.add_argument("--tabular-features", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    feats = pd.read_csv(args.tabular_features)
    case_order = feats["case_id"].tolist()
    y = feats["pcr"].to_numpy()
    folds = feats["fold"].to_numpy()
    feat_cols = [
        c for c in feats.columns if c not in ("case_id", "pcr", "dataset", "fold")
    ]
    x_full = feats[feat_cols].to_numpy()
    qc = pd.read_csv(args.graph_qc).set_index("case_id").loc[case_order]
    x_means7 = qc[_MEANS7].to_numpy()

    boot = _boot_index_matrix(len(y))

    # (label, group, oof-probability vector). GNN arms read saved OOF predictions;
    # tabular arms are fit here from the summary features.
    gnn_specs = [
        ("GNN voxel 2feat", args.twofeat_root, "voxel2feat"),
        ("GNN voxel 7feat (baseline)", args.tier_sweep_root, "baseline"),
        ("GNN voxel tier0", args.tier_sweep_root, "tier0"),
        ("GNN voxel tier1", args.tier_sweep_root, "tier1"),
        ("GNN segment", args.graph_mode_root, "segment"),
        ("GNN junction", args.graph_mode_root, "junction"),
    ]
    models: list[tuple[str, str, np.ndarray]] = []
    for label, root, arm in gnn_specs:
        models.append((label, "GNN", _gnn_pooled_oof(root, arm, case_order)))
    models.append(
        (
            "tabular 7-means logreg",
            "tabular",
            _seed_mean_oof(_oof_logreg, x_means7, y, folds),
        )
    )
    models.append(
        (
            "tabular 38-feat logreg",
            "tabular",
            _seed_mean_oof(_oof_logreg, x_full, y, folds),
        )
    )
    models.append(
        (
            "tabular 38-feat xgboost",
            "tabular",
            _seed_mean_oof(_oof_xgb, x_full, y, folds),
        )
    )

    rows = []
    for label, group, prob in models:
        point, lo, hi = _auc_ci(y, prob, boot)
        rows.append(
            {
                "model": label,
                "group": group,
                "pooled_oof_auc": point,
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )
    table = pd.DataFrame(rows).sort_values("pooled_oof_auc", ascending=False)

    print("=== Pooled OOF AUC (all 179 cases) with bootstrap 95% CI ===")
    for _, r in table.iterrows():
        print(
            f"  {r['model']:30s} [{r['group']:7s}] {r['pooled_oof_auc']:.3f}  "
            f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / "all_models_auc_ci.csv", index=False)
    _forest_plot(table, args.out_dir / "all_models_forest.png")
    print(f"\nWrote {args.out_dir}/all_models_auc_ci.csv, all_models_forest.png")


def _forest_plot(table: pd.DataFrame, out_path: Path) -> None:
    """Grouped forest plot of every model's pooled-OOF AUC with its 95% CI."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"GNN": "#1f77b4", "tabular": "#7f7f7f"}
    t = table.iloc[::-1].reset_index(drop=True)  # highest AUC at top
    ys = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(t) + 1.5))
    for group, color in colors.items():
        mask = (t["group"] == group).to_numpy()
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
            label=group,
        )
    ax.axvline(0.5, color="k", ls=":", lw=1, label="chance")
    ax.set_yticks(ys)
    ax.set_yticklabels(t["model"])
    ax.set_xlabel("Pooled OOF AUC (95% bootstrap CI), N=179")
    ax.set_title("UChicago pCR — all models (floored)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
