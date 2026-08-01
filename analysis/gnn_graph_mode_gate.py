"""Phase 2.2 gate: does any graph MODE carry pCR signal voxel-mean discards?

Compares the floored UChicago GNN across the three graph representations --
voxel (one node per skeleton voxel), segment (one node per vessel segment), and
junction (junction nodes + segment edges) -- on the shared metric: pooled OOF
AUC over all 179 cases with a case-resampling bootstrap 95% CI. The decisive
test is a **paired** bootstrap dAUC of each structural mode vs. voxel (the mode
that is nearly tabular-by-construction): if segment/junction beats voxel by a
CI-separated margin, the vessel-tree structure carries signal the per-voxel mean
throws away. The tabular 7-means bar is included for reference.

Mirrors the Duke graph_repr_compare_v1 result (voxel 0.526 / junction 0.511 /
segment 0.502, all ~chance) -- rerun here on data that, post-flooring, actually
has signal. Reuses the pooled-OOF/CI/paired-dAUC helpers from
analysis.gnn_tabular_gate. Reads only small CSVs -- head-node safe.

Usage:
    python -m analysis.gnn_graph_mode_gate \
        --voxel-root /gpfs/.../experiments/gnn_uchicago_tier_sweep_floored \
        --graph-mode-root /gpfs/.../experiments/gnn_uchicago_graph_mode_floored \
        --graph-qc /ess/.../gnn_cache_voxel_richfeat_floored/processed/graph_qc.csv \
        --tabular-features experiments/uchicago_tabular_bar_floored/tabular_baseline_features.csv \
        --out-dir experiments/uchicago_graph_mode_floored
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
    _paired_delta_ci,
    _seed_mean_oof,
)


def main() -> None:
    """Score voxel/segment/junction + tabular bar, print/save table + paired deltas."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voxel-root", type=Path, required=True)
    parser.add_argument("--voxel-arm", type=str, default="baseline")
    parser.add_argument("--graph-mode-root", type=Path, required=True)
    parser.add_argument("--graph-qc", type=Path, required=True)
    parser.add_argument("--tabular-features", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    feats = pd.read_csv(args.tabular_features)
    case_order = feats["case_id"].tolist()
    y = feats["pcr"].to_numpy()
    folds = feats["fold"].to_numpy()
    qc = pd.read_csv(args.graph_qc).set_index("case_id").loc[case_order]
    x_means7 = qc[_MEANS7].to_numpy()

    boot = _boot_index_matrix(len(y))

    contenders: dict[str, np.ndarray] = {
        "gnn_voxel": _gnn_pooled_oof(args.voxel_root, args.voxel_arm, case_order),
        "gnn_segment": _gnn_pooled_oof(args.graph_mode_root, "segment", case_order),
        "gnn_junction": _gnn_pooled_oof(args.graph_mode_root, "junction", case_order),
        "tabular_7means_logreg": _seed_mean_oof(_oof_logreg, x_means7, y, folds),
    }

    rows = []
    for name, prob in contenders.items():
        point, lo, hi = _auc_ci(y, prob, boot)
        rows.append({"model": name, "pooled_oof_auc": point, "ci_lo": lo, "ci_hi": hi})
    table = pd.DataFrame(rows).sort_values("pooled_oof_auc", ascending=False)

    print("=== Pooled OOF AUC (all 179 cases) with bootstrap 95% CI ===")
    for _, r in table.iterrows():
        print(
            f"  {r['model']:26s} {r['pooled_oof_auc']:.3f}  "
            f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
        )

    # Paired dAUC of each structural mode vs. voxel (the near-tabular baseline).
    print("\n=== Paired dAUC vs. voxel (structure test) ===")
    deltas = []
    for mode in ("gnn_segment", "gnn_junction"):
        d = _paired_delta_ci(y, contenders[mode], contenders["gnn_voxel"], boot)
        gate = d["ci_lo"] > 0.0
        deltas.append({"comparison": f"{mode}_minus_voxel", **d, "gate_passed": gate})
        print(
            f"  {mode:14s} - voxel: dAUC={d['delta_auc']:+.3f} "
            f"[{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}] P[>0]={d['frac_positive']:.2f} "
            f"-> {'BEATS voxel' if gate else 'no CI-separated gain'}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / "graph_mode_auc_ci.csv", index=False)
    pd.DataFrame(deltas).to_csv(
        args.out_dir / "graph_mode_paired_delta.csv", index=False
    )
    _forest_plot(table, args.out_dir / "graph_mode_forest.png")
    print(
        f"\nWrote {args.out_dir}/graph_mode_auc_ci.csv, graph_mode_paired_delta.csv, "
        "graph_mode_forest.png"
    )


def _forest_plot(table: pd.DataFrame, out_path: Path) -> None:
    """Forest plot of each mode's pooled-OOF AUC with its 95% CI."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = table.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(t) + 1.2))
    ys = np.arange(len(t))
    is_gnn = t["model"].str.startswith("gnn_").to_numpy()
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
    ax.set_title("UChicago pCR: graph-mode comparison (floored, N=179)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
