"""Refresh Issue #120 benchmark CSV and AUC figure from experiment metrics."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ARMS = [
    {
        "label": "ISPY2 baseline (curvature only)",
        "config": "configs/deepsets_ispy2_pointfeat_baseline.yaml",
        "regime": "baseline",
        "out_root": "experiments/deepsets_ispy2_pointfeat_baseline",
    },
    {
        "label": "ISPY2 geometry + topology",
        "config": "configs/deepsets_ispy2_pointfeat_geom_topo.yaml",
        "regime": "geometry_topology",
        "out_root": "experiments/deepsets_ispy2_pointfeat_geom_topo",
    },
    {
        "label": "ISPY2 geometry + topology + dynamic",
        "config": "configs/deepsets_ispy2_pointfeat_geom_topo_dynamic.yaml",
        "regime": "geometry_topology_dynamic",
        "out_root": "experiments/deepsets_ispy2_pointfeat_geom_topo_dynamic",
    },
    {
        "label": "ISPY2 geometry + topology + curvature",
        "config": "configs/deepsets_ispy2_pointfeat_geom_topo_plus_curvature.yaml",
        "regime": "geometry_topology_plus_curvature",
        "out_root": "experiments/deepsets_ispy2_pointfeat_geom_topo_plus_curvature",
    },
    {
        "label": "ISPY2 geometry + topology (no shells)",
        "config": "configs/deepsets_ispy2_pointfeat_geom_topo_no_shells.yaml",
        "regime": "geometry_topology_no_shells",
        "out_root": "experiments/deepsets_ispy2_pointfeat_geom_topo_no_shells",
    },
    {
        "label": "ISPY2 curvature + dynamic",
        "config": "configs/deepsets_ispy2_pointfeat_curvature_plus_dynamic.yaml",
        "regime": "curvature_plus_dynamic",
        "out_root": "experiments/deepsets_ispy2_pointfeat_curvature_plus_dynamic",
    },
    {
        "label": "All cohorts baseline (curvature only)",
        "config": "configs/deepsets_mamamia_pointfeat_baseline.yaml",
        "regime": "baseline",
        "out_root": "experiments/deepsets_mamamia_pointfeat_baseline",
    },
    {
        "label": "All cohorts geometry + topology + curvature",
        "config": "configs/deepsets_mamamia_pointfeat_geom_topo_plus_curvature.yaml",
        "regime": "geometry_topology_plus_curvature",
        "out_root": "experiments/deepsets_mamamia_pointfeat_geom_topo_plus_curvature",
    },
]


def find_latest_deepsets_metrics(out_root: Path) -> Path | None:
    """Return the newest metrics.json under <out_root>/train/."""
    train_root = out_root / "train"
    if not train_root.is_dir():
        return None
    candidates = list(train_root.rglob("metrics.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def build_summary(repo_root: Path) -> pd.DataFrame:
    """Build the Issue #120 benchmark summary table from experiment metrics."""
    deepsets_point_feature_names = importlib.import_module(
        "deepsets.build_dataset"
    ).deepsets_point_feature_names
    rows = []
    for arm in ARMS:
        out = repo_root / arm["out_root"]
        metrics_file = find_latest_deepsets_metrics(out)
        n_features = len(deepsets_point_feature_names(arm["regime"]))
        row = {
            "arm": arm["label"],
            "config": arm["config"],
            "regime": arm["regime"],
            "out_root": arm["out_root"],
            "n_features": n_features,
            "auc_mean": float("nan"),
            "auc_std": float("nan"),
            "metrics_path": "",
            "status": "missing",
        }
        if metrics_file is not None:
            payload = json.loads(metrics_file.read_text(encoding="utf-8"))
            row["auc_mean"] = float(payload["aggregated_metrics"]["auc"]["mean"])
            row["auc_std"] = float(payload["aggregated_metrics"]["auc"]["std"])
            row["metrics_path"] = str(metrics_file)
            row["status"] = "found"
        rows.append(row)
    return pd.DataFrame(rows)


def save_figure(metrics_df: pd.DataFrame, fig_dir: Path) -> None:
    """Write the AUC comparison bar chart when any metrics are present."""
    plot_df = metrics_df.loc[metrics_df["status"] == "found"]
    if plot_df.empty:
        print("No metrics found; skipped benchmark plot.")
        return
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(
        plot_df["arm"],
        plot_df["auc_mean"],
        yerr=plot_df["auc_std"],
        capsize=6,
    )
    ax.set_ylabel("Validation AUC (mean ± std)")
    ax.set_ylim(0.45, 0.60)
    ax.set_title("Deep Sets feature-regime benchmark (runs with metrics present)")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=12, ha="right")
    fig.tight_layout()
    out_png = fig_dir / "feature_set_auc_comparison.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"Saved: {out_png}")


def main() -> None:
    """Refresh benchmark CSV and figure under analysis/figures/deepsets_issue120/."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    fig_dir = repo_root / "analysis" / "figures" / "deepsets_issue120"
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = build_summary(repo_root)
    found = int(metrics_df["status"].eq("found").sum())
    print(f"Metrics loaded for {found}/{len(metrics_df)} experiments.")
    if metrics_df["status"].eq("found").any():
        summary_csv = fig_dir / "feature_set_benchmark_summary.csv"
        metrics_df.to_csv(summary_csv, index=False)
        print(f"Saved: {summary_csv}")
        save_figure(metrics_df, fig_dir)
    else:
        print("Skipped CSV/figure (no metrics yet).")


if __name__ == "__main__":
    main()
