"""Roll up the matched voxel/segment/junction graph-representation comparison.

Each arm under ``--campaign-dir`` (subdirectories ``voxel/``, ``segment/``,
``junction/``) was produced by ``gnn/slurm/submit_gnn_graph_compare_train.slurm``
against its own pre-built cache, but all three share an identical cohort, the
identical frozen folds (``split_mode: predefined`` on the shared folded labels
file), and the identical set of underlying signals -- so any difference in
cross-validated AUC reflects the graph representation, not the inputs.

This script reads each arm's ``config_used.json`` (node mode) and ``metrics.json``
(aggregated CV metrics), writes ``graph_compare_metrics.csv`` + a comparison bar
plot, and -- critically -- asserts that the three arms assigned every shared case
to the same fold (diffing their ``split_manifest.csv`` files). That assertion is
what proves the comparison is matched rather than assuming it.

Usage::

    python -m analysis.gnn_graph_compare_plot \
        --campaign-dir experiments/graph_repr_compare_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Plot arms in this order when present, so voxel/segment/junction read left->right.
_ARM_ORDER = ("voxel", "segment", "junction")


def _resolve_run_dir(arm_dir: Path) -> Path:
    """Return the single timestamped run dir under one arm directory.

    ``resolve_run_output_dir`` always appends a ``<name>_<timestamp>``
    subdirectory under the ``--outdir`` passed to training, so each arm dir
    (``experiments/<campaign>/<mode>/``) holds exactly one such run.
    """
    (config_path,) = arm_dir.glob("*/config_used.json")
    return config_path.parent


def collect_arm(arm_dir: Path) -> tuple[dict[str, object], pd.DataFrame]:
    """Read one arm's node mode + aggregated metrics and its split manifest."""
    run_dir = _resolve_run_dir(arm_dir)
    config = json.loads((run_dir / "config_used.json").read_text())
    node_mode = str(config["model_params"]["gnn_node_mode"])
    model_dir = run_dir / config["experiment_setup"]["name"]

    metrics = json.loads((model_dir / "metrics.json").read_text())
    aggregated = metrics["aggregated_metrics"]
    row: dict[str, object] = {
        "arm": arm_dir.name,
        "node_mode": node_mode,
        "n_samples": metrics.get("n_samples"),
    }
    for metric_name, stats in aggregated.items():
        if isinstance(stats, dict) and "mean" in stats:
            row[f"{metric_name}_mean"] = stats["mean"]
            row[f"{metric_name}_std"] = stats["std"]

    manifest = pd.read_csv(model_dir / "split_manifest.csv")
    val_fold = (
        manifest[manifest["train_or_val"] == "val"][["case_id", "fold"]]
        .assign(case_id=lambda d: d["case_id"].astype(str))
        .set_index("case_id")["fold"]
    )
    val_fold.name = node_mode
    return row, val_fold.to_frame()


def assert_matched_folds(fold_frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Raise unless every arm assigned each shared case to the same fold.

    Returns the joined per-case fold table (one column per arm) for the record.
    Cases present in some arms but not others are reported (e.g. a representation
    that dropped a degenerate case) -- and any disagreement on a *shared* case's
    fold is a hard failure, since it would silently unmatch the comparison.
    """
    joined = pd.concat(fold_frames, axis=1)

    missing = joined[joined.isna().any(axis=1)]
    if not missing.empty:
        raise ValueError(
            f"{len(missing)} case(s) are not present in every arm (cohort "
            f"mismatch): {missing.index.tolist()[:10]}"
        )

    # All arms are complete here (missing checked above), so a row where the min
    # and max fold differ is a case that got different folds across arms.
    disagree = joined[joined.min(axis=1) != joined.max(axis=1)]
    if not disagree.empty:
        raise ValueError(
            f"{len(disagree)} case(s) got different folds across arms -- the "
            f"comparison is NOT matched: {disagree.index.tolist()[:10]}"
        )
    return joined


def _arm_dirs(campaign_dir: Path) -> list[Path]:
    """Arm subdirectories, ordered voxel/segment/junction when present."""
    present = {p.name: p for p in campaign_dir.iterdir() if p.is_dir()}
    ordered = [present[name] for name in _ARM_ORDER if name in present]
    extras = sorted(p for name, p in present.items() if name not in _ARM_ORDER)
    return ordered + extras


def main() -> None:
    """Collect every arm, verify matched folds, and write the CSV + bar plot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    args = parser.parse_args()

    arm_dirs = _arm_dirs(args.campaign_dir)
    if not arm_dirs:
        raise ValueError(f"No arm subdirectories found under {args.campaign_dir}")

    rows: list[dict[str, object]] = []
    fold_frames: list[pd.DataFrame] = []
    for arm_dir in arm_dirs:
        row, fold_frame = collect_arm(arm_dir)
        rows.append(row)
        fold_frames.append(fold_frame)

    fold_table = assert_matched_folds(fold_frames)
    print(
        f"Matched-fold check passed: {len(fold_table)} cases share identical "
        f"folds across {len(arm_dirs)} arms."
    )

    summary = pd.DataFrame(rows)
    csv_path = args.campaign_dir / "graph_compare_metrics.csv"
    summary.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(
        summary["node_mode"],
        summary["auc_mean"],
        yerr=summary["auc_std"],
        capsize=6,
        color=["tab:blue", "tab:green", "tab:orange"][: len(summary)],
    )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="chance")
    ax.set_ylabel("Cross-validated val AUC (mean +/- std over folds)")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Graph representation comparison (matched cohort/folds/features)")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    plot_path = args.campaign_dir / "graph_compare_auc.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    print(f"Wrote {csv_path} and {plot_path}")


if __name__ == "__main__":
    main()
