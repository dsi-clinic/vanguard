"""Collect and plot the pCR-as-Gaussian learnability sweep (gnn_pcr_gaussian_grid).

Each run under ``--grid-dir`` was produced by
``gnn/slurm/submit_gnn_pcr_gaussian_sweep.slurm`` (see
``gnn/README.md#pcr-as-gaussian-learnability-sweep``): the "pcr_dummy" node
feature is redrawn as class-conditional Gaussian noise (label 0 ->
N(class0_mean, sigma^2), label 1 -> N(class1_mean, sigma^2)) and Cohen's
d = (class1_mean - class0_mean) / noise_std is the signal-to-noise ratio
under test. This script reads each run's ``config_used.json`` (for d) and
``metrics.json`` / ``loss_history.csv`` (for val/train AUC), then plots val
AUC vs d against the theoretical single-feature-optimal curve
Phi(d / sqrt(2)) (a 1D Gaussian-optimal classifier's AUC at that Cohen's d).

Usage::

    python -m analysis.gnn_pcr_gaussian_sweep_plot \
        --grid-dir experiments/gnn_pcr_gaussian_grid \
        --out experiments/gnn_pcr_gaussian_grid/auc_vs_snr.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _theoretical_auc(d: float) -> float:
    """AUC of a 1D Gaussian-optimal classifier at Cohen's d, Phi(d / sqrt(2))."""
    return 0.5 * (1.0 + math.erf(d / 2.0))


def _resolve_run_dir(tag_dir: Path) -> Path:
    """Find the single timestamped run dir under one grid-point tag directory.

    ``resolve_run_output_dir`` (``load_cohort.py``) always appends a
    ``<experiment_setup.name>_<timestamp>`` subdirectory under whatever
    ``--outdir`` was passed, even when that override already names a
    per-grid-point directory -- so each tag directory
    (``experiments/gnn_pcr_gaussian_grid/<tag>/``) contains exactly one such
    subdirectory, not the run's files directly.
    """
    (config_path,) = tag_dir.glob("*/config_used.json")
    return config_path.parent


def collect_run(tag_dir: Path) -> dict[str, float]:
    """Read one run's config + metrics into a single summary row."""
    run_dir = _resolve_run_dir(tag_dir)
    config = json.loads((run_dir / "config_used.json").read_text())
    params = config["model_params"]
    class0_mean = float(params["gnn_pcr_dummy_class0_mean"])
    class1_mean = float(params["gnn_pcr_dummy_class1_mean"])
    noise_std = float(params["gnn_pcr_dummy_noise_std"])
    cohens_d = (class1_mean - class0_mean) / noise_std

    model_dir = run_dir / config["experiment_setup"]["name"]
    metrics = json.loads((model_dir / "metrics.json").read_text())
    val_auc_mean = metrics["aggregated_metrics"]["auc"]["mean"]
    val_auc_std = metrics["aggregated_metrics"]["auc"]["std"]

    history = pd.read_csv(model_dir / "loss_history.csv")
    last_epoch = history["epoch"].max()
    train_auc_mean = history.loc[history["epoch"] == last_epoch, "train_auc"].mean()

    return {
        "run_dir": tag_dir.name,
        "class0_mean": class0_mean,
        "class1_mean": class1_mean,
        "noise_std": noise_std,
        "noise_seed": int(params["gnn_pcr_dummy_noise_seed"]),
        "cohens_d": cohens_d,
        "val_auc_mean": val_auc_mean,
        "val_auc_std": val_auc_std,
        "train_auc_mean": train_auc_mean,
        "theoretical_auc": _theoretical_auc(cohens_d),
    }


def main() -> None:
    """Collect every run under ``--grid-dir`` and write the summary CSV + plot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    run_dirs = sorted(p for p in args.grid_dir.iterdir() if p.is_dir())
    rows = [collect_run(run_dir) for run_dir in run_dirs]
    summary = pd.DataFrame(rows).sort_values(["noise_std", "cohens_d", "noise_seed"])
    summary.to_csv(args.out.with_suffix(".csv"), index=False)

    # noise_std == 1.0 is the main d-sweep; anything else (the scale-invariance
    # confirmatory point) is plotted separately and must not be averaged in.
    main_grid = summary[summary["noise_std"] == 1.0]
    scale_check = summary[summary["noise_std"] != 1.0]

    fig, ax = plt.subplots(figsize=(7, 5))
    d_curve = [i / 20.0 for i in range(0, 161)]
    ax.plot(
        d_curve,
        [_theoretical_auc(d) for d in d_curve],
        color="black",
        linestyle="--",
        label="theoretical 1D-optimal AUC",
    )

    grouped = main_grid.groupby("cohens_d")["val_auc_mean"]
    ax.errorbar(
        grouped.mean().index,
        grouped.mean().to_numpy(),
        yerr=[
            grouped.mean().to_numpy() - grouped.min().to_numpy(),
            grouped.max().to_numpy() - grouped.mean().to_numpy(),
        ],
        fmt="o",
        capsize=4,
        label="val AUC (mean, range over noise-draw seeds)",
    )
    ax.scatter(
        main_grid["cohens_d"],
        main_grid["train_auc_mean"],
        marker="x",
        color="tab:orange",
        label="train AUC (final epoch, per seed)",
    )
    if not scale_check.empty:
        ax.scatter(
            scale_check["cohens_d"],
            scale_check["val_auc_mean"],
            marker="s",
            color="tab:red",
            s=80,
            label="scale-invariance check (2x mean, 2x std, same d)",
        )

    ax.set_xlabel("Cohen's d (class separation / noise std)")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.4, 1.02)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("pCR-as-Gaussian learnability sweep")
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    plt.close(fig)
    print(f"Wrote {args.out} and {args.out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
