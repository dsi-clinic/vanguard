#!/usr/bin/env python3
r"""Quantify why block-aware feature selection matters for pCR modeling.

This script loads a labeled feature table (the same CSV produced by
``train_tabular.py`` or ``scripts/feature_selection.py``) and reports three
diagnostics that justify selecting features per-block rather than globally:

1. **Block size imbalance** — counts how many numeric features belong to each
   canonical block.  When blocks differ by 10x+ in size, a global top-k
   selector is biased toward the largest block simply because it has more
   chances to correlate with the label by chance.

2. **Intra- vs inter-block correlation** — computes mean pairwise absolute
   Spearman correlation within each block and between blocks.  High
   intra-block correlation means a global selector picks many redundant
   features from the same block while crowding out unique signal from
   smaller blocks.

3. **Global-top-k survival simulation** — ranks all numeric features by
   univariate AUC and selects the top k.  Reports how many features each
   block retains and which blocks are completely eliminated.  Repeats the
   analysis with a block-aware budget that guarantees every block at least
   a minimum number of slots.

Usage::

    python scripts/analyze_block_selection_bias.py \
        experiments/some_run/features_labeled.csv \
        -o experiments/block_selection_analysis \
        --top-k 64

The script writes a summary CSV, a block-size bar chart, a correlation
heatmap, and a survival comparison chart.

Requires: pandas, numpy, scipy, matplotlib, seaborn.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from features import FEATURE_BLOCK_ORDER, feature_block_for_column  # noqa: E402

matplotlib.use("Agg")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_MIN_LABEL_CLASSES = 2

_NON_FEATURE_COLUMNS = frozenset(
    {
        "case_id",
        "dataset",
        "has_centerline_file",
        "site",
        "bilateral",
        "tumor_subtype",
        "menopausal_status",
        "pcr",
    }
)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "csv",
        type=Path,
        help="Path to the labeled feature CSV (must include a 'pcr' column).",
    )
    ap.add_argument(
        "-o",
        "--outdir",
        type=Path,
        default=Path("experiments/block_selection_analysis"),
        help="Output directory for figures and summary tables.",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=64,
        help="Number of features to keep in the global top-k simulation.",
    )
    ap.add_argument(
        "--label-col",
        type=str,
        default="pcr",
        help="Name of the binary label column.",
    )
    return ap.parse_args()


def _numeric_feature_columns(
    table: pd.DataFrame,
) -> list[str]:
    """Return numeric columns that belong to a known feature block."""
    return [
        col
        for col in table.columns
        if col not in _NON_FEATURE_COLUMNS
        and table[col].dtype in ("float64", "float32", "int64", "int32")
        and feature_block_for_column(col) is not None
    ]


def _assign_blocks(columns: list[str]) -> dict[str, str]:
    """Map each column to its canonical block name."""
    return {col: feature_block_for_column(col) for col in columns}


# ── Diagnostic 1: block size imbalance ──────────────────────────────────


def block_size_table(col_to_block: dict[str, str]) -> pd.DataFrame:
    """Count features per block and compute share of total."""
    counts = (
        pd.Series(col_to_block)
        .value_counts()
        .reindex(FEATURE_BLOCK_ORDER)
        .fillna(0)
        .astype(int)
    )
    share = counts / counts.sum()
    return pd.DataFrame({"n_features": counts, "share": share})


def plot_block_sizes(size_df: pd.DataFrame, outpath: Path) -> None:
    """Bar chart of feature counts per block."""
    fig, ax = plt.subplots(figsize=(7, 4))
    size_df["n_features"].plot.bar(ax=ax, color="#4C72B0", edgecolor="white")
    for i, (_, row) in enumerate(size_df.iterrows()):
        ax.text(
            i,
            row["n_features"] + 1,
            str(int(row["n_features"])),
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )
    ax.set_ylabel("Number of numeric features")
    ax.set_title("Feature block sizes (imbalance diagnostic)")
    ax.set_xlabel("")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved block size chart to %s", outpath)


# ── Diagnostic 2: intra- vs inter-block correlation ─────────────────────


def correlation_summary(
    table: pd.DataFrame,
    feature_cols: list[str],
    col_to_block: dict[str, str],
) -> pd.DataFrame:
    """Compute mean |Spearman rho| within and between blocks."""
    blocks = sorted(set(col_to_block.values()))
    block_cols = {b: [c for c in feature_cols if col_to_block[c] == b] for b in blocks}

    numeric = table[feature_cols].apply(pd.to_numeric, errors="coerce")
    rho_matrix, _ = spearmanr(numeric.values, nan_policy="omit")
    if rho_matrix.ndim == 0:
        rho_matrix = np.array([[rho_matrix]])
    rho_abs = np.abs(rho_matrix)
    col_idx = {c: i for i, c in enumerate(feature_cols)}

    rows = []
    for b1 in blocks:
        for b2 in blocks:
            idxs1 = [col_idx[c] for c in block_cols[b1]]
            idxs2 = [col_idx[c] for c in block_cols[b2]]
            if not idxs1 or not idxs2:
                rows.append(
                    {"block_a": b1, "block_b": b2, "mean_abs_rho": np.nan, "n_pairs": 0}
                )
                continue
            sub = rho_abs[np.ix_(idxs1, idxs2)]
            if b1 == b2:
                # exclude diagonal
                mask = ~np.eye(sub.shape[0], sub.shape[1], dtype=bool)
                vals = sub[mask]
            else:
                vals = sub.ravel()
            rows.append(
                {
                    "block_a": b1,
                    "block_b": b2,
                    "mean_abs_rho": float(np.nanmean(vals))
                    if len(vals) > 0
                    else np.nan,
                    "n_pairs": len(vals),
                }
            )
    return pd.DataFrame(rows)


def plot_correlation_heatmap(corr_df: pd.DataFrame, outpath: Path) -> None:
    """Heatmap of mean |rho| between blocks."""
    pivot = corr_df.pivot_table(
        index="block_a", columns="block_b", values="mean_abs_rho"
    )
    pivot = pivot.reindex(index=FEATURE_BLOCK_ORDER, columns=FEATURE_BLOCK_ORDER)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Mean |Spearman ρ| within and between feature blocks")
    plt.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved correlation heatmap to %s", outpath)


# ── Diagnostic 3: global-top-k survival simulation ──────────────────────


def univariate_auc_ranking(
    table: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
) -> pd.DataFrame:
    """Rank features by univariate AUC (label vs. each feature)."""
    _min_samples_auc = 10
    label = pd.to_numeric(table[label_col], errors="coerce").to_numpy()
    results = []
    for col in feature_cols:
        vals = pd.to_numeric(table[col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(vals) & np.isfinite(label)
        finite_labels = label[mask]
        finite_vals = vals[mask]
        if (
            mask.sum() < _min_samples_auc
            or len(np.unique(finite_labels)) < _MIN_LABEL_CLASSES
        ):
            results.append({"feature": col, "auc": np.nan})
            continue
        try:
            auc_raw = float(roc_auc_score(finite_labels, finite_vals))
            auc = max(auc_raw, 1.0 - auc_raw)
        except Exception:
            auc = np.nan
        results.append({"feature": col, "auc": auc})
    ranking = (
        pd.DataFrame(results).sort_values("auc", ascending=False).reset_index(drop=True)
    )
    ranking["rank"] = range(1, len(ranking) + 1)
    return ranking


def simulate_selection(
    ranking: pd.DataFrame,
    col_to_block: dict[str, str],
    top_k: int,
) -> pd.DataFrame:
    """Simulate global vs block-aware top-k selection.

    Global: take the top-k features regardless of block.
    Block-aware: allocate floor(k / n_blocks) to each block, give remainder
    to blocks with the best top feature.
    """
    ranking = ranking.copy()
    ranking["block"] = ranking["feature"].map(col_to_block)
    valid = ranking.dropna(subset=["auc", "block"])

    # global selection
    global_selected = valid.head(top_k)
    global_counts = (
        global_selected["block"]
        .value_counts()
        .reindex(FEATURE_BLOCK_ORDER)
        .fillna(0)
        .astype(int)
    )

    # block-aware selection
    n_blocks = len(FEATURE_BLOCK_ORDER)
    per_block = top_k // n_blocks
    remainder = top_k - per_block * n_blocks
    block_aware_parts = []
    block_best_auc = {}
    for block in FEATURE_BLOCK_ORDER:
        block_feats = valid[valid["block"] == block]
        selected = block_feats.head(per_block)
        block_aware_parts.append(selected)
        block_best_auc[block] = (
            float(selected["auc"].max()) if len(selected) > 0 else 0.0
        )

    # distribute remainder to blocks with best top feature
    remainder_blocks = sorted(block_best_auc, key=block_best_auc.get, reverse=True)[
        :remainder
    ]
    for block in remainder_blocks:
        block_feats = valid[valid["block"] == block]
        already = per_block
        extra = block_feats.iloc[already : already + 1]
        block_aware_parts.append(extra)

    block_aware_selected = pd.concat(block_aware_parts)
    block_aware_counts = (
        block_aware_selected["block"]
        .value_counts()
        .reindex(FEATURE_BLOCK_ORDER)
        .fillna(0)
        .astype(int)
    )

    return pd.DataFrame(
        {
            "block": FEATURE_BLOCK_ORDER,
            "global_topk": global_counts.to_numpy(),
            "block_aware": block_aware_counts.to_numpy(),
        }
    )


def plot_survival_comparison(
    survival_df: pd.DataFrame,
    top_k: int,
    outpath: Path,
) -> None:
    """Side-by-side bar chart of block survival under each strategy."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(survival_df))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        survival_df["global_topk"],
        width,
        label="Global top-k",
        color="#C44E52",
        edgecolor="white",
    )
    bars2 = ax.bar(
        x + width / 2,
        survival_df["block_aware"],
        width,
        label="Block-aware",
        color="#4C72B0",
        edgecolor="white",
    )

    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.3,
                    str(int(h)),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(survival_df["block"], rotation=30, ha="right")
    ax.set_ylabel(f"Features selected (k={top_k})")
    ax.set_title("Block survival: global top-k vs block-aware selection")
    ax.legend()
    ax.set_ylim(0, survival_df[["global_topk", "block_aware"]].max().max() + 5)
    plt.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved survival comparison to %s", outpath)


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    """Run all three diagnostics and save outputs."""
    args = _parse_args()
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading features from %s", args.csv)
    table = pd.read_csv(args.csv)
    feature_cols = _numeric_feature_columns(table)
    col_to_block = _assign_blocks(feature_cols)
    logger.info(
        "Found %d numeric features across %d blocks",
        len(feature_cols),
        len(set(col_to_block.values())),
    )

    # 1. Block size imbalance
    sizes = block_size_table(col_to_block)
    sizes.to_csv(outdir / "block_sizes.csv")
    plot_block_sizes(sizes, outdir / "block_sizes.png")
    logger.info("Block sizes:\n%s", sizes.to_string())

    # 2. Intra- vs inter-block correlation
    corr = correlation_summary(table, feature_cols, col_to_block)
    corr.to_csv(outdir / "block_correlation.csv", index=False)
    plot_correlation_heatmap(corr, outdir / "block_correlation_heatmap.png")

    intra = corr[corr["block_a"] == corr["block_b"]]
    inter = corr[corr["block_a"] != corr["block_b"]]
    logger.info(
        "Mean intra-block |rho| = %.3f, mean inter-block |rho| = %.3f",
        intra["mean_abs_rho"].mean(),
        inter["mean_abs_rho"].mean(),
    )

    # 3. Global-top-k survival simulation
    ranking = univariate_auc_ranking(table, feature_cols, args.label_col)
    ranking["block"] = ranking["feature"].map(col_to_block)
    ranking.to_csv(outdir / "feature_auc_ranking.csv", index=False)

    survival = simulate_selection(ranking, col_to_block, args.top_k)
    survival.to_csv(outdir / "selection_survival.csv", index=False)
    plot_survival_comparison(survival, args.top_k, outdir / "selection_survival.png")

    eliminated = survival[survival["global_topk"] == 0]["block"].tolist()
    if eliminated:
        logger.info("Blocks ELIMINATED by global top-k: %s", eliminated)
    else:
        logger.info("No blocks fully eliminated by global top-k.")

    # Combined summary
    summary = sizes.copy()
    summary["global_topk_kept"] = survival.set_index("block")["global_topk"]
    summary["block_aware_kept"] = survival.set_index("block")["block_aware"]
    intra_map = dict(zip(intra["block_a"], intra["mean_abs_rho"]))
    summary["intra_block_mean_abs_rho"] = summary.index.map(intra_map)
    summary.to_csv(outdir / "block_selection_summary.csv")
    logger.info("Full summary:\n%s", summary.to_string())
    logger.info("All outputs saved to %s", outdir)


if __name__ == "__main__":
    main()
