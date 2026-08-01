"""Meta-level diagnostic plots for the GNN tier sweep (many runs -> one result).

Reads every ``predictions.csv`` under an out-root laid out as
``{arm}_seed{0..4}/*/*/predictions.csv`` (the layout
``submit_gnn_uchicago_tier_sweep.slurm`` writes) and renders a single
diagnostic figure that answers the questions you actually ask of a sweep:

  A. AUC by arm, mean +/- std across seeds (the "error-bar plot") vs chance /
     tabular references.
  B. AUC by fold -- is the variance seed-driven or fold-driven?
  C. AUC by sub-source -- where (if anywhere) is the signal?
  D. Score separation -- do predicted probabilities separate the classes at all,
     or are they collapsed? (The "is there any signal" panel.)
  E. Paired dAUC by seed x fold -- the Gate G0 / G1 deltas, as distributions.
  F. Prediction vs graph size -- is the model keying on vessel-graph size rather
     than biology? Needs the cache's graph_qc.csv (--graph-qc).

Summarizes an already-computed sweep (no training) -- hence analysis/. Reads
small CSVs only; safe on the head node.

Usage:
    python -m analysis.gnn_tier_sweep_plot \
        --out-root /gpfs/.../experiments/gnn_uchicago_tier_sweep \
        --graph-qc /ess/scratch/.../gnn_cache_voxel_richfeat/processed/graph_qc.csv \
        --out analysis/figures/gnn_uchicago_tier_sweep_diagnostics.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ARMS = ("baseline", "tier0", "tier1")
SEEDS = range(5)
_BOTH_CLASSES = 2

# Okabe-Ito (colorblind-safe), assigned in fixed arm order. Baseline is neutral
# gray to read as the reference; tier0/tier1 are the two intervention hues.
ARM_COLOR = {"baseline": "#999999", "tier0": "#0072B2", "tier1": "#D55E00"}
CHANCE = 0.5
TABULAR_REF = 0.540  # XGBoost tabular baseline (PLAN_advanced_modeling.md)


def _sub(df: pd.DataFrame, col: str, val: object) -> pd.DataFrame:
    """Rows where ``df[col] == val`` (avoids DataFrame.query's @-scope gotcha)."""
    return df[df[col] == val]


def _mstd(vals: object) -> tuple[float, float]:
    """(mean, std) ignoring NaN, or (nan, nan) when every value is NaN.

    All-NaN happens e.g. for a single-class sub-source whose AUC is undefined;
    handled explicitly to avoid a numpy empty-slice warning.
    """
    a = np.asarray(vals, dtype=float)
    if np.all(np.isnan(a)):
        return float("nan"), float("nan")
    return float(np.nanmean(a)), float(np.nanstd(a))


def _auc(df: pd.DataFrame) -> float:
    """OOF AUC; NaN when a (sub)set has only one class present."""
    if df["y_true"].nunique() < _BOTH_CLASSES:
        return float("nan")
    return roc_auc_score(df["y_true"], df["y_prob"])


def load_predictions(out_root: Path) -> dict[tuple[str, int], pd.DataFrame]:
    """One predictions.csv per (arm, seed); exactly one match required."""
    frames: dict[tuple[str, int], pd.DataFrame] = {}
    for arm in ARMS:
        for seed in SEEDS:
            matches = sorted(out_root.glob(f"{arm}_seed{seed}/*/*/predictions.csv"))
            if len(matches) != 1:
                raise RuntimeError(f"{arm}_seed{seed}: expected 1 file, got {matches}")
            frames[(arm, seed)] = pd.read_csv(matches[0])
    return frames


def _style(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", color="0.9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _refs(ax: plt.Axes, tabular: bool = True) -> None:
    ax.axhline(CHANCE, color="0.4", ls="--", lw=1, zorder=1)
    ax.text(
        0.995,
        CHANCE,
        " chance",
        va="bottom",
        ha="right",
        color="0.4",
        fontsize=8,
        transform=ax.get_yaxis_transform(),
    )
    if tabular:
        ax.axhline(TABULAR_REF, color="0.6", ls=":", lw=1, zorder=1)
        ax.text(
            0.995,
            TABULAR_REF,
            " tabular 0.54",
            va="bottom",
            ha="right",
            color="0.6",
            fontsize=8,
            transform=ax.get_yaxis_transform(),
        )


def panel_auc_by_arm(ax: plt.Axes, frames: dict) -> None:
    """A. OOF AUC per arm: mean +/- std across seeds, seeds overlaid."""
    for x, arm in enumerate(ARMS):
        vals = np.array([_auc(frames[(arm, s)]) for s in SEEDS])
        m, sd = _mstd(vals)
        ax.errorbar(
            x,
            m,
            yerr=sd,
            fmt="o",
            color=ARM_COLOR[arm],
            ms=10,
            capsize=5,
            lw=2,
            zorder=3,
        )
        ax.scatter(
            np.full_like(vals, x) + 0.14,
            vals,
            s=22,
            color=ARM_COLOR[arm],
            alpha=0.55,
            zorder=2,
        )
        ax.text(
            x,
            m + sd + 0.006,
            f"{m:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=ARM_COLOR[arm],
            fontweight="bold",
        )
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels(ARMS)
    ax.set_ylabel("OOF AUC")
    ax.set_title("A. AUC by arm (mean ± std over 5 seeds)", fontsize=10)
    ax.set_xlim(-0.5, len(ARMS) - 0.3)
    _refs(ax)
    _style(ax)


def panel_auc_by_fold(ax: plt.Axes, frames: dict) -> None:
    """B. AUC by fold, grouped by arm -- exposes fold-driven variance."""
    folds = sorted(next(iter(frames.values()))["fold"].unique())
    width = 0.25
    for ai, arm in enumerate(ARMS):
        means, sds = [], []
        for f in folds:
            vals = [_auc(_sub(frames[(arm, s)], "fold", f)) for s in SEEDS]
            m, sd = _mstd(vals)
            means.append(m)
            sds.append(sd)
        xs = np.arange(len(folds)) + (ai - 1) * width
        ax.bar(
            xs,
            means,
            width,
            yerr=sds,
            color=ARM_COLOR[arm],
            label=arm,
            capsize=3,
            zorder=2,
            ecolor="0.3",
        )
    ax.set_xticks(range(len(folds)))
    ax.set_xticklabels([f"fold {f}" for f in folds])
    ax.set_ylabel("AUC (mean ± std over seeds)")
    ax.set_title(
        "B. AUC by fold — variance is fold-driven, not arm-driven", fontsize=10
    )
    ax.axhline(CHANCE, color="0.4", ls="--", lw=1, zorder=1)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center")
    ax.set_ylim(0, 0.9)
    _style(ax)


def panel_auc_by_subsource(ax: plt.Axes, frames: dict) -> None:
    """C. Per-sub-source AUC, grouped by arm."""
    subs = sorted(next(iter(frames.values()))["dataset"].unique())
    width = 0.25
    notes = []
    for ai, arm in enumerate(ARMS):
        means, sds = [], []
        for sub in subs:
            vals = [_auc(_sub(frames[(arm, s)], "dataset", sub)) for s in SEEDS]
            m, sd = _mstd(vals)
            means.append(m)
            sds.append(sd)
        xs = np.arange(len(subs)) + (ai - 1) * width
        ax.bar(
            xs,
            np.nan_to_num(means),
            width,
            yerr=sds,
            color=ARM_COLOR[arm],
            label=arm,
            capsize=3,
            zorder=2,
            ecolor="0.3",
        )
    # annotate any single-class (undefined-AUC) sub-source
    ref = next(iter(frames.values()))
    for xi, sub in enumerate(subs):
        d = _sub(ref, "dataset", sub)
        if d["y_true"].nunique() < _BOTH_CLASSES:
            ax.text(
                xi,
                0.02,
                f"{sub}\n(single-class,\nAUC undefined)",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#B22222",
            )
            notes.append(sub)
    ax.set_xticks(range(len(subs)))
    ax.set_xticklabels(subs, fontsize=8)
    ax.set_ylabel("AUC (mean ± std over seeds)")
    ax.set_title("C. AUC by sub-source", fontsize=10)
    ax.axhline(CHANCE, color="0.4", ls="--", lw=1, zorder=1)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center")
    ax.set_ylim(0, 0.9)
    _style(ax)


def panel_score_separation(ax: plt.Axes, frames: dict, arm: str = "tier1") -> None:
    """D. Pooled predicted-probability distributions by true class."""
    pooled = pd.concat([frames[(arm, s)] for s in SEEDS])
    bins = np.linspace(0, 1, 26)
    for cls, color, lab in ((0, "#0072B2", "no pCR"), (1, "#D55E00", "pCR")):
        v = pooled.loc[pooled["y_true"] == cls, "y_prob"]
        ax.hist(v, bins=bins, density=True, color=color, alpha=0.5, label=lab, zorder=2)
        ax.axvline(v.mean(), color=color, ls="-", lw=2, zorder=3)
    ax.set_xlabel("predicted P(pCR)")
    ax.set_ylabel("density")
    ax.set_title(
        f"D. Score separation ({arm}, pooled) — overlap = no signal", fontsize=10
    )
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    ax.grid(True, axis="both", color="0.92", lw=0.7, zorder=0)


def panel_paired_delta(ax: plt.Axes, frames: dict) -> None:
    """E. Paired dAUC by seed x fold for the two gate comparisons."""
    folds = sorted(next(iter(frames.values()))["fold"].unique())
    comps = [
        ("tier0", "baseline", "G0:\ntier0−baseline"),
        ("tier1", "tier0", "G1:\ntier1−tier0"),
    ]
    for xi, (a, b, _lab) in enumerate(comps):
        deltas = []
        for s in SEEDS:
            for f in folds:
                da = _auc(_sub(frames[(a, s)], "fold", f))
                db = _auc(_sub(frames[(b, s)], "fold", f))
                if not (np.isnan(da) or np.isnan(db)):
                    deltas.append(da - db)
        deltas = np.array(deltas)
        jitter = xi + (np.random.default_rng(0).random(len(deltas)) - 0.5) * 0.25
        colors = np.where(deltas >= 0, "#0072B2", "#D55E00")
        ax.scatter(jitter, deltas, c=colors, s=26, alpha=0.7, zorder=2)
        m = deltas.mean()
        ax.hlines(m, xi - 0.2, xi + 0.2, color="black", lw=2.5, zorder=3)
        ax.text(
            xi,
            ax.get_ylim()[1],
            f"mean {m:+.3f}\n{(deltas>0).mean()*100:.0f}% > 0",
            ha="center",
            va="top",
            fontsize=8,
        )
    ax.axhline(0, color="0.4", ls="--", lw=1, zorder=1)
    ax.set_xticks(range(len(comps)))
    ax.set_xticklabels([c[2] for c in comps], fontsize=8)
    ax.set_ylabel("ΔAUC (per seed×fold)")
    ax.set_title("E. Paired ΔAUC — the gate deltas", fontsize=10)
    _style(ax)


def panel_pred_vs_size(
    ax: plt.Axes, frames: dict, qc: pd.DataFrame, arm: str = "tier1"
) -> None:
    """F. Predicted prob vs graph size -- is the model reading size?"""
    pooled = (
        pd.concat([frames[(arm, s)] for s in SEEDS])
        .groupby("case_id", as_index=False)
        .agg(y_prob=("y_prob", "mean"), y_true=("y_true", "first"))
    )
    merged = pooled.merge(qc[["case_id", "num_nodes"]], on="case_id", how="inner")
    for cls, color, lab in ((0, "#0072B2", "no pCR"), (1, "#D55E00", "pCR")):
        m = merged[merged["y_true"] == cls]
        ax.scatter(
            m["num_nodes"],
            m["y_prob"],
            s=22,
            color=color,
            alpha=0.6,
            label=lab,
            zorder=2,
        )
    r_ps = np.corrcoef(merged["num_nodes"], merged["y_prob"])[0, 1]
    r_pt = np.corrcoef(qc["num_nodes"], qc["pcr"])[0, 1]
    ax.set_xlabel("graph size (num_nodes)")
    ax.set_ylabel(f"mean predicted P(pCR) ({arm})")
    ax.set_title(
        f"F. Prediction vs graph size — "
        f"corr(pred,size)={r_ps:+.2f}, corr(pcr,size)={r_pt:+.2f}",
        fontsize=10,
    )
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    ax.grid(True, axis="both", color="0.92", lw=0.7, zorder=0)


def panel_before_after(
    ax: plt.Axes, frames: dict, compare: dict, this_label: str, compare_label: str
) -> None:
    """Pooled OOF AUC by arm for two sweeps (e.g. unfloored vs floored)."""
    width = 0.38
    for gi, (fr, lab, color) in enumerate(
        ((compare, compare_label, "#E69F00"), (frames, this_label, "#0072B2"))
    ):
        means, sds = [], []
        for arm in ARMS:
            vals = np.array([_auc(fr[(arm, s)]) for s in SEEDS])
            m, sd = _mstd(vals)
            means.append(m)
            sds.append(sd)
        xs = np.arange(len(ARMS)) + (gi - 0.5) * width
        ax.bar(
            xs,
            means,
            width,
            yerr=sds,
            color=color,
            label=lab,
            capsize=3,
            ecolor="0.3",
            zorder=2,
        )
        for x, m in zip(xs, means):
            ax.text(
                x,
                m + 0.008,
                f"{m:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=color,
            )
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels(ARMS)
    ax.set_ylabel("pooled OOF AUC (mean ± std)")
    ax.set_title(
        "C. Feature conditioning: before vs after (pooled, all cases)", fontsize=10
    )
    ax.axhline(CHANCE, color="0.4", ls="--", lw=1, zorder=1)
    ax.legend(frameon=False, fontsize=8, loc="upper center", ncol=2)
    ax.set_ylim(0, 0.75)
    _style(ax)


def main() -> None:
    """Render the 6-panel tier-sweep diagnostic figure from --out-root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--graph-qc",
        type=Path,
        default=None,
        help="cache processed/graph_qc.csv for panel F (num_nodes)",
    )
    parser.add_argument(
        "--compare-root",
        type=Path,
        default=None,
        help="a second sweep to overlay in panel C as a before/after "
        "(e.g. the unfloored sweep); replaces the per-sub-source panel",
    )
    parser.add_argument("--this-label", type=str, default="this")
    parser.add_argument("--compare-label", type=str, default="compare")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("analysis/figures/gnn_tier_sweep_diagnostics.png"),
    )
    args = parser.parse_args()

    frames = load_predictions(args.out_root)
    qc = pd.read_csv(args.graph_qc) if args.graph_qc else None
    compare = load_predictions(args.compare_root) if args.compare_root else None

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    panel_auc_by_arm(axes[0, 0], frames)
    panel_auc_by_fold(axes[0, 1], frames)
    if compare is not None:
        panel_before_after(
            axes[0, 2], frames, compare, args.this_label, args.compare_label
        )
    else:
        panel_auc_by_subsource(axes[0, 2], frames)
    panel_score_separation(axes[1, 0], frames)
    panel_paired_delta(axes[1, 1], frames)
    if qc is not None:
        panel_pred_vs_size(axes[1, 2], frames, qc)
    else:
        axes[1, 2].axis("off")
        axes[1, 2].text(
            0.5,
            0.5,
            "panel F needs --graph-qc",
            ha="center",
            va="center",
            fontsize=9,
            color="0.5",
        )

    fig.suptitle(
        f"GNN tier sweep diagnostics — {args.out_root.name} "
        f"(3 arms × 5 seeds, 7-feature voxel, predefined folds)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
