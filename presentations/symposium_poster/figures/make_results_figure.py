"""Render the final UChicago temporal-transfer result for the poster.

The left panel shows all four matched downstream arms.  The right panel shows
the paired pretraining effect within each architecture, which is the deciding
comparison.  Values come from the frozen patient-level out-of-fold result:

``/gpfs/data/karczmar-lab/workspaces/spencervenancio/experiments/``
``uchicago_temporal_transfer/finetune_179_seed0_v1/metrics.json``

Run from ``presentations/symposium_poster`` with::

    micromamba run -n vanguard python figures/make_results_figure.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

DATA_CSV = Path(__file__).parent / "transfer_results.csv"
OUT_PDF = Path(__file__).parent / "results_auc.pdf"

MAROON = "#800000"
GREY = "#6B7280"
INK = "#282828"
LIGHT = "#D1D5DB"


def _interval(
    ax: plt.Axes,
    y: float,
    estimate: float,
    lo: float,
    hi: float,
    color: str,
    marker: str = "o",
) -> None:
    """Draw one horizontal estimate and confidence interval."""
    ax.plot([lo, hi], [y, y], color=color, lw=3.2, solid_capstyle="round")
    ax.plot(
        [estimate],
        [y],
        marker=marker,
        ms=10,
        color=color,
        markeredgecolor="white",
        markeredgewidth=1.2,
        zorder=3,
    )


def main() -> None:
    """Render the two-panel forest plot to ``results_auc.pdf``."""
    table = pd.read_csv(DATA_CSV)
    auc = table.loc[table["kind"] == "auc"].reset_index(drop=True)
    delta = table.loc[table["kind"] == "delta"].reset_index(drop=True)

    fig, (ax_auc, ax_delta) = plt.subplots(
        1,
        2,
        figsize=(10.79, 3.40),
        gridspec_kw={"width_ratios": (1.42, 1.0), "wspace": 0.52},
    )

    for y, row in enumerate(auc.itertuples(index=False)):
        color = MAROON if row.architecture == "message passing" else GREY
        marker = "o" if row.initialization == "forecast pretrained" else "s"
        _interval(ax_auc, y, row.estimate, row.ci_low, row.ci_high, color, marker)
        ax_auc.text(
            0.714,
            y,
            f"{row.estimate:.3f}",
            ha="right",
            va="center",
            fontsize=12,
            color=INK,
        )
    ax_auc.axvline(0.5, color=INK, ls="--", lw=1.4, alpha=0.55, zorder=0)
    ax_auc.text(0.5, -0.63, "chance", ha="center", fontsize=11, color=INK)
    ax_auc.set_yticks(range(len(auc)))
    ax_auc.set_yticklabels(auc["label"], fontsize=12)
    ax_auc.invert_yaxis()
    ax_auc.set_xlim(0.30, 0.72)
    ax_auc.set_xlabel("Patient-level pooled OOF AUC (95% CI)", fontsize=12)
    ax_auc.set_title("All four matched downstream arms", fontsize=14, weight="bold")

    for y, row in enumerate(delta.itertuples(index=False)):
        color = MAROON if row.architecture == "message passing" else GREY
        _interval(ax_delta, y, row.estimate, row.ci_low, row.ci_high, color)
        ax_delta.text(
            0.276,
            y,
            f"{row.estimate:+.3f}",
            ha="right",
            va="center",
            fontsize=12,
            color=INK,
        )
    ax_delta.axvline(0.0, color=INK, ls="--", lw=1.4, alpha=0.55, zorder=0)
    ax_delta.text(0.0, -0.53, "no effect", ha="center", fontsize=11, color=INK)
    ax_delta.set_yticks(range(len(delta)))
    ax_delta.set_yticklabels(delta["label"], fontsize=12)
    ax_delta.invert_yaxis()
    ax_delta.set_xlim(-0.13, 0.28)
    ax_delta.set_xlabel("Paired pretraining ΔAUC (95% CI)", fontsize=12)
    ax_delta.set_title("Paired pretraining effect", fontsize=14, weight="bold")

    for ax in (ax_auc, ax_delta):
        ax.tick_params(axis="x", labelsize=11)
        ax.grid(axis="x", color=LIGHT, lw=0.8, alpha=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(INK)

    fig.subplots_adjust(left=0.23, right=0.985, top=0.82, bottom=0.20, wspace=0.68)
    fig.savefig(OUT_PDF)
    plt.close(fig)
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
