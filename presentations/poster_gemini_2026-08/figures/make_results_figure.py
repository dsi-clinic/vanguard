r"""Forest plot of pooled out-of-fold AUC for every UChicago pCR model arm.

Regenerates ``results_auc.pdf`` for the poster's Results block. Rerun this when
new results land -- the numbers are expected to move before final submission.

    micromamba run -n vanguard python figures/make_results_figure.py

Source of truth is the consolidated comparison table in the main repo, NOT a
copy pasted into this repo, so the figure cannot silently drift from the
experiment it reports:

    ~/vanguard/experiments/uchicago_all_models/all_models_auc_ci.csv

That table is pooled out-of-fold AUC over all 179 cases with a shared-draw
bootstrap 95% CI; see its README.md for how it was produced.

Drawn at final size (27.4 cm wide = \colwidth on the 36x24in poster) so the
point sizes here are the sizes the reader sees. Figure text scales with the
include width, not the poster size.

Colour encodes model class only -- maroon for GNN arms, grey for tabular
baselines -- with marker shape carrying the same distinction, so the two remain
separable under colour-vision deficiency and in greyscale print. Rank is not
encoded: an arm keeps its colour wherever it lands.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

AUC_CSV = Path.home() / "vanguard/experiments/uchicago_all_models/all_models_auc_ci.csv"
OUT_PDF = Path(__file__).parent / "results_auc.pdf"

# Poster palette (beamercolorthemeuchicago.sty) plus a neutral for the tabular
# baselines. L* is roughly 27 vs 50, a separation that survives every CVD type
# and greyscale, which is what makes the pairing legible without a hue cue.
MAROON = "#800000"
GREY = "#6B7280"
INK = "#3A3A3A"
CHANCE = 0.5


def main() -> None:
    """Render the forest plot to OUT_PDF."""
    results = pd.read_csv(AUC_CSV).sort_values("pooled_oof_auc")

    fig, ax = plt.subplots(figsize=(10.79, 4.35))  # 27.4 cm wide
    for i, row in enumerate(results.itertuples()):
        is_gnn = row.group == "GNN"
        color = MAROON if is_gnn else GREY
        ax.plot(
            [row.ci_lo, row.ci_hi],
            [i, i],
            color=color,
            linewidth=2.4,
            solid_capstyle="round",
        )
        ax.plot(
            row.pooled_oof_auc,
            i,
            marker="o" if is_gnn else "s",
            markersize=10,
            color=color,
            markeredgecolor="white",
            markeredgewidth=1.4,
            zorder=3,
        )
        ax.text(
            0.755,
            i,
            f"{row.pooled_oof_auc:.3f}",
            va="center",
            ha="right",
            fontsize=14,
            color=INK,
        )

    ax.axvline(CHANCE, color=INK, linestyle="--", linewidth=1.4, alpha=0.55, zorder=0)
    # Labelled at the foot of the line: the top-left of the axes is the only
    # region free of marks, and the legend claims it.
    ax.text(CHANCE, -0.55, " chance", fontsize=13, color=INK, alpha=0.8, va="center")

    ax.set_yticks(range(len(results)))
    ax.set_yticklabels(results["model"], fontsize=14)
    ax.set_xlabel("Pooled out-of-fold AUC (95% CI)", fontsize=15)
    ax.tick_params(axis="x", labelsize=14)
    ax.set_xlim(0.38, 0.76)
    ax.set_ylim(-0.7, len(results) - 0.3)

    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK)
    ax.grid(axis="x", color=INK, alpha=0.12, linewidth=0.8)
    ax.set_axisbelow(True)

    # Two classes are on the plot, so identity is never carried by colour alone:
    # the legend names them and the marker shape repeats the distinction.
    ax.plot([], [], "o", color=MAROON, markersize=10, label="GNN")
    ax.plot([], [], "s", color=GREY, markersize=10, label="tabular baseline")
    ax.legend(fontsize=14, frameon=False, loc="upper left", handletextpad=0.4)

    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
