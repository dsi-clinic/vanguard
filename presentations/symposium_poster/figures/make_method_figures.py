"""Render the poster's method schematics at their final printed width.

The diagrams use scientific descriptions rather than filenames or internal
configuration names.  Both PDFs are vector graphics sized for one poster
column.

Run from ``presentations/symposium_poster`` with::

    micromamba run -n vanguard python figures/make_method_figures.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT_DIR = Path(__file__).parent
MAROON = "#800000"
GREY = "#6B7280"
PALE = "#F5F1EF"
INK = "#282828"


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    label: str,
    *,
    face: str = MAROON,
    color: str = "white",
    size: float = 15,
    bold: bool = True,
) -> None:
    """Draw one rounded process box with centered text.

    ``pad`` inflates the drawn border outward from the given ``(width,
    height)`` rect without moving the centered text, which is what keeps the
    label clear of the box edge at poster print size.

    ``bold`` is an explicit choice, not inferred from ``face``: every process
    box in a given figure should share one font weight regardless of color,
    so a color-only distinction (e.g. upstream vs. in-scope stages) doesn't
    also read as two different typefaces.
    """
    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.16,rounding_size=0.10",
            facecolor=face,
            edgecolor=face,
            linewidth=1.4,
        )
    )
    ax.text(
        x + width / 2,
        y + height / 2,
        label,
        ha="center",
        va="center",
        color=color,
        fontsize=size,
        fontweight="bold" if bold else "normal",
        linespacing=1.15,
    )


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = GREY,
    style: str = "-",
) -> None:
    """Draw a process arrow."""
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=19,
            linewidth=2.0,
            linestyle=style,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def make_pipeline() -> None:
    """Render the imaging-to-prediction pipeline."""
    fig, ax = plt.subplots(figsize=(10.79, 1.85))
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0, 2.1)
    ax.axis("off")

    labels = (
        "Ultrafast breast\nMRI movie",
        "Vessel\nsegmentations",
        "One-voxel-wide\ncenterlines",
        "Vessel-network\ngraph",
        "pCR\nprediction",
    )
    # The first two stages -- the raw acquisition and its off-the-shelf
    # segmentation -- are upstream inputs this project consumes, not work done
    # here; greyed out to read as "given," with the maroon boxes starting at
    # the centerline this project actually builds on.
    faces = (GREY, GREY, MAROON, MAROON, MAROON)
    xs = (0.05, 2.55, 5.05, 7.55, 10.05)
    widths = (1.85, 1.85, 1.85, 1.85, 1.85)
    for x, width, label, face in zip(xs, widths, labels, faces):
        _box(ax, (x, 0.41), width, 1.30, label, face=face, size=14)
    for left_x, left_w, right_x in zip(xs[:-1], widths[:-1], xs[1:]):
        _arrow(ax, (left_x + left_w + 0.08, 1.06), (right_x - 0.08, 1.06))

    ax.text(
        3.47,
        0.18,
        "nnU-Net",
        ha="center",
        va="center",
        color=INK,
        fontsize=11,
    )
    ax.text(
        5.97,
        0.18,
        "tc4d",
        ha="center",
        va="center",
        color=INK,
        fontsize=11,
    )

    fig.tight_layout(pad=0.05)
    fig.savefig(OUT_DIR / "method_pipeline.pdf", bbox_inches="tight")
    plt.close(fig)


def make_pretraining_transfer() -> None:
    """Render forecasting pretraining followed by matched pCR fine-tuning."""
    fig, ax = plt.subplots(figsize=(10.79, 3.55))
    ax.set_xlim(0, 12.9)
    ax.set_ylim(0, 5.0)
    ax.axis("off")

    ax.text(
        0.10,
        4.55,
        "PRETRAINING  •  1,226 unlabeled UChicago exams",
        ha="left",
        va="center",
        fontsize=14,
        color=MAROON,
        fontweight="bold",
    )
    _box(ax, (0.10, 2.72), 2.75, 1.25, "Vessel curves +\nelapsed times", size=14)
    _box(ax, (3.72, 2.72), 2.75, 1.25, "Graph encoder", size=15)
    _box(ax, (7.34, 2.72), 2.75, 1.25, "Forecasting\ndecoder", size=14)
    _box(
        ax,
        (10.80, 2.72),
        1.95,
        1.25,
        "Future node\nenhancement",
        face=PALE,
        color=INK,
        size=12,
        bold=False,
    )
    _arrow(ax, (2.91, 3.34), (3.64, 3.34))
    _arrow(ax, (6.53, 3.34), (7.26, 3.34))
    _arrow(ax, (10.15, 3.34), (10.72, 3.34))

    ax.text(
        0.10,
        1.90,
        "FINE-TUNING  •  179 exams / 141 patients",
        ha="left",
        va="center",
        fontsize=14,
        color=MAROON,
        fontweight="bold",
    )
    _box(ax, (3.72, 0.22), 2.75, 1.25, "Same pretrained\ngraph encoder", size=14)
    _box(ax, (7.34, 0.22), 2.75, 1.25, "pCR prediction\nhead", size=14)
    _arrow(ax, (6.53, 0.84), (7.26, 0.84))

    _arrow(ax, (5.10, 2.64), (5.10, 1.55), color=MAROON)
    ax.text(
        5.30,
        2.30,
        "transfer weights",
        ha="left",
        va="center",
        fontsize=12,
        color=MAROON,
        fontweight="bold",
    )
    ax.text(
        10.55,
        0.84,
        "compared with the identical\nrandomly initialized encoder",
        ha="left",
        va="center",
        fontsize=12,
        color=INK,
        linespacing=1.15,
    )

    fig.tight_layout(pad=0.05)
    fig.savefig(OUT_DIR / "pretraining_transfer.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Write both method figures."""
    make_pipeline()
    make_pretraining_transfer()
    print(f"wrote {OUT_DIR / 'method_pipeline.pdf'}")
    print(f"wrote {OUT_DIR / 'pretraining_transfer.pdf'}")


if __name__ == "__main__":
    main()
