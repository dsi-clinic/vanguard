r"""Render the poster's two schematic figures at poster scale.

The talk versions of these diagrams (``presentations/gnn_methods_2026-07`` and
``analysis/plot_contrast_pretrain_flow.py``) were drawn for a projected slide:
their labels come out around 10pt once scaled into a poster column, which is
unreadable at a few feet. These are the same two diagrams redrawn at their
final printed size -- one poster column wide -- with type sized to be read
standing in front of the board.

The rule this module follows, and the reason it exists: draw the figure at the
width it will actually be printed (``--width``, default one 0.32 column of a
36in poster) and set fonts in points at that size. Then ``\\includegraphics``
scales 1:1 and nothing shrinks.

  graph_representations.png  voxel / segment / junction node modes
  contrast_pretrain_arch.png shared encoder, forecast head, pCR head

Both are schematics: no measured quantity appears in either.

Usage:
    python -m analysis.poster_schematic_figures \
        --out-dir presentations/symposium_poster_2026-08
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

# Poster palette (uwtheme.sty): near-black nodes, Badger Red vessels, cool grey
# for everything that is scaffolding. Colour marks what a mark *is*, never
# decoration -- red is always a vessel segment.
NODE = "#101820"
VOXEL = "#B1B3B3"
SEGMENT = "#C5050C"
ADJACENCY = "#9A9A9A"
TITLE = "#9B0000"
CALLOUT_FILL = "#F2F0ED"
CALLOUT_EDGE = "#CFCFCF"
BOX = "#7E1B14"
ARROW = "#6E6E6E"

# A four-armed skeleton: one junction J0, four endpoints E1..E4, four segments.
# Coordinates are illustrative, not anatomical.
J0 = (0.0, 0.0)
ENDPOINTS = ((-1.95, 0.10), (1.15, 1.60), (1.95, -0.05), (0.60, -1.75))


def _segment_points(
    a: tuple[float, float], b: tuple[float, float], n: int
) -> list[tuple[float, float]]:
    """``n`` evenly spaced points from ``a`` to ``b``, endpoints included."""
    return [
        (a[0] + (b[0] - a[0]) * i / (n - 1), a[1] + (b[1] - a[1]) * i / (n - 1))
        for i in range(n)
    ]


def _panel(ax: Axes, title: str, lines: str, font: float) -> None:
    """Panel heading and the two-line what-is-a-node/edge caption."""
    ax.set_xlim(-3.05, 3.05)
    ax.set_ylim(-3.45, 3.55)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(
        0,
        3.62,
        title,
        ha="center",
        va="top",
        fontsize=font,
        fontweight="bold",
        color=TITLE,
    )
    ax.text(
        0,
        3.02,
        lines,
        ha="center",
        va="top",
        fontsize=font * 0.80,
        color="#333333",
        linespacing=1.3,
    )


def _callout(ax: Axes, text: str, font: float) -> None:
    """Italic takeaway box under a panel."""
    ax.text(
        0,
        -2.75,
        text,
        ha="center",
        va="top",
        fontsize=font * 0.80,
        style="italic",
        color="#222222",
        linespacing=1.3,
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": CALLOUT_FILL,
            "edgecolor": CALLOUT_EDGE,
            "linewidth": 1.2,
        },
    )


def _label(
    ax: Axes,
    xy: tuple[float, float],
    text: str,
    font: float,
    dx: float = 0.0,
    dy: float = 0.0,
) -> None:
    """Maths-style node/segment label offset from its mark."""
    ax.text(
        xy[0] + dx,
        xy[1] + dy,
        text,
        ha="center",
        va="center",
        fontsize=font * 0.80,
        color=NODE,
    )


def _graph_representations(out_path: Path, width: float, font: float) -> None:
    """Voxel -> junction -> segment, the three deterministic transforms.

    Redraws the reference diagram from the July methods talk
    (``presentations/gnn_methods_standalone_2026-07/figures/three_representations_v2.png``),
    which is a cropped raster with no generator in the repo -- so it could
    neither be recoloured nor rescaled. This is that design as code, in the
    poster palette, at the printed width.
    """
    font = font * 0.74  # three panels share one column width
    fig, axes = plt.subplots(1, 3, figsize=(width, width * 0.53))
    arms = [_segment_points(J0, e, 8) for e in ENDPOINTS]
    end_labels = ("E_1", "E_2", "E_3", "E_4")
    seg_labels = ("S_1", "S_2", "S_3", "S_4")

    def _outward(e: tuple[float, float], d: float) -> tuple[float, float]:
        n = (e[0] ** 2 + e[1] ** 2) ** 0.5
        return (e[0] / n * d, e[1] / n * d)

    def _perp(e: tuple[float, float], d: float) -> tuple[float, float]:
        n = (e[0] ** 2 + e[1] ** 2) ** 0.5
        return (-e[1] / n * d, e[0] / n * d)

    # 1. Voxel graph -- every skeleton voxel is a node.
    ax = axes[0]
    _panel(
        ax, "1. Voxel graph", "nodes = skeleton voxels\nedges = voxel adjacency", font
    )
    for pts in arms:
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=VOXEL, lw=font * 0.13, zorder=1)
        ax.scatter(
            xs[1:-1], ys[1:-1], s=font * 7, color=VOXEL, zorder=2, edgecolors="none"
        )
    ax.scatter(*zip(*ENDPOINTS), s=font * 22, color=NODE, zorder=3, edgecolors="none")
    ax.scatter([J0[0]], [J0[1]], s=font * 30, color=NODE, zorder=3, edgecolors="none")
    for e, lab in zip(ENDPOINTS, end_labels):
        dx, dy = _outward(e, 0.55)
        _label(ax, e, f"${lab}$", font, dx, dy)
    _label(ax, J0, "$J_0$", font, -0.10, -0.52)
    _callout(ax, "exact centerline geometry;\nmany degree-2 nodes", font)

    # 2. Junction graph -- degree-2 chains contracted; segments become edges.
    ax = axes[1]
    _panel(
        ax,
        "2. Junction graph",
        "nodes = endpoints and junctions\nedges = vessel segments",
        font,
    )
    for e in ENDPOINTS:
        ax.plot(
            [J0[0], e[0]],
            [J0[1], e[1]],
            color=SEGMENT,
            lw=font * 0.30,
            solid_capstyle="round",
            zorder=1,
        )
    ax.scatter(*zip(*ENDPOINTS), s=font * 22, color=NODE, zorder=3, edgecolors="none")
    ax.scatter([J0[0]], [J0[1]], s=font * 30, color=NODE, zorder=3, edgecolors="none")
    for e, elab, slab in zip(ENDPOINTS, end_labels, seg_labels):
        dx, dy = _outward(e, 0.55)
        _label(ax, e, f"${elab}$", font, dx, dy)
        px, py = _perp(e, 0.48)
        ax.text(
            0.52 * e[0] + px,
            0.52 * e[1] + py,
            f"${slab}$",
            ha="center",
            va="center",
            fontsize=font * 0.84,
            color=SEGMENT,
            fontweight="bold",
        )
    _label(ax, J0, "$J_0$", font, -0.10, -0.52)
    _callout(
        ax, "branching topology is explicit;\nsegment features live on edges", font
    )

    # 3. Segment graph -- the line graph; each segment becomes one node.
    ax = axes[2]
    _panel(
        ax,
        "3. Segment graph",
        "nodes = vessel segments\nedges = shared-junction adjacency",
        font,
    )
    boxes = ((-1.45, 0.35), (0.35, 1.75), (1.45, 0.35), (0.35, -1.30))
    for i, bi in enumerate(boxes):
        for bj in boxes[i + 1 :]:
            ax.plot(
                [bi[0], bj[0]],
                [bi[1], bj[1]],
                color=ADJACENCY,
                lw=font * 0.07,
                ls=(0, (3, 3)),
                zorder=0,
            )
    for (bx, by), slab in zip(boxes, seg_labels):
        ax.add_patch(
            FancyBboxPatch(
                (bx - 0.62, by - 0.50),
                1.24,
                1.00,
                boxstyle="round,pad=0.06,rounding_size=0.16",
                facecolor="white",
                edgecolor=NODE,
                linewidth=1.4,
                zorder=2,
            )
        )
        ax.plot(
            [bx - 0.34, bx + 0.34],
            [by - 0.18, by + 0.18],
            color=SEGMENT,
            lw=font * 0.16,
            solid_capstyle="round",
            zorder=3,
        )
        ax.scatter(
            [bx - 0.34, bx + 0.34],
            [by - 0.18, by + 0.18],
            s=font * 9,
            color=NODE,
            zorder=4,
            edgecolors="none",
        )
        ax.text(
            bx - 0.40,
            by + 0.30,
            f"${slab}$",
            ha="center",
            va="center",
            fontsize=font * 0.74,
            color=SEGMENT,
            fontweight="bold",
        )
    _callout(ax, "compact; segment summaries\nbecome node features", font)

    # Transform arrows between panels, in figure coordinates.
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.03, wspace=0.10)
    for left, right, label in (
        (axes[0], axes[1], "contract\ndegree-2 chains"),
        (axes[1], axes[2], "take the\nline graph"),
    ):
        x_a, x_b = left.get_position().x1, right.get_position().x0
        y = 0.56
        fig.add_artist(
            FancyArrowPatch(
                (x_a - 0.012, y),
                (x_b + 0.012, y),
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=font * 1.2,
                lw=font * 0.09,
                color=ARROW,
            )
        )
        fig.text(
            (x_a + x_b) / 2,
            y + 0.035,
            label,
            ha="center",
            va="bottom",
            fontsize=font * 0.74,
            color="#333333",
            linespacing=1.25,
        )

    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


def _box(
    ax: Axes, x: float, y: float, w: float, h: float, label: str, font: float
) -> None:
    """Rounded maroon box with centred white text."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            facecolor=BOX,
            edgecolor=BOX,
            zorder=2,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=font,
        color="white",
        linespacing=1.3,
        zorder=3,
    )


def _contrast_pretrain_arch(out_path: Path, width: float, font: float) -> None:
    """Shared encoder feeding a forecast head (pretraining) and a pCR head."""
    height = width * 0.40
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 13.0)
    ax.set_ylim(0, 13.0 * height / width)
    ax.axis("off")

    _box(ax, 0.20, 1.85, 3.30, 1.50, "Contrast\ntime series", font)
    _box(ax, 4.55, 1.85, 3.60, 1.50, "Shared GNN encoder\nper-frame GNN + GRU", font)
    _box(ax, 9.25, 3.30, 3.55, 1.25, "Forecast head", font)
    _box(ax, 9.25, 0.35, 3.55, 1.25, "pCR head", font)

    for p0, p1, rad, dotted in (
        ((3.50, 2.60), (4.55, 2.60), 0.0, False),
        ((8.15, 2.80), (9.25, 3.75), -0.25, False),
        ((8.15, 2.40), (9.25, 1.15), 0.25, True),
    ):
        ax.add_patch(
            FancyArrowPatch(
                p0,
                p1,
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",
                mutation_scale=font * 1.5,
                linewidth=font * 0.09,
                linestyle=(0, (1.6, 2.0)) if dotted else "-",
                color=ARROW,
                zorder=1,
            )
        )

    for x, y, label in ((8.70, 3.62, "pretraining"), (8.70, 1.28, "fine-tuning")):
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=font * 0.85,
            color="#333333",
            bbox={"facecolor": "#EDEDED", "edgecolor": "none", "pad": 3.0},
            zorder=4,
        )

    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    """Write both schematic figures at poster scale."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--width",
        type=float,
        default=10.4,
        help="Printed width in inches; one 0.32 column of the 36in poster.",
    )
    parser.add_argument(
        "--font",
        type=float,
        default=22.0,
        help="Base label size in points at the printed width.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _graph_representations(
        args.out_dir / "graph_representations.png", args.width, args.font
    )
    _contrast_pretrain_arch(
        args.out_dir / "contrast_pretrain_arch.png", args.width, args.font
    )


if __name__ == "__main__":
    main()
