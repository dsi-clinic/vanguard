"""Render the end-to-end pipeline figure for the symposium poster.

Four real-data panels plus one schematic panel, left to right, on one case:

  1. ultrafast DCE-MRI      -- max-intensity projection of one postcontrast
                               frame over a thin z-slab
  2. vessel segmentation    -- the same image with the vessel support mask
  3. centerline extraction  -- the same image with the 1-voxel-wide skeleton
  4. vessel graph           -- a tight crop with the real 26-connected voxel
                               graph (nodes + edges) drawn on the MRI
  5. GNN -> pCR             -- schematic; nothing is measured here

Panels 1-3 share one field of view and one intensity window so the reader sees
the same anatomy gain structure at each stage; panel 4 zooms to the richest
branch point because a whole-breast view renders the graph as dots.

Loads one case's raw 4D DCE -- run via Slurm, not the head node.

Usage:
    python -m analysis.poster_pipeline_figure --case-id <id> \
        --centerline-root .../preprocessing_out_v5/centerlines \
        --dce-root .../preprocessing_out_v5/dce \
        --out-path presentations/symposium_poster_2026-08/pipeline.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from analysis.floor_activation_map import _resolve_case  # noqa: E402
from gnn.data_loader import _load_study_metadata  # noqa: E402
from gnn.raw_dce import discover_raw_dce_paths, load_raw_dce_series  # noqa: E402

# Poster palette: one accent for vessels, one for graph edges, so color always
# encodes "which object is this", never decoration.
VESSEL = "#f2c14e"
NODE = "#3fa7ff"
EDGE = "#e67e22"
BOX = "#7E1B14"

TITLE_SIZE = 27
LABEL_SIZE = 21

_OFFSETS_3D = [
    (dz, dy, dx)
    for dz in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dx in (-1, 0, 1)
    if not (dz == 0 and dy == 0 and dx == 0)
]


def _slab_bounds(skeleton: np.ndarray, half_width: int) -> tuple[int, int]:
    """Z-range of a thin slab centred on the slice carrying the most skeleton."""
    centre = int(np.argmax(skeleton.sum(axis=(1, 2))))
    return (
        max(centre - half_width, 0),
        min(centre + half_width + 1, skeleton.shape[0]),
    )


def _voxel_graph_in_slab(
    skeleton: np.ndarray, z0: int, z1: int
) -> tuple[set[tuple[int, int, int]], list, dict]:
    """The real 26-connected voxel graph restricted to one z-slab.

    Returns the node set, the edges already projected to (y, x) pairs, and each
    node's degree within the slab (used to find the busiest branch point).
    """
    coords = np.argwhere(skeleton)
    in_slab = coords[(coords[:, 0] >= z0) & (coords[:, 0] < z1)]
    nodes = {(int(z), int(y), int(x)) for z, y, x in in_slab}

    edges = []
    degree = dict.fromkeys(nodes, 0)
    for z, y, x in nodes:
        for dz, dy, dx in _OFFSETS_3D:
            neighbor = (z + dz, y + dy, x + dx)
            if neighbor in nodes and neighbor > (z, y, x):
                edges.append(((y, x), (neighbor[1], neighbor[2])))
                degree[(z, y, x)] += 1
                degree[neighbor] += 1
    return nodes, edges, degree


def _breast_bbox(
    support: np.ndarray, z0: int, z1: int, pad: int, keep_x: int
) -> tuple[int, ...]:
    """Padded (y0, y1, x0, x1) box around the vessel support inside the slab.

    Both breasts are in the field of view, which renders as a wide, short panel
    next to the near-square graph crop. The box is therefore narrowed to the
    left/right half containing ``keep_x`` (the branch point panel 4 zooms on),
    so every panel has a comparable aspect ratio.
    """
    coords = np.argwhere(support[z0:z1])
    y0, x0 = int(coords[:, 1].min()), int(coords[:, 2].min())
    y1, x1 = int(coords[:, 1].max()), int(coords[:, 2].max())
    midpoint = (x0 + x1) // 2
    if keep_x < midpoint:
        x1 = midpoint
    else:
        x0 = midpoint
    return (
        max(y0 - pad, 0),
        min(y1 + pad, support.shape[1]),
        max(x0 - pad, 0),
        min(x1 + pad, support.shape[2]),
    )


def _show_mri(ax: Axes, image: np.ndarray, box: tuple[int, ...]) -> None:
    """Grayscale MRI panel on one shared 1-99th-percentile window."""
    y0, y1, x0, x1 = box
    finite = image[np.isfinite(image)]
    ax.imshow(
        image,
        cmap="gray",
        origin="upper",
        vmin=float(np.percentile(finite, 1.0)),
        vmax=float(np.percentile(finite, 99.0)),
        extent=(x0, x1, y1, y0),
    )
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _overlay_mask(
    ax: Axes, mask_2d: np.ndarray, box: tuple[int, ...], color: str
) -> None:
    """Paint a boolean projection over the MRI, transparent where False."""
    y0, y1, x0, x1 = box
    rgba = np.zeros((*mask_2d.shape, 4))
    rgba[mask_2d] = matplotlib.colors.to_rgba(color)
    ax.imshow(rgba, origin="upper", extent=(x0, x1, y1, y0), interpolation="nearest")


def _model_panel(ax: Axes) -> None:
    """Schematic final stage: the graph goes into a GNN, a pCR score comes out."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # Match the near-square MRI panels so all five titles sit on one line.
    ax.set_box_aspect(1.0)
    for y, label in ((0.66, "GNN\nmessage passing"), (0.30, "pCR score")):
        ax.add_patch(
            FancyBboxPatch(
                (0.12, y - 0.10),
                0.76,
                0.20,
                boxstyle="round,pad=0.02,rounding_size=0.04",
                facecolor=BOX,
                edgecolor=BOX,
            )
        )
        ax.text(
            0.5,
            y,
            label,
            ha="center",
            va="center",
            color="white",
            fontsize=LABEL_SIZE,
        )
    ax.add_patch(
        FancyArrowPatch(
            (0.5, 0.54),
            (0.5, 0.42),
            arrowstyle="-|>",
            mutation_scale=22,
            color="#555555",
            lw=2,
        )
    )


def _between_panel_arrows(fig: Figure, axes: list[Axes]) -> None:
    """Grey arrows in figure coordinates, one between each pair of panels."""
    for left, right in zip(axes[:-1], axes[1:]):
        x_a = left.get_position().x1
        x_b = right.get_position().x0
        y = 0.5 * (left.get_position().y0 + left.get_position().y1)
        fig.add_artist(
            FancyArrowPatch(
                (x_a + 0.004, y),
                (x_b - 0.004, y),
                transform=fig.transFigure,
                arrowstyle="-|>",
                mutation_scale=26,
                color="#555555",
                lw=2.4,
            )
        )


def main() -> None:
    """Render the five-panel pipeline figure for one case."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", type=str, required=True)
    parser.add_argument("--centerline-root", type=Path, required=True)
    parser.add_argument("--dce-root", type=Path, required=True)
    parser.add_argument(
        "--time-index",
        type=int,
        default=-1,
        help="Frame shown in every MRI panel; default is the last (postcontrast).",
    )
    parser.add_argument(
        "--slab-half-width",
        type=int,
        default=4,
        help="Slices above/below the busiest slice projected into one 2D image.",
    )
    parser.add_argument(
        "--graph-window",
        type=int,
        default=90,
        help="Side length (voxels) of the panel-4 crop around the busiest branch.",
    )
    parser.add_argument("--out-path", type=Path, required=True)
    args = parser.parse_args()

    mask_path, support_path = _resolve_case(args.centerline_root, args.case_id)
    skeleton = np.load(mask_path).astype(bool)
    support = np.load(support_path).astype(bool)
    timepoints, baseline_frame_count, relative_enhancement = _load_study_metadata(
        args.case_id, mask_path.parent
    )
    dce_paths = discover_raw_dce_paths(args.dce_root, args.case_id, timepoints)
    dce_4d = load_raw_dce_series(dce_paths, expected_shape_zyx=support.shape)

    z0, z1 = _slab_bounds(skeleton, args.slab_half_width)
    nodes, edges, degree = _voxel_graph_in_slab(skeleton, z0, z1)
    print(
        f"case={args.case_id}  T={dce_4d.shape[0]}  "
        f"baseline_frames={baseline_frame_count}  "
        f"relative_enhancement={relative_enhancement}  "
        f"slab=z[{z0},{z1})  slab_nodes={len(nodes)}  slab_edges={len(edges)}  "
        f"skeleton_voxels={int(skeleton.sum())}  "
        f"support_voxels={int(support.sum())}"
    )

    branch = max(degree, key=degree.get)
    box = _breast_bbox(support, z0, z1, pad=12, keep_x=branch[2])
    y0, y1, x0, x1 = box
    frame = dce_4d[args.time_index, z0:z1, y0:y1, x0:x1].max(axis=0)
    support_2d = support[z0:z1, y0:y1, x0:x1].any(axis=0)
    skeleton_2d = skeleton[z0:z1, y0:y1, x0:x1].any(axis=0)

    fig, axes = plt.subplots(1, 5, figsize=(26, 6.2))
    titles = (
        "1. Ultrafast DCE-MRI",
        "2. Vessel segmentation",
        "3. Centerline extraction",
        "4. Vessel graph",
        "5. Prediction",
    )

    for ax in axes[:3]:
        _show_mri(ax, frame, box)
    _overlay_mask(axes[1], support_2d, box, VESSEL)
    _overlay_mask(axes[2], skeleton_2d, box, VESSEL)

    # Panel 4: crop to the busiest branch point so the graph fills the frame.
    half = args.graph_window // 2
    gy0, gy1 = max(branch[1] - half, 0), min(branch[1] + half, support.shape[1])
    gx0, gx1 = max(branch[2] - half, 0), min(branch[2] + half, support.shape[2])
    graph_box = (gy0, gy1, gx0, gx1)
    _show_mri(
        axes[3], dce_4d[args.time_index, z0:z1, gy0:gy1, gx0:gx1].max(axis=0), graph_box
    )
    for (ya, xa), (yb, xb) in edges:
        if gy0 <= ya < gy1 and gx0 <= xa < gx1 and gy0 <= yb < gy1 and gx0 <= xb < gx1:
            axes[3].plot([xa, xb], [ya, yb], color=EDGE, lw=3.0, zorder=1)
    inside = [n for n in nodes if gy0 <= n[1] < gy1 and gx0 <= n[2] < gx1]
    axes[3].scatter(
        [n[2] for n in inside],
        [n[1] for n in inside],
        s=34,
        color=NODE,
        zorder=2,
        edgecolors="none",
    )

    _model_panel(axes[4])

    captions = (
        f"{dce_4d.shape[0]} frames, {baseline_frame_count} precontrast",
        f"{int(support.sum()):,} vessel voxels",
        f"{int(skeleton.sum()):,} centerline voxels",
        f"{len(inside):,} nodes shown",
        "trained on 179 labelled cases",
    )
    for ax, title, caption in zip(axes, titles, captions):
        ax.set_title(title, fontsize=TITLE_SIZE, pad=18)
        ax.set_xlabel(caption, fontsize=LABEL_SIZE, labelpad=14)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.12, wspace=0.16)
    _between_panel_arrows(fig, list(axes))

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_path, dpi=160)
    plt.close(fig)
    print(f"wrote {args.out_path}")


if __name__ == "__main__":
    main()
