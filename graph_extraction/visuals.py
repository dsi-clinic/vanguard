"""Visualization helpers for 3D vessel skeletons."""

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

# Definition of relative coordinates of all 26 possible neighbors around
# each voxel. Generate all combinations of shifts by -1, 0, +1 in each
# dimension and then remove 0, 0, 0 (itself).
_OFFSETS_3D = np.array(
    [
        (dz, dy, dx)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if not (dz == 0 and dy == 0 and dx == 0)
    ],
    dtype=np.int64,
)


def edges_to_segments(edges: np.ndarray) -> np.ndarray:
    """Convert 3D bitmask to an array of line segments.

    Each segment is defined by a pair of (x, y, z) coordinates.
    """
    Z, H, W = edges.shape
    segments = []
    for k in range(Z):
        for i in range(H):
            for j in range(W):
                mask = edges[k, i, j]
                if mask == 0:
                    continue
                for b in range(26):
                    if (mask >> b) & 1:
                        dz, dy, dx = _OFFSETS_3D[b]
                        nk, ni, nj = k + dz, i + dy, j + dx
                        # Avoid duplicate segments: only add if neighbor is ahead in lexicographic order
                        if 0 <= nk < Z and 0 <= ni < H and 0 <= nj < W:
                            if (nk, ni, nj) > (k, i, j):
                                segments.append(((j, i, k), (nj, ni, nk)))  # (x,y,z)
    return np.array(segments)


def plot_skeleton3d(segments: np.ndarray) -> go.Figure:
    """Visualize a 3D vessel skeleton as line segments in a clean, consistent Plotly style.

    Args:
        segments (np.ndarray): Array of shape (N, 2, 3), where each pair of points
            represents the endpoints of a vessel segment.

    Returns:
        go.Figure: A ready-to-display Plotly 3D figure.
    """
    # Style constants
    vessel_color = "#9a14b5"
    background_color = "#DFD8DE"
    line_width = 3
    opacity = 1.0

    # Extract endpoints
    x0, y0, z0 = segments[:, 0, 0], segments[:, 0, 1], segments[:, 0, 2]
    x1, y1, z1 = segments[:, 1, 0], segments[:, 1, 1], segments[:, 1, 2]

    # Create line traces (each segment = one line)
    lines = [
        go.Scatter3d(
            x=[x0[i], x1[i]],
            y=[y0[i], y1[i]],
            z=[z0[i], z1[i]],
            mode="lines",
            line={"color": vessel_color, "width": line_width},
            opacity=opacity,
            showlegend=False,
        )
        for i in range(len(segments))
    ]

    # Minimalist 3D layout
    axis_style = {
        "showbackground": True,
        "backgroundcolor": background_color,
        "showgrid": False,
        "zeroline": False,
        "showticklabels": False,
        "title": "",
    }

    fig = go.Figure(data=lines)
    fig.update_layout(
        scene={
            "xaxis": axis_style,
            "yaxis": axis_style,
            "zaxis": axis_style,
            "aspectmode": "data",
            "bgcolor": "white",
        },
        paper_bgcolor=background_color,
        margin={"l": 0, "r": 0, "b": 0, "t": 0},
        scene_camera={"eye": {"x": 1.3, "y": 1.3, "z": 1.0}},
    )

    return fig


def plot_skeleton_projections(skeleton: np.ndarray) -> None:
    """Display 2D orthogonal projections (XY, XZ, YZ) of a 3D skeleton volume.

    Uses a consistent visual style matching the 3D visualization.

    Args:
        skeleton (np.ndarray): 3D binary or integer array (Z, H, W)
            representing the skeletonized volume.
    """
    # Compute projections
    proj_xy = (skeleton > 0).any(axis=0)  # top view
    proj_xz = (skeleton > 0).any(axis=1)  # front view
    proj_yz = (skeleton > 0).any(axis=2)  # side view

    # Style constants
    background_color = "#DFD8DE"

    # Create figure
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), facecolor=background_color)
    titles = [
        "XY projection (top view)",
        "XZ projection (front view)",
        "YZ projection (side view)",
    ]

    for ax, proj, title in zip(axs, [proj_xy, proj_xz, proj_yz], titles):
        ax.imshow(proj, cmap="Purples", vmin=0, vmax=1)
        ax.set_title(title, color="black", fontsize=12, pad=10)
        ax.axis("off")
        ax.set_facecolor(background_color)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.02)
    fig.patch.set_facecolor(background_color)
    plt.show()
