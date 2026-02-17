"""Shared vessel MIP rendering helpers for debug and production pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

NDIM_3D = 3


def max_project_axis(volume_zyx: np.ndarray, *, axis: int) -> np.ndarray:
    """Return max projection for a 3D volume along one orthogonal axis."""
    if volume_zyx.ndim != NDIM_3D:
        raise ValueError(f"Expected 3D volume for MIP, got {volume_zyx.shape}")
    if axis == 0:
        return np.max(volume_zyx, axis=0)
    if axis == 1:
        return np.max(volume_zyx, axis=1)
    if axis == 2:
        return np.max(volume_zyx, axis=2)
    raise ValueError(f"Axis must be one of {{0,1,2}}, got {axis}")


def compute_hit_miss_vs_radiologist(
    mask_zyx: np.ndarray,
    radiologist_mask_zyx: np.ndarray | None,
) -> dict[str, float | int] | None:
    """Compute hit/miss against radiologist mask (recall-style)."""
    if radiologist_mask_zyx is None:
        return None
    pred = np.asarray(mask_zyx, dtype=bool)
    rad = np.asarray(radiologist_mask_zyx, dtype=bool)
    if pred.shape != rad.shape:
        raise ValueError(f"Shape mismatch for hit/miss: {pred.shape} vs {rad.shape}")

    rad_voxels = int(np.count_nonzero(rad))
    hits = int(np.count_nonzero(pred & rad))
    misses = int(rad_voxels - hits)
    if rad_voxels > 0:
        hit_rate = float(hits / float(rad_voxels))
        miss_rate = float(misses / float(rad_voxels))
    else:
        hit_rate = 0.0
        miss_rate = 0.0
    return {
        "radiologist_voxels": rad_voxels,
        "hits": hits,
        "misses": misses,
        "hit_rate": hit_rate,
        "miss_rate": miss_rate,
    }


def build_hit_miss_header(
    row_masks: list[tuple[str, np.ndarray]],
    radiologist_mask_zyx: np.ndarray | None,
) -> str:
    """Build one-line hit/miss summary for title header."""
    if radiologist_mask_zyx is None:
        return ""
    rad = np.asarray(radiologist_mask_zyx, dtype=bool)
    if not np.any(rad):
        return ""

    parts: list[str] = []
    for label, mask in row_masks:
        label_l = str(label).strip().lower()
        if "radiologist" in label_l:
            continue
        metrics = compute_hit_miss_vs_radiologist(mask, rad)
        if metrics is None:
            continue
        parts.append(
            f"{label_l} hit/miss={metrics['hit_rate']:.3f}/{metrics['miss_rate']:.3f} "
            f"({metrics['hits']}/{metrics['radiologist_voxels']})"
        )
    return " | ".join(parts)


def render_vessel_coverage_mip(
    *,
    row_masks: list[tuple[str, np.ndarray]],
    output_path: Path,
    case_label: str,
    title_prefix: str,
    radiologist_mask_zyx: np.ndarray | None = None,
    breast_mask_zyx: np.ndarray | None = None,
    tumor_mask_zyx: np.ndarray | None = None,
    peritumor_mask_zyx: np.ndarray | None = None,
    vessel_color: str = "#111827",
    dpi: int = 180,
) -> dict[str, Any]:
    """Render orthogonal MIPs for one or more vessel masks."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if not row_masks:
        raise ValueError("row_masks must be non-empty")
    shape_ref = tuple(int(v) for v in row_masks[0][1].shape)

    for label, row in row_masks:
        if tuple(int(v) for v in row.shape) != shape_ref:
            raise ValueError(
                "row mask shape mismatch for MIP rendering: "
                f"label={label}, shape={row.shape}, expected={shape_ref}"
            )

    optional_volumes = [
        ("radiologist", radiologist_mask_zyx),
        ("breast", breast_mask_zyx),
        ("tumor", tumor_mask_zyx),
        ("peritumor", peritumor_mask_zyx),
    ]
    for label, vol in optional_volumes:
        if vol is None:
            continue
        if tuple(int(v) for v in vol.shape) != shape_ref:
            raise ValueError(
                f"{label} mask shape mismatch for MIP rendering: "
                f"{vol.shape} vs {shape_ref}"
            )

    axis_specs = (
        (0, "xy mip (x→, y↑)", "x", "y"),
        (1, "xz mip (x→, z↑)", "x", "z"),
        (2, "yz mip (y→, z↑)", "y", "z"),
    )
    fig, axes = plt.subplots(
        nrows=len(row_masks),
        ncols=len(axis_specs),
        figsize=(14.0, 9.8),
        gridspec_kw={"wspace": 0.02, "hspace": 0.06},
        squeeze=False,
    )
    fig.patch.set_facecolor("white")

    for row_idx, (row_label, row_mask) in enumerate(row_masks):
        row_mask_bool = np.asarray(row_mask, dtype=bool)
        for col_idx, (axis, axis_title, x_label, y_label) in enumerate(axis_specs):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor("#f6f7f8")
            if row_idx == 0:
                ax.set_title(axis_title, fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f"{str(row_label).lower()}\n{y_label}", fontsize=10)
            else:
                ax.set_ylabel(y_label, fontsize=8)
            if row_idx == len(row_masks) - 1:
                ax.set_xlabel(x_label, fontsize=9)

            if breast_mask_zyx is not None:
                breast_proj = max_project_axis(
                    np.asarray(breast_mask_zyx, dtype=np.uint8),
                    axis=axis,
                )
                if np.any(breast_proj > 0):
                    breast_img = np.where(breast_proj > 0, 1.0, np.nan).astype(
                        np.float32,
                        copy=False,
                    )
                    ax.imshow(
                        breast_img,
                        cmap="Greys",
                        origin="lower",
                        interpolation="nearest",
                        vmin=0.0,
                        vmax=1.0,
                        alpha=0.12,
                    )
                    ax.contour(
                        breast_proj,
                        levels=[0.5],
                        colors="#6b7280",
                        linewidths=0.70,
                        alpha=0.75,
                        origin="lower",
                    )

            row_proj = max_project_axis(
                row_mask_bool.astype(np.uint8),
                axis=axis,
            )
            if np.any(row_proj > 0):
                row_img = np.zeros((*row_proj.shape, 4), dtype=np.float32)
                row_img[..., 0] = int(vessel_color[1:3], 16) / 255.0
                row_img[..., 1] = int(vessel_color[3:5], 16) / 255.0
                row_img[..., 2] = int(vessel_color[5:7], 16) / 255.0
                row_img[..., 3] = np.where(row_proj > 0, 0.62, 0.0).astype(
                    np.float32,
                    copy=False,
                )
                ax.imshow(
                    row_img,
                    origin="lower",
                    interpolation="nearest",
                )
            elif "missing" in str(row_label).lower():
                ax.text(
                    0.5,
                    0.5,
                    "n/a",
                    color="#9ca3af",
                    fontsize=10,
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )

            if peritumor_mask_zyx is not None:
                peritumor_proj = max_project_axis(
                    np.asarray(peritumor_mask_zyx, dtype=np.uint8),
                    axis=axis,
                )
                if np.any(peritumor_proj > 0):
                    ax.contour(
                        peritumor_proj,
                        levels=[0.5],
                        colors="#2dd4ff",
                        linewidths=0.65,
                        alpha=0.90,
                        origin="lower",
                    )
            if tumor_mask_zyx is not None:
                tumor_proj = max_project_axis(
                    np.asarray(tumor_mask_zyx, dtype=np.uint8),
                    axis=axis,
                )
                if np.any(tumor_proj > 0):
                    ax.contour(
                        tumor_proj,
                        levels=[0.5],
                        colors="#ff8c00",
                        linewidths=0.85,
                        alpha=0.95,
                        origin="lower",
                    )

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal", adjustable="box")

    legend_handles: list[Patch] = [
        Patch(facecolor=vessel_color, edgecolor=vessel_color, alpha=0.62, label="vessels"),
    ]
    if breast_mask_zyx is not None:
        legend_handles.append(
            Patch(facecolor="#9ca3af", edgecolor="#9ca3af", alpha=0.75, label="breast outline")
        )
    if peritumor_mask_zyx is not None:
        legend_handles.append(
            Patch(facecolor="#2dd4ff", edgecolor="#2dd4ff", alpha=0.90, label="peritumoral shell")
        )
    if tumor_mask_zyx is not None:
        legend_handles.append(
            Patch(facecolor="#ff8c00", edgecolor="#ff8c00", alpha=0.95, label="tumor")
        )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )

    hit_miss_header = build_hit_miss_header(row_masks, radiologist_mask_zyx)
    title = f"{title_prefix} | case: {case_label}"
    if hit_miss_header:
        title = f"{title}\n{hit_miss_header}"
    fig.suptitle(title, fontsize=11)

    fig.subplots_adjust(
        left=0.04,
        right=0.995,
        top=0.90,
        bottom=0.10,
        wspace=0.02,
        hspace=0.06,
    )
    fig.savefig(output_path, dpi=int(dpi))
    plt.close(fig)

    diagnostics: dict[str, Any] = {"title_hit_miss_header": hit_miss_header}
    if radiologist_mask_zyx is not None:
        metrics_by_row: dict[str, dict[str, float | int]] = {}
        for label, row_mask in row_masks:
            metrics = compute_hit_miss_vs_radiologist(row_mask, radiologist_mask_zyx)
            if metrics is None:
                continue
            metrics_by_row[str(label).lower()] = metrics
        diagnostics["radiologist_metrics_by_row"] = metrics_by_row
    return diagnostics
