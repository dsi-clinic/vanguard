"""Compact visual QC for the HR-skeleton to UFAST mapping."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def _projection(volume_zyx: np.ndarray, axis: int) -> np.ndarray:
    """Return a maximum-intensity projection along one array axis."""
    return np.max(volume_zyx, axis=axis)


def write_temporal_mip(
    *,
    dce_dir: Path,
    exam_id: str,
    output_path: Path,
    shared_window: tuple[float, float],
) -> None:
    """Write plain orthogonal MIPs across every UFAST phase (no skeleton overlay).

    Takes the element-wise maximum across the full 4D series first, then a
    spatial maximum projection per view, so any vessel that only enhances in
    some phases still shows up (same combine-then-project convention as
    ``analysis/segment_ufast_vessels.py::_combine_vessel_phases``). Using a
    single phase (e.g. phase 0, the pre-contrast baseline) instead would miss
    vessels entirely, since they only become visible once contrast arrives.

    Same shared-window convention as ``write_mapping_qc``: the grayscale
    window comes from the complete source UFAST 4D series.
    """
    phase_paths = sorted(dce_dir.glob(f"{exam_id}_[0-9][0-9][0-9][0-9].nii.gz"))
    if not phase_paths:
        raise FileNotFoundError(f"no UFAST phase NIfTI files found under {dce_dir}")
    combined_zyx: np.ndarray | None = None
    for phase_path in phase_paths:
        phase_xyz = np.asarray(nib.load(str(phase_path)).dataobj, dtype=np.float32)
        phase_zyx = np.transpose(phase_xyz, (2, 1, 0))
        combined_zyx = (
            phase_zyx if combined_zyx is None else np.maximum(combined_zyx, phase_zyx)
        )

    lower, upper = (float(value) for value in shared_window)
    if not upper > lower:
        raise ValueError("shared 4D display window is degenerate")

    views = ((0, "axial"), (1, "coronal"), (2, "sagittal"))
    figure, axes = plt.subplots(1, len(views), figsize=(12, 4), constrained_layout=True)
    for axis, (projection_axis, title) in zip(axes, views, strict=True):
        background = _projection(combined_zyx, projection_axis)
        axis.imshow(background, cmap="gray", vmin=lower, vmax=upper, origin="lower")
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle(f"UFAST full temporal MIP ({len(phase_paths)} phases)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def write_mapping_qc(
    *,
    phase0_nifti: Path,
    skeleton_zyx: np.ndarray,
    output_path: Path,
    shared_window: tuple[float, float],
) -> None:
    """Overlay mapped skeleton projections on UFAST phase 0.

    The grayscale window comes from the complete source UFAST 4D series, not
    from phase 0, so the panel cannot hide temporal intensity differences via
    per-timepoint windowing.
    """
    phase0_xyz = np.asarray(nib.load(str(phase0_nifti)).dataobj, dtype=np.float32)
    phase0_zyx = np.transpose(phase0_xyz, (2, 1, 0))
    skeleton = np.asarray(skeleton_zyx, dtype=bool)
    if phase0_zyx.shape != skeleton.shape:
        raise ValueError(
            f"QC phase/skeleton shape mismatch: {phase0_zyx.shape} vs {skeleton.shape}"
        )
    lower, upper = (float(value) for value in shared_window)
    if not upper > lower:
        raise ValueError("shared 4D display window is degenerate")

    views = ((0, "axial"), (1, "coronal"), (2, "sagittal"))
    figure, axes = plt.subplots(1, len(views), figsize=(12, 4), constrained_layout=True)
    for axis, (projection_axis, title) in zip(axes, views, strict=True):
        background = _projection(phase0_zyx, projection_axis)
        overlay = _projection(skeleton, projection_axis)
        axis.imshow(background, cmap="gray", vmin=lower, vmax=upper, origin="lower")
        axis.imshow(
            np.ma.masked_where(~overlay, overlay),
            cmap="Reds",
            vmin=0,
            vmax=1,
            alpha=0.8,
            origin="lower",
        )
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle("HR TC4D skeleton mapped to UFAST phase 0")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
