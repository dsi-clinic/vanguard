"""Run baseline 3D and full 4D extraction with one exam-level skeleton output.

This CLI is intended for quick experimentation:
- Baseline: 3D graph-pruning skeletonization on one selected timepoint.
- Proposed: 4D center-manifold extraction across all timepoints, then collapse to
  one consensus 3D skeleton for the exam.
- Outputs: masks and one XY/XZ/YZ comparison plot for side-by-side inspection.
"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from graph_extraction.processing import (
    DEFAULT_SEGMENTATION_DIR,
)
from graph_extraction.processing import (
    baseline_3d_mask as _shared_baseline_3d_mask,
)
from graph_extraction.processing import (
    collapse_4d_to_exam_skeleton as _shared_collapse_4d_to_exam_skeleton,
)
from graph_extraction.processing import (
    discover_study_timepoints as _shared_discover_study_timepoints,
)
from graph_extraction.processing import (
    load_time_series_from_files as _shared_load_time_series_from_files,
)

DEFAULT_TUMOR_MASK_DIR = Path(
    "/net/projects2/vanguard/MAMA-MIA-syn60868042/segmentations/expert"
)
NDIM_3D = 3
NDIM_4D = 4
NDIM_5D = 5


def _load_time_series_from_files(paths: list[Path], npy_channel: int) -> np.ndarray:
    """Load and stack separate timepoint files into `(t, z, y, x)`."""
    return _shared_load_time_series_from_files(paths, npy_channel=npy_channel)


def _discover_study_timepoints(
    input_dir: Path, study_id: str
) -> tuple[list[Path], list[int]]:
    """Discover and sort per-timepoint segmentation files for one study."""
    return _shared_discover_study_timepoints(input_dir=input_dir, study_id=study_id)


def _load_time_series_from_single_npy(
    path: Path, layout: str, npy_channel: int
) -> np.ndarray:
    """Load a single NPY into `(t, z, y, x)` based on explicit layout."""
    arr = np.load(path)

    if layout == "tzyx":
        if arr.ndim != NDIM_4D:
            raise ValueError(
                f"layout=tzyx expects 4D array, got shape {arr.shape} from {path}"
            )
        return arr.astype(np.float32, copy=False)

    if layout == "ctzyx":
        if arr.ndim != NDIM_5D:
            raise ValueError(
                f"layout=ctzyx expects 5D array, got shape {arr.shape} from {path}"
            )
        if npy_channel < 0 or npy_channel >= arr.shape[0]:
            raise ValueError(
                f"Requested channel {npy_channel} but array has {arr.shape[0]} channels."
            )
        return arr[npy_channel].astype(np.float32, copy=False)

    if layout == "tczyx":
        if arr.ndim != NDIM_5D:
            raise ValueError(
                f"layout=tczyx expects 5D array, got shape {arr.shape} from {path}"
            )
        if npy_channel < 0 or npy_channel >= arr.shape[1]:
            raise ValueError(
                f"Requested channel {npy_channel} but array has {arr.shape[1]} channels."
            )
        return arr[:, npy_channel].astype(np.float32, copy=False)

    raise ValueError(f"Unsupported layout: {layout}")


def _compute_projections(
    mask_zyx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute XY, XZ, and YZ max-intensity style binary projections."""
    proj_xy = (mask_zyx > 0).any(axis=0)  # (y, x)
    proj_xz = (mask_zyx > 0).any(axis=1)  # (z, x)
    proj_yz = (mask_zyx > 0).any(axis=2)  # (z, y)
    return proj_xy, proj_xz, proj_yz


def _save_projection_comparison(
    projections_3d: tuple[np.ndarray, np.ndarray, np.ndarray],
    projections_4d: tuple[np.ndarray, np.ndarray, np.ndarray],
    output_path: Path,
    title: str,
) -> None:
    """Save a 2x3 figure comparing 3D baseline and 4D method projections."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(title, fontsize=13)

    names = ("XY", "XZ", "YZ")
    for col, (name, img) in enumerate(zip(names, projections_3d)):
        axes[0, col].imshow(img.astype(np.uint8), cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"3D {name}")
        axes[0, col].axis("off")

    for col, (name, img) in enumerate(zip(names, projections_4d)):
        axes[1, col].imshow(img.astype(np.uint8), cmap="gray", vmin=0, vmax=1)
        axes[1, col].set_title(f"4D {name}")
        axes[1, col].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _subsample_xyz(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, max_points: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uniformly subsample coordinate arrays to at most `max_points`."""
    n = int(x.size)
    if n <= max_points:
        return x, y, z
    keep = np.linspace(0, n - 1, num=max_points, dtype=np.int64)
    return x[keep], y[keep], z[keep]


def _load_tumor_mask_npy(
    path: Path,
    *,
    expected_shape_zyx: tuple[int, int, int],
    t_dim: int,
    npy_channel: int,
    threshold: float,
) -> np.ndarray:
    """Load tumor segmentation as a 3D bool mask aligned to `(z, y, x)`.

    Supported NPY layouts:
    - 3D `(z, y, x)`
    - 4D `(t, z, y, x)` -> exam union across time
    - 4D `(c, z, y, x)` -> channel selected by `npy_channel`
    - 5D `(t, c, z, y, x)` -> exam union across time for selected channel
    - 5D `(c, t, z, y, x)` -> selected channel then exam union across time
    """
    arr = np.load(path)
    expected = tuple(int(v) for v in expected_shape_zyx)

    if arr.ndim == NDIM_3D:
        if tuple(arr.shape) != expected:
            raise ValueError(
                f"Tumor mask shape mismatch: expected {expected}, got {arr.shape} from {path}"
            )
        mask = arr > threshold
    elif arr.ndim == NDIM_4D:
        if tuple(arr.shape[1:]) != expected:
            raise ValueError(
                "Unsupported 4D tumor mask layout. Expected either (t,z,y,x) "
                f"or (c,z,y,x) with zyx={expected}, got {arr.shape} from {path}"
            )
        if arr.shape[0] == t_dim:
            mask = np.any(arr > threshold, axis=0)
        else:
            if npy_channel < 0 or npy_channel >= arr.shape[0]:
                raise ValueError(
                    f"Requested tumor npy-channel={npy_channel} but array has {arr.shape[0]} channels."
                )
            mask = arr[npy_channel] > threshold
    elif arr.ndim == NDIM_5D:
        if tuple(arr.shape[2:]) != expected:
            raise ValueError(
                "Unsupported 5D tumor mask layout. Expected (t,c,z,y,x) or (c,t,z,y,x) "
                f"with zyx={expected}, got {arr.shape} from {path}"
            )
        if arr.shape[0] == t_dim:
            if npy_channel < 0 or npy_channel >= arr.shape[1]:
                raise ValueError(
                    f"Requested tumor npy-channel={npy_channel} but array has {arr.shape[1]} channels."
                )
            mask = np.any(arr[:, npy_channel] > threshold, axis=0)
        else:
            if npy_channel < 0 or npy_channel >= arr.shape[0]:
                raise ValueError(
                    f"Requested tumor npy-channel={npy_channel} but array has {arr.shape[0]} channels."
                )
            mask = np.any(arr[npy_channel] > threshold, axis=0)
    else:
        raise ValueError(
            f"Unsupported tumor mask ndim={arr.ndim} from {path}. Use 3D/4D/5D NPY."
        )

    mask = mask.astype(bool, copy=False)
    if not np.any(mask):
        raise ValueError(
            f"Tumor mask is empty after thresholding ({threshold}) from {path}. "
            "Provide the correct file/layout/threshold."
        )
    return mask


def _load_tumor_mask_volume(
    path: Path,
    *,
    expected_shape_zyx: tuple[int, int, int],
    t_dim: int,
    npy_channel: int,
    threshold: float,
) -> np.ndarray:
    """Load tumor segmentation from NPY or NIfTI-like volume as 3D bool mask."""
    path_str = str(path).lower()
    if path_str.endswith(".npy"):
        return _load_tumor_mask_npy(
            path,
            expected_shape_zyx=expected_shape_zyx,
            t_dim=t_dim,
            npy_channel=npy_channel,
            threshold=threshold,
        )

    if (
        path_str.endswith(".nii")
        or path_str.endswith(".nii.gz")
        or path_str.endswith(".nrrd")
    ):
        import SimpleITK as sitk

        arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(
            np.float32, copy=False
        )
        if arr.ndim != NDIM_3D:
            raise ValueError(
                f"Tumor mask NIfTI/NRRD must be 3D, got shape {arr.shape} from {path}"
            )

        expected = tuple(int(v) for v in expected_shape_zyx)
        candidates = (
            ("zyx", arr),
            ("yxz", np.transpose(arr, (1, 2, 0))),
            ("xyz", np.transpose(arr, (2, 1, 0))),
        )
        selected: np.ndarray | None = None
        selected_layout: str | None = None
        for layout_name, candidate in candidates:
            if tuple(candidate.shape) == expected:
                selected = candidate
                selected_layout = layout_name
                break

        if selected is None:
            raise ValueError(
                "Tumor mask shape mismatch for NIfTI/NRRD input. "
                f"Expected {expected}, got {arr.shape} (and common transposes) from {path}"
            )

        print(
            f"[info] Tumor mask layout resolved as '{selected_layout}' for {path.name}"
        )
        mask = selected > threshold
        if not np.any(mask):
            raise ValueError(
                f"Tumor mask is empty after thresholding ({threshold}) from {path}. "
                "Provide the correct file/threshold."
            )
        return mask.astype(bool, copy=False)

    raise ValueError(
        f"Unsupported tumor mask format for {path}. Supported: .npy, .nii, .nii.gz, .nrrd"
    )


def _save_rotating_3d_comparison_mp4(
    mask_3d: np.ndarray,
    mask_4d_skeleton: np.ndarray,
    output_path: Path,
    *,
    tumor_mask_zyx: np.ndarray | None = None,
    study_label: str | None = None,
    n_frames: int = 50,
    fps: int = 24,
    elev: float = 20.0,
    marker_size: float = 1.0,
) -> None:
    """Save a rotating side-by-side 3D comparison as MP4."""
    import matplotlib.pyplot as plt
    from matplotlib import animation

    if mask_3d.ndim != NDIM_3D or mask_4d_skeleton.ndim != NDIM_3D:
        raise ValueError(
            f"Expected two 3D masks, got {mask_3d.shape=} and {mask_4d_skeleton.shape=}"
        )

    z0, y0, x0 = np.nonzero(mask_3d)
    z1, y1, x1 = np.nonzero(mask_4d_skeleton)
    if x0.size == 0 or x1.size == 0:
        raise ValueError("Cannot render 3D comparison because one mask is empty.")

    if "ffmpeg" not in animation.writers.list():
        raise RuntimeError("Matplotlib ffmpeg writer is unavailable. Install ffmpeg.")

    tx_fill = ty_fill = tz_fill = None
    tx_shell = ty_shell = tz_shell = None
    if tumor_mask_zyx is not None:
        from scipy import ndimage

        if tumor_mask_zyx.ndim != NDIM_3D:
            raise ValueError(f"Tumor mask must be 3D, got shape {tumor_mask_zyx.shape}")
        if tuple(tumor_mask_zyx.shape) != tuple(mask_3d.shape):
            raise ValueError(
                "Tumor mask shape mismatch with skeleton masks: "
                f"{tumor_mask_zyx.shape} vs {mask_3d.shape}"
            )
        if not np.any(tumor_mask_zyx):
            raise ValueError("Tumor mask is empty; cannot overlay.")

        tz_all, ty_all, tx_all = np.nonzero(tumor_mask_zyx)
        tx_fill, ty_fill, tz_fill = _subsample_xyz(
            tx_all.astype(np.float32),
            ty_all.astype(np.float32),
            tz_all.astype(np.float32),
            max_points=6000,
        )

        tumor_eroded = ndimage.binary_erosion(
            tumor_mask_zyx, structure=np.ones((3, 3, 3), dtype=bool), border_value=0
        )
        tumor_shell = tumor_mask_zyx & ~tumor_eroded
        if np.any(tumor_shell):
            tz_s, ty_s, tx_s = np.nonzero(tumor_shell)
        else:
            tz_s, ty_s, tx_s = tz_all, ty_all, tx_all
        tx_shell, ty_shell, tz_shell = _subsample_xyz(
            tx_s.astype(np.float32),
            ty_s.astype(np.float32),
            tz_s.astype(np.float32),
            max_points=9000,
        )

    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    ax_l = fig.add_subplot(1, 2, 1, projection="3d")
    ax_r = fig.add_subplot(1, 2, 2, projection="3d")

    x_parts = [x0.astype(np.float32), x1.astype(np.float32)]
    y_parts = [y0.astype(np.float32), y1.astype(np.float32)]
    z_parts = [z0.astype(np.float32), z1.astype(np.float32)]
    if tx_shell is not None and ty_shell is not None and tz_shell is not None:
        x_parts.append(tx_shell)
        y_parts.append(ty_shell)
        z_parts.append(tz_shell)
    x_all = np.concatenate(x_parts)
    y_all = np.concatenate(y_parts)
    z_all = np.concatenate(z_parts)
    x_min, x_max = float(x_all.min()), float(x_all.max())
    y_min, y_max = float(y_all.min()), float(y_all.max())
    z_min, z_max = float(z_all.min()), float(z_all.max())

    for ax, title in (
        (ax_l, "3D baseline (selected timepoint)"),
        (ax_r, "4D exam-level skeleton"),
    ):
        ax.set_title(title, fontsize=11)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_box_aspect(
            (x_max - x_min + 1.0, y_max - y_min + 1.0, z_max - z_min + 1.0)
        )

    if (
        tx_fill is not None
        and ty_fill is not None
        and tz_fill is not None
        and tx_shell is not None
        and ty_shell is not None
        and tz_shell is not None
    ):
        for ax in (ax_l, ax_r):
            # Semi-transparent interior + brighter shell for see-through context.
            ax.scatter(
                tx_fill,
                ty_fill,
                tz_fill,
                s=0.5,
                c="#f4a261",
                alpha=0.035,
                marker="o",
            )
            ax.scatter(
                tx_shell,
                ty_shell,
                tz_shell,
                s=0.8,
                c="#ff8c00",
                alpha=0.16,
                marker="o",
            )

    ax_l.scatter(x0, y0, z0, s=marker_size, c="#1f77b4", alpha=0.85, marker="o")
    ax_r.scatter(x1, y1, z1, s=marker_size, c="#d62728", alpha=0.85, marker="o")
    title = "3D baseline vs 4D exam-level skeleton"
    if study_label is not None:
        title = f"{title} | Study: {study_label}"
    if tumor_mask_zyx is not None:
        title += " (+ tumor overlay)"
    fig.suptitle(title, fontsize=13)

    def _update(frame_idx: int) -> tuple[()]:
        azim = 360.0 * float(frame_idx) / float(n_frames)
        ax_l.view_init(elev=elev, azim=azim)
        ax_r.view_init(elev=elev, azim=azim)
        return ()

    anim = animation.FuncAnimation(
        fig, _update, frames=n_frames, interval=1000.0 / float(fps), blit=False
    )
    writer = animation.FFMpegWriter(
        fps=fps, metadata={"artist": "vanguard"}, bitrate=2200
    )
    save_start = time.perf_counter()
    print(
        "[mp4] starting render: " f"frames={n_frames}, fps={fps}, output={output_path}"
    )

    def _progress(frame_idx: int, total_frames: int) -> None:
        if frame_idx == 0 or frame_idx + 1 == total_frames or frame_idx % 10 == 0:
            print(f"[mp4] rendered frame {frame_idx + 1}/{total_frames}")

    anim.save(
        str(output_path),
        writer=writer,
        dpi=140,
        progress_callback=_progress,
    )
    print(
        "[mp4] render complete: "
        f"{output_path} ({time.perf_counter() - save_start:.2f} seconds)"
    )
    plt.close(fig)


def _count_components(mask_zyx: np.ndarray) -> int:
    """Count 26-connected 3D components in a binary mask."""
    from scipy import ndimage

    structure = np.ones((3, 3, 3), dtype=np.uint8)
    _, n_comp = ndimage.label(mask_zyx.astype(np.uint8), structure=structure)
    return int(n_comp)


def _baseline_3d_mask(priority_zyx: np.ndarray, threshold_low: float) -> np.ndarray:
    """Compute baseline 3D mask using shared processing helper."""
    return _shared_baseline_3d_mask(priority_zyx, threshold_low=threshold_low)


def _collapse_4d_to_exam_skeleton(
    mask_4d: np.ndarray,
    min_temporal_support: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse 4D manifold into one 3D exam-level skeleton."""
    return _shared_collapse_4d_to_exam_skeleton(
        mask_4d=mask_4d,
        min_temporal_support=min_temporal_support,
    )


def main() -> None:
    """CLI entrypoint."""
    start_time = time.perf_counter()

    parser = argparse.ArgumentParser(
        description="Compare baseline 3D skeletons against full 4D center-manifold extraction."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_SEGMENTATION_DIR,
        help="Directory containing per-timepoint segmentation .npy files.",
    )
    parser.add_argument(
        "--study-id",
        type=str,
        default=None,
        help=(
            "Study/patient ID used to auto-discover all timepoints from --input-dir "
            "(e.g., ISPY2_202539)."
        ),
    )
    parser.add_argument(
        "--input-4d",
        type=Path,
        default=None,
        help="Single .npy containing the full time series.",
    )
    parser.add_argument(
        "--input-files",
        type=Path,
        nargs="+",
        default=None,
        help="Per-timepoint .npy files (sorted order is used).",
    )
    parser.add_argument(
        "--layout",
        choices=("tzyx", "ctzyx", "tczyx"),
        default="tzyx",
        help="Layout for --input-4d.",
    )
    parser.add_argument(
        "--npy-channel",
        type=int,
        default=1,
        help="Channel index for channelized NPY inputs.",
    )
    parser.add_argument(
        "--time-index",
        type=int,
        default=0,
        help="Timepoint index for 3D vs 4D projection comparison.",
    )
    parser.add_argument(
        "--threshold-low",
        type=float,
        default=0.5,
        help="Low threshold for active voxels.",
    )
    parser.add_argument(
        "--threshold-high",
        type=float,
        default=0.85,
        help=(
            "Optional high threshold for undeletable anchor voxels " "(default: 0.85)."
        ),
    )
    parser.add_argument(
        "--max-temporal-radius",
        type=int,
        default=1,
        help="Temporal edge spatial search radius (voxels).",
    )
    parser.add_argument(
        "--min-voxels-per-timepoint",
        type=int,
        default=64,
        help="Never reduce any timepoint below this voxel count.",
    )
    parser.add_argument(
        "--min-anchor-fraction",
        type=float,
        default=0.005,
        help="Minimum anchored fraction per timepoint.",
    )
    parser.add_argument(
        "--min-anchor-voxels",
        type=int,
        default=128,
        help="Minimum anchored voxels per timepoint.",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Optional cap on number of voxels considered for pruning.",
    )
    parser.add_argument(
        "--tumor-mask",
        type=Path,
        default=None,
        help=(
            "Optional tumor segmentation file to overlay in rotating 3D MP4 "
            "(supports .npy/.nii/.nii.gz/.nrrd)."
        ),
    )
    parser.add_argument(
        "--tumor-mask-dir",
        type=Path,
        default=DEFAULT_TUMOR_MASK_DIR,
        help=(
            "Directory used to auto-resolve tumor mask from study id when "
            "--tumor-mask is not provided; expected filename {study_id}.nii.gz."
        ),
    )
    parser.add_argument(
        "--tumor-threshold",
        type=float,
        default=0.5,
        help="Threshold used to binarize tumor-mask inputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for masks, plots, and summary JSON.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip writing projection PNGs (useful in environments without matplotlib).",
    )
    parser.add_argument(
        "--min-temporal-support",
        type=int,
        default=2,
        help=(
            "Require this many retained 4D timepoints at a voxel before "
            "contributing to the exam-level skeleton."
        ),
    )
    args = parser.parse_args()

    from graph_extraction.skeleton4d import skeletonize4d

    has_study_mode = args.study_id is not None
    has_input4d_mode = args.input_4d is not None
    has_files_mode = bool(args.input_files)
    mode_count = int(has_study_mode) + int(has_input4d_mode) + int(has_files_mode)
    if mode_count != 1:
        raise ValueError(
            "Choose exactly one input mode: "
            "(--study-id with --input-dir) OR (--input-4d) OR (--input-files)."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    discovered_files: list[Path] | None = None
    discovered_timepoints: list[int] | None = None

    if has_study_mode:
        discovered_files, discovered_timepoints = _discover_study_timepoints(
            input_dir=args.input_dir,
            study_id=args.study_id,
        )
        priority_4d = _load_time_series_from_files(
            discovered_files, npy_channel=args.npy_channel
        )
    elif has_input4d_mode:
        priority_4d = _load_time_series_from_single_npy(
            args.input_4d, layout=args.layout, npy_channel=args.npy_channel
        )
    else:
        files = sorted(args.input_files)
        priority_4d = _load_time_series_from_files(files, npy_channel=args.npy_channel)

    if priority_4d.ndim != NDIM_4D:
        raise ValueError(f"Expected (t,z,y,x) array, got shape {priority_4d.shape}")

    t_dim = priority_4d.shape[0]
    if args.time_index < 0 or args.time_index >= t_dim:
        raise ValueError(f"time-index must be in [0, {t_dim - 1}]")

    print(f"[info] Loaded priority volume shape: {priority_4d.shape}")
    print(
        f"[info] Priority min/max: {float(priority_4d.min()):.6f}/{float(priority_4d.max()):.6f}"
    )
    active_mask = priority_4d >= args.threshold_low
    active_total = int(np.count_nonzero(active_mask))
    active_per_t = [int(x) for x in np.count_nonzero(active_mask, axis=(1, 2, 3))]
    active_fraction = float(active_total / active_mask.size)
    print(
        "[info] Active voxels at threshold-low: "
        f"total={active_total}, fraction={active_fraction:.6f}, per_timepoint={active_per_t}"
    )

    selected_priority = priority_4d[args.time_index]
    t_baseline_start = time.perf_counter()
    mask_3d = _baseline_3d_mask(selected_priority, threshold_low=args.threshold_low)
    print(f"[timing] baseline_3d_seconds={time.perf_counter() - t_baseline_start:.2f}")

    t_4d_start = time.perf_counter()
    mask_4d = skeletonize4d(
        priority_4d,
        threshold_low=args.threshold_low,
        threshold_high=args.threshold_high,
        max_temporal_radius=args.max_temporal_radius,
        min_voxels_per_timepoint=args.min_voxels_per_timepoint,
        min_anchor_fraction=args.min_anchor_fraction,
        min_anchor_voxels=args.min_anchor_voxels,
        max_candidates=args.max_candidates,
        verbose=True,
    )
    print(f"[timing] manifold_4d_seconds={time.perf_counter() - t_4d_start:.2f}")

    t_collapse_start = time.perf_counter()
    exam_skeleton, support_mask, support_count = _collapse_4d_to_exam_skeleton(
        mask_4d, min_temporal_support=args.min_temporal_support
    )
    print(
        f"[timing] collapse_to_exam_seconds={time.perf_counter() - t_collapse_start:.2f}"
    )

    resolved_tumor_path: Path | None = args.tumor_mask
    if resolved_tumor_path is None and has_study_mode:
        candidate = args.tumor_mask_dir / f"{args.study_id}.nii.gz"
        if not candidate.exists():
            raise ValueError(
                "Tumor mask auto-resolution failed for study mode. "
                f"Expected: {candidate}. Provide --tumor-mask explicitly if needed."
            )
        resolved_tumor_path = candidate

    tumor_mask_zyx: np.ndarray | None = None
    if resolved_tumor_path is not None:
        tumor_mask_zyx = _load_tumor_mask_volume(
            resolved_tumor_path,
            expected_shape_zyx=tuple(int(v) for v in exam_skeleton.shape),
            t_dim=t_dim,
            npy_channel=args.npy_channel,
            threshold=args.tumor_threshold,
        )
        print(
            "[info] Loaded tumor overlay mask: "
            f"{resolved_tumor_path} (voxels={int(np.count_nonzero(tumor_mask_zyx))})"
        )

    projections_3d = _compute_projections(mask_zyx=mask_3d)
    projections_4d = _compute_projections(mask_zyx=exam_skeleton)

    out_cmp = args.output_dir / f"projection_compare_t{args.time_index:03d}_vs_exam.png"
    out_rot = args.output_dir / f"rotation_compare_t{args.time_index:03d}_vs_exam.mp4"
    title_study = f" | Study: {args.study_id}" if args.study_id is not None else ""
    if not args.no_plots:
        t_plot_start = time.perf_counter()
        _save_projection_comparison(
            projections_3d,
            projections_4d,
            out_cmp,
            title=f"3D timepoint {args.time_index} vs 4D exam-level skeleton{title_study}",
        )
        _save_rotating_3d_comparison_mp4(
            mask_3d=mask_3d,
            mask_4d_skeleton=exam_skeleton,
            output_path=out_rot,
            tumor_mask_zyx=tumor_mask_zyx,
            study_label=args.study_id,
        )
        print(f"[timing] plot_mp4_seconds={time.perf_counter() - t_plot_start:.2f}")

    np.save(args.output_dir / "center_manifold_4d_mask.npy", mask_4d.astype(np.uint8))
    np.save(
        args.output_dir / "skeleton_4d_exam_mask.npy",
        exam_skeleton.astype(np.uint8),
    )
    np.save(
        args.output_dir / "skeleton_4d_exam_support_mask.npy",
        support_mask.astype(np.uint8),
    )
    np.save(
        args.output_dir / f"skeleton_3d_t{args.time_index:03d}_mask.npy",
        mask_3d.astype(np.uint8),
    )

    summary = {
        "shape_tzyx": [int(x) for x in priority_4d.shape],
        "time_index": int(args.time_index),
        "threshold_low": float(args.threshold_low),
        "threshold_high": (
            None if args.threshold_high is None else float(args.threshold_high)
        ),
        "max_temporal_radius": int(args.max_temporal_radius),
        "min_voxels_per_timepoint": int(args.min_voxels_per_timepoint),
        "min_anchor_fraction": float(args.min_anchor_fraction),
        "min_anchor_voxels": int(args.min_anchor_voxels),
        "min_temporal_support": int(args.min_temporal_support),
        "max_candidates": None
        if args.max_candidates is None
        else int(args.max_candidates),
        "3d_selected_time_voxels": int(np.count_nonzero(mask_3d)),
        "4d_exam_skeleton_voxels": int(np.count_nonzero(exam_skeleton)),
        "4d_exam_support_voxels": int(np.count_nonzero(support_mask)),
        "4d_all_time_voxels": int(np.count_nonzero(mask_4d)),
        "3d_selected_time_components_26": _count_components(mask_3d),
        "4d_exam_skeleton_components_26": _count_components(exam_skeleton),
        "retained_voxels_per_timepoint_4d": [
            int(x) for x in np.count_nonzero(mask_4d, axis=(1, 2, 3))
        ],
        "retained_support_count_histogram": {
            str(k): int(v)
            for k, v in enumerate(
                np.bincount(support_count.ravel(), minlength=priority_4d.shape[0] + 1)
            )
        },
        "tumor_overlay": {
            "path": None if resolved_tumor_path is None else str(resolved_tumor_path),
            "threshold": (
                None if resolved_tumor_path is None else float(args.tumor_threshold)
            ),
            "voxels": (
                0 if tumor_mask_zyx is None else int(np.count_nonzero(tumor_mask_zyx))
            ),
            "auto_resolved_from_study_id": bool(
                args.tumor_mask is None
                and resolved_tumor_path is not None
                and has_study_mode
            ),
        },
        "study_mode": {
            "study_id": args.study_id,
            "input_dir": str(args.input_dir) if has_study_mode else None,
            "timepoints": discovered_timepoints
            if discovered_timepoints is not None
            else None,
            "files": (
                [str(p) for p in discovered_files]
                if discovered_files is not None
                else None
            ),
        },
        "outputs": {
            "projection_compare": None if args.no_plots else str(out_cmp),
            "rotation_compare_3d_mp4": None if args.no_plots else str(out_rot),
        },
    }
    elapsed_seconds = float(time.perf_counter() - start_time)
    summary["elapsed_seconds"] = elapsed_seconds

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[done] Wrote outputs:")
    if not args.no_plots:
        print(f"  - {out_cmp}")
        print(f"  - {out_rot}")
    print(f"  - {summary_path}")
    print(f"[done] Total time: {elapsed_seconds:.2f} seconds")


if __name__ == "__main__":
    main()
