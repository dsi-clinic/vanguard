"""Compare legacy-vs-primary 3D skeleton pipelines with rotating debug visualization.

This CLI mirrors the input-loading flow from ``run_compare_3d_4d_debug.py`` but focuses
on one selected 3D timepoint and compares:
- Legacy 3D pipeline: threshold-tunable (ablation path)
- Primary 3D pipeline: fixed threshold (principal path)

Outputs:
- Legacy and primary 3D masks
- One rotating side-by-side 3D MP4
- Summary JSON with voxel counts, component counts, overlap, and runtime
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
    baseline_3d_mask,
    discover_study_timepoints,
    load_time_series_from_files,
)

DEFAULT_TUMOR_MASK_DIR = Path(
    "/net/projects2/vanguard/MAMA-MIA-syn60868042/segmentations/expert"
)
NDIM_3D = 3
NDIM_4D = 4
NDIM_5D = 5

# Primary flow keeps threshold fixed by design.
PRIMARY_3D_THRESHOLD_LOW = 0.5


def _load_time_series_from_single_npy(
    path: Path, layout: str, npy_channel: int
) -> np.ndarray:
    """Load a single NPY into ``(t, z, y, x)`` based on explicit layout."""
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


def _subsample_xyz(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, max_points: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uniformly subsample coordinate arrays to at most ``max_points``."""
    n = int(x.size)
    if n <= max_points:
        return x, y, z
    keep = np.linspace(0, n - 1, num=max_points, dtype=np.int64)
    return x[keep], y[keep], z[keep]


def _extract_xyz(mask_zyx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return float32 XYZ coordinates for nonzero voxels in a 3D mask."""
    z, y, x = np.nonzero(mask_zyx)
    return (
        x.astype(np.float32, copy=False),
        y.astype(np.float32, copy=False),
        z.astype(np.float32, copy=False),
    )


def _count_components(mask_zyx: np.ndarray) -> int:
    """Count 26-connected 3D components in a binary mask."""
    from scipy import ndimage

    structure = np.ones((3, 3, 3), dtype=np.uint8)
    _, n_comp = ndimage.label(mask_zyx.astype(np.uint8), structure=structure)
    return int(n_comp)


def _load_tumor_mask_npy(
    path: Path,
    *,
    expected_shape_zyx: tuple[int, int, int],
    t_dim: int,
    npy_channel: int,
    threshold: float,
) -> np.ndarray:
    """Load tumor segmentation as a 3D bool mask aligned to ``(z, y, x)``."""
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


def _prepare_tumor_overlay_points(
    tumor_mask_zyx: np.ndarray | None,
    expected_shape_zyx: tuple[int, int, int],
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    tuple[np.ndarray, np.ndarray, np.ndarray] | None,
]:
    """Prepare subsampled fill/shell points for tumor overlay."""
    if tumor_mask_zyx is None:
        return None, None

    from scipy import ndimage

    if tumor_mask_zyx.ndim != NDIM_3D:
        raise ValueError(f"Tumor mask must be 3D, got shape {tumor_mask_zyx.shape}")
    if tuple(tumor_mask_zyx.shape) != expected_shape_zyx:
        raise ValueError(
            "Tumor mask shape mismatch with skeleton masks: "
            f"{tumor_mask_zyx.shape} vs {expected_shape_zyx}"
        )
    if not np.any(tumor_mask_zyx):
        raise ValueError("Tumor mask is empty; cannot overlay.")

    tx_all, ty_all, tz_all = _extract_xyz(tumor_mask_zyx)
    tx_fill, ty_fill, tz_fill = _subsample_xyz(
        tx_all,
        ty_all,
        tz_all,
        max_points=6000,
    )

    tumor_eroded = ndimage.binary_erosion(
        tumor_mask_zyx,
        structure=np.ones((3, 3, 3), dtype=bool),
        border_value=0,
    )
    tumor_shell = tumor_mask_zyx & ~tumor_eroded
    if np.any(tumor_shell):
        tx_s, ty_s, tz_s = _extract_xyz(tumor_shell)
    else:
        tx_s, ty_s, tz_s = tx_all, ty_all, tz_all

    tx_shell, ty_shell, tz_shell = _subsample_xyz(
        tx_s,
        ty_s,
        tz_s,
        max_points=9000,
    )
    return (tx_fill, ty_fill, tz_fill), (tx_shell, ty_shell, tz_shell)


def _configure_3d_axes(
    ax: object,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
) -> None:
    """Apply consistent axis styling for 3D panels."""
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect((x_max - x_min + 1.0, y_max - y_min + 1.0, z_max - z_min + 1.0))


def _save_rotating_legacy_primary_mp4(
    legacy_mask_zyx: np.ndarray,
    primary_mask_zyx: np.ndarray,
    output_path: Path,
    *,
    tumor_mask_zyx: np.ndarray | None,
    case_label: str,
    n_frames: int,
    fps: int,
    elev: float,
    marker_size: float,
) -> None:
    """Save rotating MP4 with side-by-side legacy and primary 3D masks."""
    import matplotlib.pyplot as plt
    from matplotlib import animation

    if legacy_mask_zyx.ndim != NDIM_3D or primary_mask_zyx.ndim != NDIM_3D:
        raise ValueError("Both masks must be 3D.")
    if tuple(legacy_mask_zyx.shape) != tuple(primary_mask_zyx.shape):
        raise ValueError(
            "Mask shape mismatch: "
            f"legacy={legacy_mask_zyx.shape}, primary={primary_mask_zyx.shape}"
        )
    if not np.any(legacy_mask_zyx):
        raise ValueError("Legacy mask is empty; cannot render MP4.")
    if not np.any(primary_mask_zyx):
        raise ValueError("Primary mask is empty; cannot render MP4.")

    if "ffmpeg" not in animation.writers.list():
        raise RuntimeError("Matplotlib ffmpeg writer is unavailable. Install ffmpeg.")

    lx, ly, lz = _extract_xyz(legacy_mask_zyx)
    px, py, pz = _extract_xyz(primary_mask_zyx)

    x_parts = [lx, px]
    y_parts = [ly, py]
    z_parts = [lz, pz]

    tumor_fill_pts, tumor_shell_pts = _prepare_tumor_overlay_points(
        tumor_mask_zyx=tumor_mask_zyx,
        expected_shape_zyx=tuple(int(v) for v in legacy_mask_zyx.shape),
    )
    if tumor_shell_pts is not None:
        tx_shell, ty_shell, tz_shell = tumor_shell_pts
        x_parts.append(tx_shell)
        y_parts.append(ty_shell)
        z_parts.append(tz_shell)

    x_all = np.concatenate(x_parts)
    y_all = np.concatenate(y_parts)
    z_all = np.concatenate(z_parts)

    x_min, x_max = float(x_all.min()), float(x_all.max())
    y_min, y_max = float(y_all.min()), float(y_all.max())
    z_min, z_max = float(z_all.min()), float(z_all.max())

    fig = plt.figure(figsize=(11.5, 5.6), constrained_layout=True)
    ax_legacy = fig.add_subplot(1, 2, 1, projection="3d")
    ax_primary = fig.add_subplot(1, 2, 2, projection="3d")
    axes = [ax_legacy, ax_primary]

    for ax in axes:
        _configure_3d_axes(
            ax,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            z_min=z_min,
            z_max=z_max,
        )

    ax_legacy.set_title("Legacy 3D pipeline", fontsize=12)
    ax_primary.set_title("Primary 3D pipeline", fontsize=12)

    if tumor_fill_pts is not None and tumor_shell_pts is not None:
        tx_fill, ty_fill, tz_fill = tumor_fill_pts
        tx_shell, ty_shell, tz_shell = tumor_shell_pts
        for ax in axes:
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

    ax_legacy.scatter(lx, ly, lz, s=marker_size, c="#1f77b4", alpha=0.85, marker="o")
    ax_primary.scatter(
        px,
        py,
        pz,
        s=marker_size,
        c="#d62728",
        alpha=0.85,
        marker="o",
    )

    title = f"Legacy vs Primary 3D | Case: {case_label}"
    if tumor_mask_zyx is not None:
        title += " (+ tumor overlay)"
    fig.suptitle(title, fontsize=13)

    def _update(frame_idx: int) -> tuple[()]:
        azim = 360.0 * float(frame_idx) / float(n_frames)
        for ax in axes:
            ax.view_init(elev=elev, azim=azim)
        return ()

    anim = animation.FuncAnimation(
        fig,
        _update,
        frames=n_frames,
        interval=1000.0 / float(fps),
        blit=False,
    )
    writer = animation.FFMpegWriter(
        fps=fps,
        metadata={"artist": "vanguard"},
        bitrate=2200,
    )

    save_start = time.perf_counter()
    print(
        "[mp4] starting render: "
        f"panels=2, frames={n_frames}, fps={fps}, output={output_path}"
    )

    def _progress(frame_idx: int, total_frames: int) -> None:
        if frame_idx == 0 or frame_idx + 1 == total_frames or frame_idx % 10 == 0:
            print(f"[mp4] rendered frame {frame_idx + 1}/{total_frames}")

    anim.save(str(output_path), writer=writer, dpi=140, progress_callback=_progress)
    print(
        "[mp4] render complete: "
        f"{output_path} ({time.perf_counter() - save_start:.2f} seconds)"
    )
    plt.close(fig)


def _run_legacy_3d_mask(priority_zyx: np.ndarray, *, threshold_low: float) -> np.ndarray:
    """Run legacy 3D path (threshold-tunable)."""
    return baseline_3d_mask(priority_zyx, threshold_low=threshold_low)


def _run_primary_3d_mask(priority_zyx: np.ndarray) -> np.ndarray:
    """Run primary 3D path (fixed threshold)."""
    return baseline_3d_mask(priority_zyx, threshold_low=PRIMARY_3D_THRESHOLD_LOW)


def _compute_overlap_metrics(
    legacy_mask_zyx: np.ndarray,
    primary_mask_zyx: np.ndarray,
) -> dict[str, float | int]:
    """Compute overlap metrics between legacy and primary masks."""
    legacy_vox = int(np.count_nonzero(legacy_mask_zyx))
    primary_vox = int(np.count_nonzero(primary_mask_zyx))
    inter = int(np.count_nonzero(legacy_mask_zyx & primary_mask_zyx))
    union = int(np.count_nonzero(legacy_mask_zyx | primary_mask_zyx))

    dice_denom = legacy_vox + primary_vox
    dice = 0.0 if dice_denom == 0 else float((2.0 * inter) / float(dice_denom))
    jaccard = 0.0 if union == 0 else float(inter / float(union))

    return {
        "intersection_voxels": inter,
        "union_voxels": union,
        "dice": dice,
        "jaccard": jaccard,
        "legacy_overlap_fraction": 0.0
        if legacy_vox == 0
        else float(inter / float(legacy_vox)),
        "primary_overlap_fraction": 0.0
        if primary_vox == 0
        else float(inter / float(primary_vox)),
    }


def main() -> None:
    """CLI entrypoint."""
    start_time = time.perf_counter()

    parser = argparse.ArgumentParser(
        description=(
            "Compare legacy and primary 3D skeleton pipelines and render a rotating debug MP4."
        )
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
        help="Timepoint index used for legacy-vs-primary 3D comparison.",
    )
    parser.add_argument(
        "--legacy-threshold-low",
        type=float,
        default=0.5,
        help="Legacy low threshold (primary threshold remains fixed).",
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
        "--auto-tumor-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Auto-resolve tumor mask from study id when --tumor-mask is not provided."
        ),
    )
    parser.add_argument(
        "--mp4-frames",
        type=int,
        default=50,
        help="Number of rendered frames for the rotating MP4.",
    )
    parser.add_argument(
        "--mp4-fps",
        type=int,
        default=24,
        help="Frames per second for the rotating MP4.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for masks, MP4, and summary JSON.",
    )
    args = parser.parse_args()

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
        discovered_files, discovered_timepoints = discover_study_timepoints(
            input_dir=args.input_dir,
            study_id=args.study_id,
        )
        priority_4d = load_time_series_from_files(
            discovered_files,
            npy_channel=args.npy_channel,
        )
        case_label = args.study_id
    elif has_input4d_mode:
        priority_4d = _load_time_series_from_single_npy(
            args.input_4d,
            layout=args.layout,
            npy_channel=args.npy_channel,
        )
        case_label = args.input_4d.stem
    else:
        files = sorted(args.input_files)
        priority_4d = load_time_series_from_files(files, npy_channel=args.npy_channel)
        case_label = files[0].stem if len(files) == 1 else "multi-file-input"

    if priority_4d.ndim != NDIM_4D:
        raise ValueError(f"Expected (t,z,y,x) array, got shape {priority_4d.shape}")

    t_dim = int(priority_4d.shape[0])
    if args.time_index < 0 or args.time_index >= t_dim:
        raise ValueError(f"time-index must be in [0, {t_dim - 1}]")

    priority_zyx = priority_4d[args.time_index]
    print(f"[info] Loaded priority volume shape: {priority_4d.shape}")
    print(
        "[info] Selected timepoint: "
        f"t={args.time_index}, min/max="
        f"{float(priority_zyx.min()):.6f}/{float(priority_zyx.max()):.6f}"
    )

    t_legacy_start = time.perf_counter()
    legacy_mask = _run_legacy_3d_mask(
        priority_zyx,
        threshold_low=float(args.legacy_threshold_low),
    )
    legacy_seconds = float(time.perf_counter() - t_legacy_start)

    t_primary_start = time.perf_counter()
    primary_mask = _run_primary_3d_mask(priority_zyx)
    primary_seconds = float(time.perf_counter() - t_primary_start)

    if not np.any(legacy_mask):
        raise ValueError("Legacy mask is empty. Try lowering --legacy-threshold-low.")
    if not np.any(primary_mask):
        raise ValueError("Primary mask is empty. Check input segmentation values.")

    print(
        "[legacy] "
        f"threshold_low={args.legacy_threshold_low}, "
        f"voxels={int(np.count_nonzero(legacy_mask))}, "
        f"components_26={_count_components(legacy_mask)}, "
        f"seconds={legacy_seconds:.2f}"
    )
    print(
        "[primary] "
        f"threshold_low={PRIMARY_3D_THRESHOLD_LOW}, "
        f"voxels={int(np.count_nonzero(primary_mask))}, "
        f"components_26={_count_components(primary_mask)}, "
        f"seconds={primary_seconds:.2f}"
    )

    resolved_tumor_path: Path | None = args.tumor_mask
    if resolved_tumor_path is None and has_study_mode and args.auto_tumor_mask:
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
            expected_shape_zyx=tuple(int(v) for v in priority_zyx.shape),
            t_dim=t_dim,
            npy_channel=args.npy_channel,
            threshold=args.tumor_threshold,
        )
        print(
            "[info] Loaded tumor overlay mask: "
            f"{resolved_tumor_path} (voxels={int(np.count_nonzero(tumor_mask_zyx))})"
        )

    out_rot = args.output_dir / "rotation_compare_legacy_vs_primary_3d.mp4"
    t_video_start = time.perf_counter()
    _save_rotating_legacy_primary_mp4(
        legacy_mask_zyx=legacy_mask.astype(bool, copy=False),
        primary_mask_zyx=primary_mask.astype(bool, copy=False),
        output_path=out_rot,
        tumor_mask_zyx=tumor_mask_zyx,
        case_label=case_label,
        n_frames=int(args.mp4_frames),
        fps=int(args.mp4_fps),
        elev=20.0,
        marker_size=1.0,
    )
    video_seconds = float(time.perf_counter() - t_video_start)
    print(f"[timing] video_seconds={video_seconds:.2f}")

    out_legacy = args.output_dir / f"skeleton_legacy_t{args.time_index:03d}_mask.npy"
    out_primary = args.output_dir / f"skeleton_primary_t{args.time_index:03d}_mask.npy"
    np.save(out_legacy, legacy_mask.astype(np.uint8))
    np.save(out_primary, primary_mask.astype(np.uint8))

    overlap = _compute_overlap_metrics(legacy_mask, primary_mask)
    summary = {
        "shape_tzyx": [int(x) for x in priority_4d.shape],
        "selected_time_index": int(args.time_index),
        "legacy": {
            "threshold_low": float(args.legacy_threshold_low),
            "voxels": int(np.count_nonzero(legacy_mask)),
            "components_26": _count_components(legacy_mask),
            "runtime_seconds": legacy_seconds,
            "mask_path": str(out_legacy),
        },
        "primary": {
            "threshold_low": float(PRIMARY_3D_THRESHOLD_LOW),
            "voxels": int(np.count_nonzero(primary_mask)),
            "components_26": _count_components(primary_mask),
            "runtime_seconds": primary_seconds,
            "mask_path": str(out_primary),
        },
        "overlap": overlap,
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
                and args.auto_tumor_mask
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
            "rotation_compare_legacy_vs_primary_mp4": str(out_rot),
        },
        "timing_seconds": {
            "legacy": legacy_seconds,
            "primary": primary_seconds,
            "video": video_seconds,
            "total": float(time.perf_counter() - start_time),
        },
    }

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[done] Wrote outputs:")
    print(f"  - {out_rot}")
    print(f"  - {out_legacy}")
    print(f"  - {out_primary}")
    print(f"  - {summary_path}")
    print(f"[done] Total time: {summary['timing_seconds']['total']:.2f} seconds")


if __name__ == "__main__":
    main()
