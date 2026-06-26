#!/usr/bin/env python3
"""
3D segmentation comparison: original vs. resampled vessel segmentations.

For 6 randomly selected cases (2 per cohort: ISPY1, ISPY2, NACT), loads all
available timepoints and produces per-case figures with:
  - Top panel:    original segmentation 3D point cloud
  - Bottom panel: resampled segmentation 3D point cloud

Points are colored by height (z position) using a gradient so you can read
depth in the static PNG. The rotating MP4 shows both panels spinning together.

Usage
-----
  python QC/compare_seg_3d.py
  python QC/compare_seg_3d.py --out-dir /path/to/output
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

_DEFAULT_ORIG = Path(
    "/gpfs/data/karczmar-lab/workspaces/saritbose/vessel_segmentations"
)
_DEFAULT_RESAMPLED = Path(
    "/gpfs/data/karczmar-lab/workspaces/saritbose/resampled-vessel-segmentations"
)
_DEFAULT_OUT = Path(
    os.environ.get("VANGUARD_QC_OUTPUT", str(Path.home() / "vanguard_qc_pngs"))
)
_COHORTS = ["ISPY1", "ISPY2", "NACT"]
_BG = "#111111"


def _cases_in_both(orig_root: Path, resamp_root: Path, cohort: str) -> list[str]:
    orig_cases = {p.name for p in (orig_root / cohort).iterdir() if p.is_dir()}
    resamp_cases = {p.name for p in (resamp_root / cohort).iterdir() if p.is_dir()}
    return sorted(orig_cases & resamp_cases)


def _load_union_mask(
    root: Path, cohort: str, case_id: str, threshold: float
) -> np.ndarray:
    """Load all timepoints and return union binary mask (True = vessel in any timepoint)."""
    images_dir = root / cohort / case_id / "images"
    npz_files = sorted(images_dir.glob(f"{case_id}_*_vessel_segmentation.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No segmentation .npz files in {images_dir}")

    max_prob: np.ndarray | None = None
    for f in npz_files:
        arr = np.load(f)["vessel"].astype(np.float32)
        if max_prob is None:
            max_prob = arr
        else:
            min_z = min(arr.shape[2], max_prob.shape[2])
            arr = arr[:, :, :min_z]
            max_prob = max_prob[:, :, :min_z]
            np.maximum(max_prob, arr, out=max_prob)

    assert max_prob is not None
    return max_prob > threshold


def _coords(mask: np.ndarray, stride: int) -> np.ndarray:
    """Return (N, 3) array of vessel voxel coordinates, downsampled by stride."""
    coords = np.argwhere(mask[::stride, ::stride, ::stride])
    return coords * stride


def _norm(coords: np.ndarray) -> np.ndarray:
    """Normalize coords to [0, 1] independently per axis for consistent display."""
    if len(coords) == 0:
        return coords
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    return (coords - lo) / np.maximum(hi - lo, 1.0)


def _draw(ax, coords: np.ndarray, title: str, point_size: float) -> None:
    """Draw one 3D scatter panel."""
    ax.set_facecolor(_BG)
    ax.set_axis_off()
    ax.set_title(title, color="white", fontsize=11, pad=4)
    if len(coords) == 0:
        return
    # Color by z-height so you can read depth in the static image
    ax.scatter(
        coords[:, 1], coords[:, 0], coords[:, 2],
        s=point_size,
        c=coords[:, 2],
        cmap="plasma",
        alpha=0.85,
        linewidths=0,
    )
    ax.view_init(elev=20, azim=45)


def _render_case(
    case_id: str,
    orig_coords: np.ndarray,
    resamp_coords: np.ndarray,
    out_dir: Path,
    n_frames: int,
    point_size: float,
) -> None:
    no = _norm(orig_coords)
    nr = _norm(resamp_coords)
    print(
        f"  {case_id}: orig={len(orig_coords)} pts, resamp={len(resamp_coords)} pts"
    )

    # ---- Static PNG: top = original, bottom = resampled --------------------
    fig = plt.figure(figsize=(7, 11), facecolor=_BG)
    ax_o = fig.add_subplot(211, projection="3d")
    ax_r = fig.add_subplot(212, projection="3d")

    _draw(ax_o, no, f"{case_id}  —  original",   point_size)
    _draw(ax_r, nr, f"{case_id}  —  resampled",  point_size)

    png_path = out_dir / f"{case_id}_3d_comparison.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    print(f"  Saved PNG: {png_path}")

    # ---- Rotating MP4: both panels spin together ---------------------------
    fig2 = plt.figure(figsize=(7, 11), facecolor=_BG)
    ax1 = fig2.add_subplot(211, projection="3d")
    ax2 = fig2.add_subplot(212, projection="3d")

    _draw(ax1, no, f"{case_id}  —  original",   point_size)
    _draw(ax2, nr, f"{case_id}  —  resampled",  point_size)

    def _update(frame: int):
        azim = frame * 360 / n_frames
        ax1.view_init(elev=20, azim=azim)
        ax2.view_init(elev=20, azim=azim)
        return (fig2,)

    ani = animation.FuncAnimation(
        fig2, _update, frames=n_frames, interval=50, blit=False
    )
    mp4_path = out_dir / f"{case_id}_3d_comparison.mp4"
    ani.save(str(mp4_path), writer=animation.FFMpegWriter(fps=20, bitrate=1200), dpi=120)
    plt.close(fig2)
    print(f"  Saved MP4: {mp4_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--orig-root",    type=Path,  default=_DEFAULT_ORIG)
    parser.add_argument("--resamp-root",  type=Path,  default=_DEFAULT_RESAMPLED)
    parser.add_argument("--out-dir",      type=Path,  default=_DEFAULT_OUT)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--n-per-cohort", type=int,   default=2)
    parser.add_argument("--threshold",    type=float, default=0.5)
    parser.add_argument("--stride",       type=int,   default=1,
                        help="Voxel stride for downsampling (1 = all points).")
    parser.add_argument("--point-size",   type=float, default=20,
                        help="Scatter point size in matplotlib units.")
    parser.add_argument("--n-frames",     type=int,   default=60)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    selected: list[tuple[str, str]] = []
    for cohort in _COHORTS:
        cases = _cases_in_both(args.orig_root, args.resamp_root, cohort)
        if len(cases) < args.n_per_cohort:
            raise RuntimeError(
                f"Only {len(cases)} cases in both roots for {cohort}; "
                f"need {args.n_per_cohort}."
            )
        chosen = rng.choice(cases, size=args.n_per_cohort, replace=False).tolist()
        for c in chosen:
            selected.append((cohort, c))
        print(f"{cohort}: selected {chosen}")

    for cohort, case_id in selected:
        print(f"\nProcessing {case_id} ({cohort}) ...")
        orig_mask   = _load_union_mask(args.orig_root,   cohort, case_id, args.threshold)
        resamp_mask = _load_union_mask(args.resamp_root, cohort, case_id, args.threshold)

        _render_case(
            case_id,
            _coords(orig_mask,   args.stride),
            _coords(resamp_mask, args.stride),
            args.out_dir,
            args.n_frames,
            args.point_size,
        )

    print(f"\nDone. All outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
