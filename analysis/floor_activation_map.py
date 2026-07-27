"""Verify the kinetic baseline floor: WHERE (and why) does it activate?

For one case, replicates the exact per-voxel denominator logic of
``gnn.kinetics.baseline_relative_curve`` (relative enhancement mode):

    baseline      = mean of the first ``baseline_frame_count`` DCE frames
    signal_scale  = max_t |S(t)|
    floor active  <=>  baseline_floor_frac * signal_scale > baseline

A voxel is "floor active" exactly when its precontrast baseline is a smaller
fraction of its own peak signal than ``baseline_floor_frac`` -- i.e. the
near-void-baseline voxels whose (S - S0)/S0 would otherwise explode. This maps
those voxels on the top-down center axial slice so we can see whether they sit
where the anatomy is genuinely near-void (mask edges / low-signal regions),
verifying the floor targets artifacts rather than real tissue.

Loads one case's raw 4D DCE -- run via Slurm, not the head node.

Usage:
    python -m analysis.floor_activation_map --case-id <id> \
        --centerline-root .../preprocessing_out_v5/centerlines \
        --dce-root .../preprocessing_out_v5/dce \
        --floor-frac 0.05 --out-dir experiments/floor_activation_check
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gnn.data_loader import (
    _CENTERLINE_SUFFIX,
    _SUPPORT_PATTERN,
    _load_study_metadata,
)
from gnn.raw_dce import (
    discover_raw_dce_paths,
    load_raw_dce_series,
)

_EPS = float(np.finfo(np.float32).eps)


def _resolve_case(centerline_root: Path, case_id: str) -> tuple[Path, Path]:
    """Find the skeleton mask + support mask for one case under the tree."""
    matches = sorted(centerline_root.rglob(f"{case_id}{_CENTERLINE_SUFFIX}"))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one skeleton for {case_id}, found {len(matches)}"
        )
    mask_path = matches[0]
    support_path = mask_path.parent / _SUPPORT_PATTERN.format(case_id=case_id)
    if not support_path.exists():
        raise FileNotFoundError(f"support mask not found: {support_path}")
    return mask_path, support_path


def main() -> None:
    """Compute the floor-activation mask for one case and render the overlay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", type=str, required=True)
    parser.add_argument("--centerline-root", type=Path, required=True)
    parser.add_argument("--dce-root", type=Path, required=True)
    parser.add_argument("--floor-frac", type=float, default=0.05)
    parser.add_argument(
        "--z",
        type=int,
        default=None,
        help="axial slice; default = center of the vessel-support z-extent",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    mask_path, support_path = _resolve_case(args.centerline_root, args.case_id)
    study_dir = mask_path.parent
    skeleton = np.load(mask_path).astype(bool)
    support = np.load(support_path).astype(bool)

    timepoints, baseline_frame_count, relative_enhancement = _load_study_metadata(
        args.case_id, study_dir
    )
    if not relative_enhancement:
        raise ValueError(
            f"case={args.case_id}: relative_enhancement is False (absolute "
            "convention) -- the floor never applies here, nothing to verify."
        )
    dce_paths = discover_raw_dce_paths(args.dce_root, args.case_id, timepoints)
    dce_4d = load_raw_dce_series(dce_paths, expected_shape_zyx=support.shape)

    baseline = dce_4d[:baseline_frame_count].mean(axis=0)
    signal_scale = np.abs(dce_4d).max(axis=0)
    floored_denom_term = args.floor_frac * signal_scale
    # The floor bites exactly where it raises the denominator above the raw
    # baseline (matches np.maximum(baseline, frac*signal_scale) in kinetics.py).
    floor_active = (floored_denom_term > baseline) & (signal_scale > _EPS)
    # Truly dead voxels: even the floored denominator is ~0, so enhancement stays 0.
    zeroed = ~(np.isfinite(np.maximum(baseline, floored_denom_term))) | (
        np.maximum(baseline, floored_denom_term) <= _EPS
    )

    supp_n = int(support.sum())
    skel_n = int(skeleton.sum())
    supp_active = int((floor_active & support).sum())
    skel_active = int((floor_active & skeleton).sum())
    print(
        f"case={args.case_id}  baseline_frames={baseline_frame_count}  "
        f"relative_enhancement={relative_enhancement}"
    )
    print(
        f"support voxels: {supp_n}  floor-active: {supp_active} "
        f"({100 * supp_active / max(supp_n, 1):.1f}%)"
    )
    print(
        f"skeleton (graph-node) voxels: {skel_n}  floor-active: {skel_active} "
        f"({100 * skel_active / max(skel_n, 1):.1f}%)"
    )
    print(f"dead/zeroed voxels within support: {int((zeroed & support).sum())}")
    ratio = baseline / np.where(signal_scale > _EPS, signal_scale, np.nan)
    supp_ratio = ratio[support]
    print(
        f"baseline/signal_scale within support: median={np.nanmedian(supp_ratio):.3f} "
        f"(floor threshold = {args.floor_frac})"
    )

    # Isolate the problematic voxels and confirm the "S0 ~ 0" claim in ABSOLUTE
    # terms (not just the relative floor trigger): their precontrast S0 should be
    # a tiny fraction of the healthy in-support S0, while their peak signal max|S|
    # stays comparable -- dark before contrast, real peak after -> blowup.
    active_mask = floor_active & support
    med_support_s0 = float(np.median(baseline[support]))
    if active_mask.any():
        s0_active = baseline[active_mask]
        print(
            f"floor-active S0 (absolute): median={np.median(s0_active):.2f} "
            f"p90={np.percentile(s0_active, 90):.2f}  vs in-support median S0="
            f"{med_support_s0:.2f}  (ratio={np.median(s0_active) / med_support_s0:.3f})"
        )
        print(
            f"floor-active max|S| (absolute): median={np.median(signal_scale[active_mask]):.2f} "
            f"vs in-support median max|S|={np.median(signal_scale[support]):.2f} "
            "-> near-0 baseline but real peak (the blowup)"
        )
    mask_path_out = args.out_dir / f"floor_active_mask_{args.case_id[:24]}.npy"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(mask_path_out, active_mask)
    print(
        f"wrote isolated problematic-voxel mask {mask_path_out} "
        f"({int(active_mask.sum())} voxels)"
    )

    # Center axial slice of the vessel-support z-extent (top-down view).
    z_support = np.where(support.any(axis=(1, 2)))[0]
    z = args.z if args.z is not None else int((z_support.min() + z_support.max()) // 2)
    print(f"axial slice z={z} (support z-extent {z_support.min()}..{z_support.max()})")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _plot(
        baseline,
        signal_scale,
        dce_4d.max(axis=0),
        support,
        skeleton,
        floor_active,
        z,
        args.floor_frac,
        args.case_id,
        args.out_dir / f"floor_activation_{args.case_id[:24]}_z{z}.png",
    )
    _isolation_plot(
        active_mask,
        support,
        dce_4d.max(axis=0),
        args.case_id,
        args.out_dir / f"floor_active_zproj_{args.case_id[:24]}.png",
    )


def _isolation_plot(
    active_mask: np.ndarray,
    support: np.ndarray,
    mip: np.ndarray,
    case_id: str,
    out_path: Path,
) -> None:
    """Top-down (max-over-z) projection isolating ALL floor-active voxels.

    Collapses the volume along z so every problematic voxel at any depth shows in
    one axial view, over the vessel-support footprint and a faint anatomy MIP --
    reveals whether they cluster at the breast/mask periphery (expected for
    near-void baselines) or sit inside the vasculature.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    active_proj = active_mask.any(axis=0)
    supp_proj = support.any(axis=0)
    mip_proj = mip.max(axis=0)
    red = ListedColormap([(1, 0, 0, 0), (1, 0, 0, 1.0)])

    fig, ax = plt.subplots(figsize=(6.5, 6))
    hi = np.percentile(mip_proj[supp_proj], 99) if supp_proj.any() else 1.0
    ax.imshow(mip_proj, cmap="gray", vmin=0, vmax=max(hi, _EPS))
    ax.contour(supp_proj, levels=[0.5], colors="cyan", linewidths=0.5)
    ax.imshow(np.where(active_proj, 1.0, np.nan), cmap=red, vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"Problematic voxels (S0~0), all depths — {case_id[:32]}\n"
        f"red = floor-active (n={int(active_mask.sum())}); cyan = vessel support"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def _plot(
    baseline: np.ndarray,
    signal_scale: np.ndarray,
    mip: np.ndarray,
    support: np.ndarray,
    skeleton: np.ndarray,
    floor_active: np.ndarray,
    z: int,
    frac: float,
    case_id: str,
    out_path: Path,
) -> None:
    """Three-panel top-down slice: baseline S0, baseline/scale ratio, anatomy+red overlay."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    supp = support[z]
    b = baseline[z]
    mip_z = mip[z]
    active = floor_active[z] & supp
    ratio = np.full_like(b, np.nan)
    np.divide(b, signal_scale[z], out=ratio, where=signal_scale[z] > _EPS)

    # Windows from the in-support distribution (one shared window per panel).
    b_hi = np.percentile(b[supp], 99) if supp.any() else 1.0
    mip_hi = np.percentile(mip_z[supp], 99) if supp.any() else 1.0
    red = ListedColormap([(1, 0, 0, 0), (1, 0, 0, 0.85)])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    axes[0].imshow(b, cmap="gray", vmin=0, vmax=max(b_hi, _EPS))
    axes[0].contour(supp, levels=[0.5], colors="cyan", linewidths=0.5)
    axes[0].set_title(f"Precontrast baseline S0 (z={z})\ncyan = vessel support")

    im = axes[1].imshow(np.where(supp, ratio, np.nan), cmap="viridis", vmin=0, vmax=0.3)
    axes[1].contour((ratio < frac) & supp, levels=[0.5], colors="red", linewidths=0.6)
    axes[1].set_title(f"baseline / max|S|  (red contour < {frac})")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(mip_z, cmap="gray", vmin=0, vmax=max(mip_hi, _EPS))
    axes[2].imshow(np.where(active, 1.0, np.nan), cmap=red, vmin=0, vmax=1)
    axes[2].contour(skeleton[z], levels=[0.5], colors="yellow", linewidths=0.4)
    n_active = int(active.sum())
    axes[2].set_title(
        f"max-over-time + floor-active (red, n={n_active})\nyellow = skeleton"
    )

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Floor activation — {case_id[:40]}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
