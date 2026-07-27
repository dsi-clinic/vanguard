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
        "max-over-time",
        args.out_dir / f"floor_activation_{args.case_id[:24]}_z{z}.png",
    )
    # Single-panel overlay on the PRECONTRAST slice: red floor-active voxels on
    # the raw precontrast image (where they sit on the dark near-void regions).
    # Uses the slice carrying the most floor-active voxels so they're visible.
    z_best = int((active_mask & support).sum(axis=(1, 2)).argmax())
    _precontrast_overlay_plot(
        baseline,
        support,
        active_mask,
        z_best,
        args.case_id,
        args.out_dir
        / f"floor_active_precontrast_overlay_{args.case_id[:24]}_z{z_best}.png",
    )
    _mechanism_plot(
        dce_4d,
        baseline,
        active_mask,
        support,
        baseline_frame_count,
        z,
        args.case_id,
        args.out_dir / f"floor_mechanism_{args.case_id[:24]}_z{z}.png",
    )
    _precontrast_plot(
        baseline,
        support,
        active_mask,
        z,
        args.case_id,
        args.out_dir / f"precontrast_{args.case_id[:24]}_z{z}.png",
    )


def _precontrast_overlay_plot(
    baseline: np.ndarray,
    support: np.ndarray,
    active_mask: np.ndarray,
    z: int,
    case_id: str,
    out_path: Path,
) -> None:
    """Precontrast + red floor-active voxels, beside the pure precontrast slice.

    Same slice and window on both panels, so the red voxels can be checked
    against the pure image -- they should land where it is black (S0 ~ 0).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pre = baseline[z]
    supp = support[z]
    ys, xs = np.where(active_mask[z] & supp)
    vmax = max(np.percentile(pre[supp], 99) if supp.any() else 1.0, _EPS)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    axes[0].imshow(pre, cmap="gray", vmin=0, vmax=vmax)
    axes[0].scatter(xs, ys, c="red", s=16, marker="s", linewidths=0)
    axes[0].set_title(f"precontrast + floor-active (red, n={len(xs)})")
    axes[1].imshow(pre, cmap="gray", vmin=0, vmax=vmax)
    axes[1].set_title("pure precontrast (same slice)")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"z={z} — {case_id[:36]}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path} ({len(xs)} red voxels)")


def _precontrast_plot(
    baseline: np.ndarray,
    support: np.ndarray,
    active_mask: np.ndarray,
    z: int,
    case_id: str,
    out_path: Path,
) -> None:
    """Show that most precontrast voxels are NOT ~0.

    Left: the raw precontrast S0 image (mean of the precontrast frames) at the
    center slice -- normal grayscale anatomy, not a black frame. Right: the
    in-support S0 distribution, with the floor-active voxels overlaid: the bulk
    of vessel voxels sit at a healthy baseline, and only a small tail is near 0.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    b = baseline[z]
    supp_s0 = baseline[support]
    active_s0 = baseline[active_mask]
    med = float(np.median(supp_s0))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    hi = np.percentile(b[b > 0], 99) if (b > 0).any() else 1.0
    im = axes[0].imshow(b, cmap="gray", vmin=0, vmax=max(hi, _EPS))
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_title(
        f"Raw precontrast S0 (z={z})\nin-support median S0={med:.1f} (not 0)"
    )
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].hist(
        supp_s0, bins=80, color="steelblue", alpha=0.8, label="all vessel voxels"
    )
    if active_s0.size:
        axes[1].hist(
            active_s0, bins=80, color="red", alpha=0.9, label="floor-active (S0~0)"
        )
    axes[1].axvline(med, color="k", ls="--", lw=1, label=f"median {med:.1f}")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("precontrast S0")
    axes[1].set_ylabel("voxel count (log)")
    axes[1].set_title("Vessel-voxel S0 distribution\nhealthy bulk + a tiny near-0 tail")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Are all precontrast voxels 0? No — {case_id[:32]}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def _mechanism_plot(
    dce_4d: np.ndarray,
    baseline: np.ndarray,
    active_mask: np.ndarray,
    support: np.ndarray,
    baseline_frame_count: int,
    z: int,
    case_id: str,
    out_path: Path,
) -> None:
    """Why S0~0 voxels are still visible: black precontrast, bright postcontrast.

    Overlays the same problematic voxels on the precontrast S0 frame (where they
    are ~black) and the post-contrast peak frame (where they are bright), then
    plots the actual per-voxel S(t) curves -- problematic voxels start near 0 and
    rise to a real peak (numerator real, denominator ~0 -> the ratio explodes),
    healthy voxels start at a substantial baseline. The exploding quantity is the
    derived (S-S0)/S0 feature, never the raw signal shown in grayscale.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    supp = support[z]
    pre = baseline[z]
    post = dce_4d[baseline_frame_count:].max(axis=0)[z]
    active_z = active_mask[z]
    red = ListedColormap([(1, 0, 0, 0), (1, 0, 0, 0.9)])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    pre_hi = np.percentile(pre[supp], 99) if supp.any() else 1.0
    post_hi = np.percentile(post[supp], 99) if supp.any() else 1.0
    axes[0].imshow(pre, cmap="gray", vmin=0, vmax=max(pre_hi, _EPS))
    axes[0].imshow(np.where(active_z, 1.0, np.nan), cmap=red, vmin=0, vmax=1)
    axes[0].set_title(f"PRECONTRAST S0 (z={z})\nred voxels are ~black here (S0~0)")
    axes[1].imshow(post, cmap="gray", vmin=0, vmax=max(post_hi, _EPS))
    axes[1].imshow(np.where(active_z, 1.0, np.nan), cmap=red, vmin=0, vmax=1)
    axes[1].set_title("POSTCONTRAST peak\nsame voxels are bright (real enhancement)")
    for ax in axes[:2]:
        ax.set_xticks([])
        ax.set_yticks([])

    # S(t) for the most-extreme problematic voxels vs. near-median healthy voxels.
    t = np.arange(dce_4d.shape[0])
    coords = np.argwhere(active_mask)
    s0_active = baseline[active_mask]
    n = min(6, len(coords))
    for i in np.argsort(s0_active)[:n]:
        zz, yy, xx = coords[i]
        axes[2].plot(t, dce_4d[:, zz, yy, xx], color="red", alpha=0.6, lw=1)
    healthy = support & ~active_mask
    hcoords = np.argwhere(healthy)
    hs0 = baseline[healthy]
    for i in np.argsort(np.abs(hs0 - np.median(hs0)))[:n]:
        zz, yy, xx = hcoords[i]
        axes[2].plot(t, dce_4d[:, zz, yy, xx], color="steelblue", alpha=0.6, lw=1)
    axes[2].axvspan(-0.5, baseline_frame_count - 0.5, color="gray", alpha=0.15)
    axes[2].axhline(0, color="k", lw=0.5)
    axes[2].plot([], [], color="red", label="problematic (S0~0)")
    axes[2].plot([], [], color="steelblue", label="healthy")
    axes[2].axvspan(0, 0, color="gray", alpha=0.15, label="precontrast frames")
    axes[2].legend(fontsize=8)
    axes[2].set_xlabel("DCE frame (t)")
    axes[2].set_ylabel("raw signal S(t)")
    axes[2].set_title("both enhance; only the starting baseline differs")
    fig.suptitle(f"Why S0~0 voxels are still visible — {case_id[:32]}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


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
    anatomy: np.ndarray,
    support: np.ndarray,
    skeleton: np.ndarray,
    floor_active: np.ndarray,
    z: int,
    frac: float,
    case_id: str,
    anatomy_label: str,
    out_path: Path,
) -> None:
    """Three-panel top-down slice: baseline S0, baseline/scale ratio, anatomy+red overlay.

    ``anatomy`` is the grayscale background for the overlay panel -- pass the
    precontrast baseline to see the floor-active voxels land on the dark regions,
    or the max-over-time image to see them on the enhancing anatomy.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    supp = support[z]
    b = baseline[z]
    anat_z = anatomy[z]
    active = floor_active[z] & supp
    ratio = np.full_like(b, np.nan)
    np.divide(b, signal_scale[z], out=ratio, where=signal_scale[z] > _EPS)

    # Windows from the in-support distribution (one shared window per panel).
    b_hi = np.percentile(b[supp], 99) if supp.any() else 1.0
    anat_hi = np.percentile(anat_z[supp], 99) if supp.any() else 1.0
    red = ListedColormap([(1, 0, 0, 0), (1, 0, 0, 0.85)])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    axes[0].imshow(b, cmap="gray", vmin=0, vmax=max(b_hi, _EPS))
    axes[0].contour(supp, levels=[0.5], colors="cyan", linewidths=0.5)
    axes[0].set_title(f"Precontrast baseline S0 (z={z})\ncyan = vessel support")

    im = axes[1].imshow(np.where(supp, ratio, np.nan), cmap="viridis", vmin=0, vmax=0.3)
    axes[1].contour((ratio < frac) & supp, levels=[0.5], colors="red", linewidths=0.6)
    axes[1].set_title(f"baseline / max|S|  (red contour < {frac})")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(anat_z, cmap="gray", vmin=0, vmax=max(anat_hi, _EPS))
    axes[2].imshow(np.where(active, 1.0, np.nan), cmap=red, vmin=0, vmax=1)
    axes[2].contour(skeleton[z], levels=[0.5], colors="yellow", linewidths=0.4)
    n_active = int(active.sum())
    axes[2].set_title(
        f"{anatomy_label} + floor-active (red, n={n_active})\nyellow = skeleton"
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
