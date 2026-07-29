"""Verify the kinetic baseline floor: WHERE (and why) does it activate?

For each case, replicates the exact per-voxel denominator logic of
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

Each case gets a flat directory ``<out-dir>/<case_id>/`` holding three figures:

    plot.png                       every precontrast frame at one slice, twice:
                                   floor-active voxels marked on top, pure below
    postcontrast.png               the same layout and slice for four sampled
                                   post-contrast frames (early/middle/end/peak)
    precontrast_voxel_values.png   sampled floor-active voxels, one point per
                                   precontrast frame, and mean vs max baseline

``summary.csv`` at the out-dir root carries one row per case, and
``analysis.floor_cohort_summary`` turns it into the cohort-level figures.

Loads each case's raw 4D DCE -- run via Slurm, not the head node.

Usage:
    python -m analysis.floor_activation_map \
        --centerline-root .../preprocessing_out_v5/centerlines \
        --dce-root .../preprocessing_out_v5/dce \
        --floor-frac 0.05 --out-dir .../precontrast_floor_experiment

``--case-id <id> [<id> ...]`` restricts the run to specific cases; omitting it
processes every case found under ``--centerline-root``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from analysis.floor_cohort_summary import render as render_cohort_figures  # noqa: E402
from gnn.data_loader import (  # noqa: E402
    _CENTERLINE_SUFFIX,
    _SUPPORT_PATTERN,
    _load_study_metadata,
)
from gnn.raw_dce import (  # noqa: E402
    discover_raw_dce_paths,
    load_raw_dce_series,
)

_EPS = float(np.finfo(np.float32).eps)
# A case can have hundreds of floor-active voxels; plotting every one of them
# with a point per precontrast frame is unreadable. The per-voxel figure draws a
# seeded random sample instead, with a two-voxel frame-by-frame detail.
_SAMPLE_VOXELS = 20
_DETAIL_VOXELS = 2
_SAMPLE_SEED = 0


def _discover_cases(centerline_root: Path) -> list[str]:
    """Every case id with a skeleton mask under the centerline tree, sorted."""
    return sorted(
        path.name[: -len(_CENTERLINE_SUFFIX)]
        for path in centerline_root.rglob(f"*{_CENTERLINE_SUFFIX}")
    )


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
    """Compute the floor-activation mask and render the figures for every case."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case-id",
        type=str,
        nargs="+",
        default=None,
        help="case ids to render; default = every case under --centerline-root",
    )
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

    case_ids = args.case_id or _discover_cases(args.centerline_root)
    print(f"rendering {len(case_ids)} case(s) -> {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        _render_case(
            case_id,
            args.centerline_root,
            args.dce_root,
            args.floor_frac,
            args.z,
            args.out_dir / case_id,
        )
        for case_id in case_ids
    ]

    summary_path = args.out_dir / "summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {summary_path} ({len(rows)} cases)")
    render_cohort_figures(summary_path)


def _render_case(
    case_id: str,
    centerline_root: Path,
    dce_root: Path,
    floor_frac: float,
    z_arg: int | None,
    case_dir: Path,
) -> dict:
    """Run the whole diagnostic for one case and return its summary row."""
    print(f"=================== {case_id} ===================")
    mask_path, support_path = _resolve_case(centerline_root, case_id)
    study_dir = mask_path.parent
    skeleton = np.load(mask_path).astype(bool)
    support = np.load(support_path).astype(bool)

    timepoints, baseline_frame_count, relative_enhancement = _load_study_metadata(
        case_id, study_dir
    )
    if not relative_enhancement:
        raise ValueError(
            f"case={case_id}: relative_enhancement is False (absolute "
            "convention) -- the floor never applies here, nothing to verify."
        )
    dce_paths = discover_raw_dce_paths(dce_root, case_id, timepoints)
    dce_4d = load_raw_dce_series(dce_paths, expected_shape_zyx=support.shape)

    precontrast = dce_4d[:baseline_frame_count]
    precontrast_timepoints = [int(t) for t in timepoints[:baseline_frame_count]]
    baseline = precontrast.mean(axis=0)
    signal_scale = np.abs(dce_4d).max(axis=0)
    floored_denom_term = floor_frac * signal_scale
    # The floor bites exactly where it raises the denominator above the raw
    # baseline (matches np.maximum(baseline, frac*signal_scale) in kinetics.py).
    floor_active = (floored_denom_term > baseline) & (signal_scale > _EPS)
    # Truly dead voxels: even the floored denominator is ~0, so enhancement stays 0.
    zeroed = ~(np.isfinite(np.maximum(baseline, floored_denom_term))) | (
        np.maximum(baseline, floored_denom_term) <= _EPS
    )
    active_mask = floor_active & support

    # The alternative to flooring: build the baseline with a max over the
    # precontrast frames instead of a mean. A voxel whose S0 is dragged to ~0 by
    # one bad frame recovers under max and needs no floor; a voxel that is ~0 in
    # every precontrast frame stays below the threshold either way. Counting
    # both says which of the two the cohort actually contains.
    baseline_max = precontrast.max(axis=0)
    floor_active_maxop = (floored_denom_term > baseline_max) & (signal_scale > _EPS)
    active_mask_maxop = floor_active_maxop & support

    supp_n = int(support.sum())
    skel_n = int(skeleton.sum())
    supp_active = int(active_mask.sum())
    skel_active = int((floor_active & skeleton).sum())
    print(
        f"case={case_id}  baseline_frames={baseline_frame_count}  "
        f"relative_enhancement={relative_enhancement}  "
        f"precontrast timepoints={precontrast_timepoints}"
    )
    print(
        f"support voxels: {supp_n}  floor-active: {supp_active} "
        f"({100 * supp_active / max(supp_n, 1):.1f}%)"
    )
    print(
        f"skeleton (graph-node) voxels: {skel_n}  floor-active: {skel_active} "
        f"({100 * skel_active / max(skel_n, 1):.1f}%)"
    )
    n_dead = int((zeroed & support).sum())
    print(f"dead/zeroed voxels within support: {n_dead}")
    supp_active_maxop = int(active_mask_maxop.sum())
    skel_active_maxop = int((floor_active_maxop & skeleton).sum())
    print(
        f"max-operator baseline: support floor-active {supp_active_maxop} "
        f"({100 * supp_active_maxop / max(supp_n, 1):.1f}%), skeleton "
        f"{skel_active_maxop} -> {supp_active - supp_active_maxop} support voxels "
        "rescued by max instead of mean"
    )
    ratio = baseline / np.where(signal_scale > _EPS, signal_scale, np.nan)
    supp_ratio = ratio[support]
    median_ratio = float(np.nanmedian(supp_ratio))
    print(
        f"baseline/signal_scale within support: median={median_ratio:.3f} "
        f"(floor threshold = {floor_frac})"
    )

    # Isolate the problematic voxels and confirm the "S0 ~ 0" claim in ABSOLUTE
    # terms (not just the relative floor trigger): their precontrast S0 should be
    # a tiny fraction of the healthy in-support S0, while their peak signal max|S|
    # stays comparable -- dark before contrast, real peak after -> blowup.
    med_support_s0 = float(np.median(baseline[support]))
    med_support_peak = float(np.median(signal_scale[support]))
    med_active_s0 = float("nan")
    med_active_peak = float("nan")
    if supp_active:
        s0_active = baseline[active_mask]
        med_active_s0 = float(np.median(s0_active))
        med_active_peak = float(np.median(signal_scale[active_mask]))
        print(
            f"floor-active S0 (absolute): median={med_active_s0:.2f} "
            f"p90={np.percentile(s0_active, 90):.2f}  vs in-support median S0="
            f"{med_support_s0:.2f}  (ratio={med_active_s0 / med_support_s0:.3f})"
        )
        print(
            f"floor-active max|S| (absolute): median={med_active_peak:.2f} "
            f"vs in-support median max|S|={med_support_peak:.2f} "
            "-> near-0 baseline but real peak (the blowup)"
        )

    case_dir.mkdir(parents=True, exist_ok=True)

    # The slice-level figures use the slice carrying the most floor-active
    # voxels, so the red overlay is actually visible. With no floor-active
    # voxels anywhere there is no such slice, and the center of the
    # vessel-support z-extent stands in.
    z_support = np.where(support.any(axis=(1, 2)))[0]
    z_center = int((z_support.min() + z_support.max()) // 2)
    if z_arg is not None:
        z = z_arg
    elif supp_active:
        z = int(active_mask.sum(axis=(1, 2)).argmax())
    else:
        z = z_center
    print(f"axial slice z={z} (support z-extent {z_support.min()}..{z_support.max()})")

    _precontrast_panel(
        precontrast,
        support,
        active_mask,
        z,
        precontrast_timepoints,
        case_id,
        case_dir / "plot.png",
    )
    _postcontrast_panel(
        dce_4d,
        support,
        active_mask,
        z,
        timepoints,
        baseline_frame_count,
        case_id,
        case_dir / "postcontrast.png",
    )
    _precontrast_voxel_value_plot(
        precontrast,
        active_mask,
        signal_scale,
        precontrast_timepoints,
        floor_frac,
        case_id,
        case_dir / "precontrast_voxel_values.png",
    )

    row = {
        "case_id": case_id,
        "family": case_id.split("_1_3_")[0],
        "n_timepoints": len(timepoints),
        "baseline_frame_count": baseline_frame_count,
        "precontrast_timepoints": " ".join(str(t) for t in precontrast_timepoints),
        "support_voxels": supp_n,
        "support_floor_active": supp_active,
        "support_floor_active_pct": 100 * supp_active / max(supp_n, 1),
        "skeleton_voxels": skel_n,
        "skeleton_floor_active": skel_active,
        "skeleton_floor_active_pct": 100 * skel_active / max(skel_n, 1),
        "support_floor_active_maxop": supp_active_maxop,
        "support_floor_active_maxop_pct": 100 * supp_active_maxop / max(supp_n, 1),
        "skeleton_floor_active_maxop": skel_active_maxop,
        "rescued_by_max": supp_active - supp_active_maxop,
        "dead_voxels": n_dead,
        "median_baseline_over_peak": median_ratio,
        "median_support_s0": med_support_s0,
        "median_active_s0": med_active_s0,
        "median_support_peak": med_support_peak,
        "median_active_peak": med_active_peak,
        "z_center": z_center,
        "z_slice_figures": z,
    }
    return row


def _precontrast_panel(
    precontrast: np.ndarray,
    support: np.ndarray,
    active_mask: np.ndarray,
    z: int,
    precontrast_timepoints: list[int],
    case_id: str,
    out_path: Path,
) -> None:
    """Every precontrast frame twice: floor-active voxels marked, then pure.

    Top row carries the red floor-active voxels, bottom row is the same image
    untouched, so each marked voxel can be checked against the unobstructed
    version directly below it -- it should sit where that image is black. Every
    panel shares one intensity window computed over the whole precontrast stack,
    so differences between columns are real signal rather than windowing, and
    the per-image in-support median under each column quantifies that.

    The rightmost column is the mean of the frames -- S0 itself, the quantity
    the floor actually compares against, set off by a divider because it is
    derived rather than acquired. A voxel marked red while some individual frame
    looks bright is one the mean is dragging down.
    """
    vmax = max(float(np.percentile(precontrast[:, support], 99)), _EPS)
    ys, xs = np.where(active_mask[z])
    n = len(precontrast_timepoints)
    columns = [
        *(
            (f"frame {i} (t={t})", precontrast[i])
            for i, t in enumerate(precontrast_timepoints)
        ),
        (f"S0 = mean of {n} frames", precontrast.mean(axis=0)),
    ]

    fig, axes = plt.subplots(2, len(columns), figsize=(3.2 * len(columns), 7.2))
    for i, (label, volume) in enumerate(columns):
        med = float(np.median(volume[support]))
        axes[0, i].imshow(volume[z], cmap="gray", vmin=0, vmax=vmax)
        axes[0, i].scatter(xs, ys, c="red", s=14, marker="s", linewidths=0)
        axes[0, i].set_title(f"{label} + floor-active", fontsize=9)
        axes[1, i].imshow(volume[z], cmap="gray", vmin=0, vmax=vmax)
        axes[1, i].set_title(f"{label} pure\nin-support median {med:.1f}", fontsize=9)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"Precontrast frames + S0 at z={z} — {case_id[:36]}\n"
        f"red = floor-active (n={len(xs)} in this slice); "
        f"shared window [0, {vmax:.1f}]"
    )
    fig.tight_layout()
    # Divider between the acquired frames and the derived S0 column, placed once
    # tight_layout has settled the axes positions it is measured from.
    x_divider = (axes[0, n - 1].get_position().x1 + axes[0, n].get_position().x0) / 2
    fig.add_artist(plt.Line2D([x_divider, x_divider], [0.02, 0.90], color="gray", lw=1))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path} ({len(xs)} red voxels, window [0, {vmax:.1f}])")


def _postcontrast_panel(
    dce_4d: np.ndarray,
    support: np.ndarray,
    active_mask: np.ndarray,
    z: int,
    timepoints: list[int],
    baseline_frame_count: int,
    case_id: str,
    out_path: Path,
) -> None:
    """Post-contrast frames at the same slice, laid out like the precontrast panel.

    Samples the post-contrast series at four points -- the first frame after
    contrast, the middle of the series, the last frame, and the frame with the
    highest in-support mean signal (the actual enhancement peak) -- so the
    acquisition can be eyeballed for dropouts, motion, or wrap the precontrast
    frames alone would not reveal. The peak frame can coincide with one of the
    others; its label says so rather than substituting a different frame.

    Top row marks the floor-active voxels, bottom row is the same image pure,
    matching ``_precontrast_panel``. The window here is computed over the FULL
    4D series rather than the precontrast stack, because post-contrast signal is
    far brighter and the precontrast window would saturate every panel; it is
    stated in the title, and it is shared across all panels of this figure so
    frame-to-frame differences remain real signal (AGENTS.md).
    """
    n_post = dce_4d.shape[0] - baseline_frame_count
    if n_post < 1:
        raise ValueError(
            f"case={case_id}: no post-contrast frames "
            f"({dce_4d.shape[0]} timepoints, {baseline_frame_count} precontrast)"
        )
    in_support_mean = np.array(
        [float(dce_4d[i][support].mean()) for i in range(dce_4d.shape[0])]
    )
    first = baseline_frame_count
    last = dce_4d.shape[0] - 1
    middle = baseline_frame_count + n_post // 2
    peak = int(baseline_frame_count + np.argmax(in_support_mean[baseline_frame_count:]))
    columns = [("early", first), ("middle", middle), ("end", last), ("peak", peak)]

    vmax = max(float(np.percentile(dce_4d[:, support], 99)), _EPS)
    ys, xs = np.where(active_mask[z])

    fig, axes = plt.subplots(2, len(columns), figsize=(3.2 * len(columns), 7.2))
    for i, (label, t_idx) in enumerate(columns):
        frame = dce_4d[t_idx, z]
        dupes = [n for n, j in columns if j == t_idx and n != label]
        tag = f" (= {', '.join(dupes)})" if dupes else ""
        axes[0, i].imshow(frame, cmap="gray", vmin=0, vmax=vmax)
        axes[0, i].scatter(xs, ys, c="red", s=14, marker="s", linewidths=0)
        axes[0, i].set_title(
            f"{label}{tag}: frame {t_idx} (t={timepoints[t_idx]})\n+ floor-active",
            fontsize=9,
        )
        axes[1, i].imshow(frame, cmap="gray", vmin=0, vmax=vmax)
        axes[1, i].set_title(
            f"{label} pure\nin-support mean {in_support_mean[t_idx]:.1f}", fontsize=9
        )
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"Post-contrast frames at z={z} — {case_id[:36]}\n"
        f"red = floor-active (n={len(xs)} in this slice); shared window [0, {vmax:.1f}] "
        f"over the full 4D series; {n_post} post-contrast frames"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(
        f"wrote {out_path} (frames early={first} middle={middle} end={last} "
        f"peak={peak}, window [0, {vmax:.1f}])"
    )


def _precontrast_voxel_value_plot(
    precontrast: np.ndarray,
    active_mask: np.ndarray,
    signal_scale: np.ndarray,
    precontrast_timepoints: list[int],
    floor_frac: float,
    case_id: str,
    out_path: Path,
) -> None:
    """Do the floor-active voxels sit at ~0 in every precontrast frame, or one?

    Answers the operator question -- would a max over the precontrast frames
    remove the need for the floor? -- at three zoom levels:

    left    a random sample of at most ``_SAMPLE_VOXELS`` floor-active voxels,
            each a column of its 5 precontrast values, with its mean (the
            current S0), its max, and the floor it is measured against
    middle   two of those voxels frame by frame, values annotated
    right    mean vs max against the floor for every floor-active voxel, which
            is what decides whether max alone clears the threshold

    Sampling is seeded, so the same voxels are drawn on a rerun.
    """
    n_vox = int(active_mask.sum())
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))
    if not n_vox:
        for ax in axes:
            ax.text(0.5, 0.5, "no floor-active voxels", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"Floor-active precontrast values — {case_id[:36]} (none)")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"wrote {out_path} (no floor-active voxels)")
        return

    values = precontrast[:, active_mask]  # (n_frames, n_active)
    floor = floor_frac * signal_scale[active_mask]
    rng = np.random.default_rng(_SAMPLE_SEED)
    sample = rng.choice(n_vox, size=min(_SAMPLE_VOXELS, n_vox), replace=False)
    sample = sample[np.argsort(values[:, sample].mean(axis=0))]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(precontrast_timepoints)))

    ax = axes[0]
    x = np.arange(len(sample))
    for i, t in enumerate(precontrast_timepoints):
        ax.scatter(
            x,
            values[i, sample],
            s=42,
            color=colors[i],
            linewidths=0,
            label=f"frame {i} (t={t})",
        )
    ax.scatter(
        x,
        values[:, sample].mean(axis=0),
        s=70,
        marker="_",
        color="black",
        label="mean = S0",
    )
    ax.scatter(
        x,
        values[:, sample].max(axis=0),
        s=70,
        marker="_",
        color="tab:orange",
        label="max",
    )
    ax.scatter(
        x,
        floor[sample],
        s=70,
        marker="_",
        color="crimson",
        label=f"floor = {floor_frac}·max|S|",
    )
    ax.set_xlabel(f"sampled floor-active voxel ({len(sample)} of {n_vox}, random)")
    ax.set_ylabel("precontrast signal")
    ax.set_title(
        f"{len(sample)} sampled floor-active voxels\nall 5 precontrast values each"
    )
    ax.legend(fontsize=7, loc="upper left")

    ax = axes[1]
    detail = sample[:: max(len(sample) // _DETAIL_VOXELS, 1)][:_DETAIL_VOXELS]
    frame_x = np.arange(len(precontrast_timepoints))
    for k, v in enumerate(detail):
        ax.plot(frame_x, values[:, v], marker="o", lw=1.2, label=f"voxel {v}")
        for i in frame_x:
            ax.annotate(
                f"{values[i, v]:.1f}",
                (i, values[i, v]),
                textcoords="offset points",
                xytext=(0, 7 if k == 0 else -12),
                ha="center",
                fontsize=7,
            )
        ax.axhline(
            floor[v],
            color="crimson",
            ls="--",
            lw=0.9,
            label=f"floor voxel {v} = {floor[v]:.1f}" if k == 0 else None,
        )
    ax.set_xticks(frame_x)
    ax.set_xticklabels([f"{i}\n(t={t})" for i, t in enumerate(precontrast_timepoints)])
    ax.set_xlabel("precontrast frame")
    ax.set_ylabel("precontrast signal")
    ax.set_title(
        f"{len(detail)} voxels frame by frame\n~0 in all frames, or only some?"
    )
    ax.legend(fontsize=7)

    ax = axes[2]
    order = np.argsort(values.mean(axis=0))
    vx = np.arange(n_vox)
    ax.scatter(
        vx,
        values.mean(axis=0)[order],
        s=3,
        color="black",
        linewidths=0,
        label="mean = S0",
    )
    ax.scatter(
        vx,
        values.max(axis=0)[order],
        s=3,
        color="tab:orange",
        linewidths=0,
        label="max",
    )
    ax.plot(
        vx, floor[order], color="crimson", lw=1.0, label=f"floor = {floor_frac}·max|S|"
    )
    still_floored = int((values.max(axis=0) < floor).sum())
    ax.set_xlabel(f"floor-active voxel (n={n_vox}, sorted by mean)")
    ax.set_ylabel("precontrast signal")
    ax.set_title(
        f"mean vs max operator, all {n_vox} floor-active voxels\n"
        f"max still under the floor for {still_floored} ({100 * still_floored / n_vox:.0f}%)"
    )
    ax.legend(fontsize=7, markerscale=3)

    fig.suptitle(
        f"Would a max over the precontrast frames replace the floor? — {case_id[:36]}"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path} (max still below floor for {still_floored}/{n_vox})")


if __name__ == "__main__":
    main()
