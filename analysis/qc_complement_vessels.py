r"""QC the MATLAB-complement additions: are they vessels, or motion and noise?

The complement stage adds skeleton voxels the production merge does not have. By
construction those additions have no second opinion agreeing with them, so "it found
something" is not evidence it found a vessel. This script runs three checks that can
each actually fail, and writes the figures to look at alongside them.

CHECK 1 -- NEGATIVE CONTROL (the decisive one)
    Rebuild the stage's inputs out of pre-contrast baseline frames only: the difference
    between two frames acquired BEFORE any contrast arrived. That volume has the same
    noise statistics and the same patient motion as the real subtraction, and no
    vessels, because nothing has enhanced yet. Run the identical SegVessel pipeline on
    it, then judge its branches against the SAME thresholds the real exam calibrated.
    Every branch that passes is a false positive: the bar admitted a structure that
    cannot be a vessel. This is the check that can catch "we are just picking up
    movement", which no amount of staring at the real exam can do.

    Honest limitation: consecutive baseline frames are closer together in time than
    baseline is to peak, so the motion between them is smaller than the motion the real
    subtraction sees. The control is therefore optimistic, and a clean result bounds the
    problem from below rather than proving its absence.

CHECK 2 -- ENHANCEMENT KINETICS
    A vessel fills with contrast: its signal rises sharply after the bolus and stays up.
    Noise stays flat; motion artifacts swing around without a consistent direction. The
    mean time-course of the added voxels is compared against three references measured
    on the same exam -- voxels the merge already calls vessel (should match), random
    breast tissue (should be much flatter), and voxels near the breast-mask boundary
    (the known artifact population).

CHECK 3 -- MOTION-ARTIFACT PROXY
    Motion shows up where the image has strong edges: skin, chest wall, fat-gland
    boundaries. If the additions sat on anatomical edges far more than confirmed vessels
    do, that would be the signature of motion rather than vasculature. Reported as the
    baseline-image gradient magnitude at added voxels versus at confirmed vessel voxels.

Usage::

    micromamba run -n vanguard python -m analysis.qc_complement_vessels \\
        --preprocessing-root /gpfs/data/.../preprocessing_out \\
        --output-dir qc_complement \\
        --exam-id <exam> [--exam-id <exam> ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.ndimage as ndi  # noqa: E402
import SimpleITK as sitk  # noqa: E402

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from preprocessing.complement import (  # noqa: E402
    _MIN_POINTS_FOR_ELONGATION,
    DEFAULT_EDGE_MARGIN_VOXELS,
    DEFAULT_MIN_COMPONENT_VOXELS,
    DEFAULT_MIN_INTERIOR_FRACTION,
    DEFAULT_QUALITY_PERCENTILE,
    DEFAULT_TC_TOLERANCE_VOXELS,
    VESSELNESS_NORMALISATION_EROSION_VOXELS,
    _branch_labels,
    _breast_mask_ufast_yxz,
    _load_dyn_subtraction,
    _load_hr_subtraction,
    _per_component_stats,
    filter_complement,
)
from preprocessing.dicom import DicomGeometry  # noqa: E402
from preprocessing.merge import find_ufast_phases  # noqa: E402
from segmentation.matlab_vessel_segmentation import SegVessel  # noqa: E402

#: Neutral grey for reference populations, one accent for the thing under test. Colour
#: is carrying "is this the population being judged?", nothing else.
_ADDED_COLOUR = "#d1495b"
_CONFIRMED_COLOUR = "#3f6f8f"
_NEUTRAL = "#8d8d8d"
_MUTED = "#c7c7c7"

#: How many random in-breast voxels to sample for the "ordinary tissue" reference.
_TISSUE_SAMPLE = 20000


def _load_signal_tzyx(preprocessing_root: Path, exam_id: str) -> np.ndarray:
    phases = find_ufast_phases(preprocessing_root, exam_id)
    return np.stack(
        [sitk.GetArrayFromImage(sitk.ReadImage(str(p))) for p in phases], axis=0
    ).astype(np.float32)


def _null_subtraction_yxz(
    signal_tzyx: np.ndarray, baseline_frame_count: int
) -> np.ndarray:
    """A no-contrast stand-in for the peak-minus-baseline enhancement volume.

    Uses the last pre-contrast frame minus the mean of the earlier ones, mirroring the
    real construction (`peak minus baseline mean`) with the "peak" taken from a frame
    that cannot contain contrast. Same clipping at zero, so the downstream filter sees
    the same kind of one-sided volume it normally does.
    """
    if baseline_frame_count < 2:
        raise ValueError(
            f"need >=2 baseline frames for a negative control, got {baseline_frame_count}"
        )
    pseudo_peak = signal_tzyx[baseline_frame_count - 1]
    pseudo_baseline = signal_tzyx[: baseline_frame_count - 1].mean(axis=0)
    null_zyx = np.clip(pseudo_peak - pseudo_baseline, 0, None)
    return np.transpose(null_zyx, (1, 2, 0))


def _run_segvessel(sub_dyn, sub_hr, breast, spacing):
    roi = ndi.binary_erosion(breast, iterations=VESSELNESS_NORMALISATION_EROSION_VOXELS)
    _vmask, morph = SegVessel(
        sub_dyn * breast,
        sub_hr * breast,
        spacing,
        verbose=False,
        normalisation_roi=roi,
    )
    return morph["skel_label"] > 0, morph["vness_dyn"]


def _branches_passing(
    skeleton, vesselness, enhancement, spacing, interior, thresholds
) -> tuple[int, int, int, np.ndarray]:
    """Judge a skeleton's branches against thresholds calibrated somewhere else.

    Returns (n_branches, n_passing_branches, n_passing_voxels, passing_mask). Used for
    the negative control: the real exam sets the bar, and this counts what a vessel-free
    volume manages to get past it.
    """
    labels = _branch_labels(skeleton)
    if labels.max() == 0:
        return 0, 0, 0, np.zeros_like(skeleton)
    stats = _per_component_stats(
        labels, vesselness, enhancement, spacing, fraction_masks={"interior": interior}
    )
    passing = [
        label
        for label, s in stats.items()
        if s["size"] >= DEFAULT_MIN_COMPONENT_VOXELS
        and s["frac_interior"] >= DEFAULT_MIN_INTERIOR_FRACTION
        and s["mean_vesselness"] >= thresholds["mean_vesselness"]
        and s["mean_enhancement"] >= thresholds["mean_enhancement"]
        and s["elongation"] >= thresholds["elongation"]
    ]
    n_voxels = int(sum(stats[label]["size"] for label in passing))
    return len(stats), len(passing), n_voxels, np.isin(labels, passing)


def _calibrated_thresholds(skeleton, vesselness, enhancement, spacing, merged) -> dict:
    """Reproduce the gate's own calibration so the control can be judged by it."""
    tc_near = ndi.binary_dilation(merged, iterations=DEFAULT_TC_TOLERANCE_VOXELS)
    labels = _branch_labels(skeleton)
    stats = _per_component_stats(
        labels, vesselness, enhancement, spacing, fraction_masks={"near": tc_near}
    )
    calibration = [
        s
        for s in stats.values()
        if s["frac_near"] >= 0.5 and s["size"] >= _MIN_POINTS_FOR_ELONGATION
    ]
    if not calibration:
        return {}
    return {
        key: float(
            np.percentile([s[key] for s in calibration], DEFAULT_QUALITY_PERCENTILE)
        )
        for key in ("mean_vesselness", "mean_enhancement", "elongation")
    }


def _percent_enhancement_curve(signal_tzyx, mask_yxz, baseline_frame_count):
    """Mean signal over a voxel set, as % change from its own pre-contrast baseline."""
    mask_zyx = np.transpose(mask_yxz, (2, 0, 1))
    if not mask_zyx.any():
        return None
    series = signal_tzyx[:, mask_zyx].mean(axis=1)
    baseline = series[:baseline_frame_count].mean()
    if abs(baseline) < 1e-6:
        return None
    return 100.0 * (series - baseline) / baseline


def _random_tissue_mask(breast, exclude, rng):
    """Ordinary in-breast voxels, away from anything already called vessel."""
    candidates = np.argwhere(breast & ~exclude)
    if len(candidates) == 0:
        return np.zeros_like(breast)
    pick = candidates[
        rng.choice(len(candidates), min(_TISSUE_SAMPLE, len(candidates)), replace=False)
    ]
    out = np.zeros_like(breast)
    out[pick[:, 0], pick[:, 1], pick[:, 2]] = True
    return out


def _mip_panel(ax, volume_yxz, overlays, title):
    """Maximum-intensity projection down z with point overlays."""
    ax.imshow(volume_yxz.max(axis=2).T, cmap="gray", origin="lower")
    for mask, colour, label, size in overlays:
        pts = np.argwhere(mask.any(axis=2))
        if len(pts):
            ax.scatter(
                pts[:, 0], pts[:, 1], s=size, c=colour, label=label, linewidths=0
            )
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def qc_exam(preprocessing_root: Path, exam_id: str, output_dir: Path) -> dict:
    case_root = preprocessing_root / "work" / exam_id
    provenance = json.loads((case_root / "preprocessing_provenance.json").read_text())
    baseline_frame_count = int(provenance["ufast_source"]["baseline_frame_count"])
    geometry = DicomGeometry.from_dict(provenance["ufast_output_geometry"])
    spacing = (
        geometry.spacing_xyz_mm[1],
        geometry.spacing_xyz_mm[0],
        geometry.spacing_xyz_mm[2],
    )

    skel_paths = sorted(
        (preprocessing_root / "centerlines").glob(
            f"*/{exam_id}/*skeleton_4d_exam_mask*.npy"
        )
    )
    if not skel_paths:
        raise FileNotFoundError(f"no centerline skeleton for {exam_id}")
    merged = np.transpose(np.load(skel_paths[0]).astype(bool), (1, 2, 0))

    sub_dyn = _load_dyn_subtraction(preprocessing_root, exam_id, baseline_frame_count)
    sub_hr = _load_hr_subtraction(case_root, provenance)
    breast = _breast_mask_ufast_yxz(case_root, provenance)
    interior = ndi.binary_erosion(breast, iterations=DEFAULT_EDGE_MARGIN_VOXELS)

    print(f"  [{exam_id[:36]}] real SegVessel...", flush=True)
    skeleton, vesselness = _run_segvessel(sub_dyn, sub_hr, breast, spacing)
    added, stats = filter_complement(
        matlab_skeleton_yxz=skeleton,
        vesselness_yxz=vesselness,
        enhancement_yxz=sub_dyn * breast,
        merged_skeleton_yxz=merged,
        breast_mask_yxz=breast,
        spacing_yxz=spacing,
    )
    thresholds = _calibrated_thresholds(
        skeleton, vesselness, sub_dyn * breast, spacing, merged
    )

    # ---- CHECK 1: negative control -------------------------------------------------
    signal_tzyx = _load_signal_tzyx(preprocessing_root, exam_id)
    null_sub = _null_subtraction_yxz(signal_tzyx, baseline_frame_count)
    print("  null-control SegVessel...", flush=True)
    null_skeleton, null_vesselness = _run_segvessel(null_sub, null_sub, breast, spacing)
    if thresholds:
        n_null_branches, n_null_pass, n_null_voxels, null_passing_mask = (
            _branches_passing(
                null_skeleton,
                null_vesselness,
                null_sub * breast,
                spacing,
                interior,
                thresholds,
            )
        )
        _n_real_branches, _n_real_pass, n_real_voxels, _real_mask = _branches_passing(
            skeleton, vesselness, sub_dyn * breast, spacing, interior, thresholds
        )
    else:
        n_null_branches = n_null_pass = n_null_voxels = 0
        n_real_voxels = 0
        null_passing_mask = np.zeros_like(merged)

    # ---- CHECK 2: kinetics ---------------------------------------------------------
    confirmed = merged & ndi.binary_dilation(skeleton, iterations=1)
    rng = np.random.default_rng(0)
    tissue = _random_tissue_mask(
        breast, ndi.binary_dilation(merged, iterations=3) | added, rng
    )
    edge_band = breast & ~interior
    curves = {
        "added": _percent_enhancement_curve(signal_tzyx, added, baseline_frame_count),
        "confirmed vessel": _percent_enhancement_curve(
            signal_tzyx, confirmed, baseline_frame_count
        ),
        "breast tissue": _percent_enhancement_curve(
            signal_tzyx, tissue, baseline_frame_count
        ),
        "mask-edge band": _percent_enhancement_curve(
            signal_tzyx, edge_band, baseline_frame_count
        ),
    }

    # ---- CHECK 3: motion proxy -----------------------------------------------------
    baseline_mean_zyx = signal_tzyx[:baseline_frame_count].mean(axis=0)
    gradient = np.transpose(
        ndi.gaussian_gradient_magnitude(baseline_mean_zyx, sigma=1.0), (1, 2, 0)
    )
    scale = np.percentile(gradient[breast], 95) or 1.0
    grad_added = gradient[added] / scale if added.any() else np.array([])
    grad_confirmed = gradient[confirmed] / scale if confirmed.any() else np.array([])
    grad_tissue = gradient[tissue] / scale if tissue.any() else np.array([])

    # Persist the masks so the 3-D interactive viewer can be built from this same run
    # rather than paying for another SegVessel pass.
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{exam_id}_added_zyx.npy", np.transpose(added, (2, 0, 1)))
    np.save(output_dir / f"{exam_id}_merged_zyx.npy", np.transpose(merged, (2, 0, 1)))
    np.save(
        output_dir / f"{exam_id}_null_passing_zyx.npy",
        np.transpose(null_passing_mask, (2, 0, 1)),
    )

    _write_figures(
        exam_id,
        output_dir,
        sub_dyn * breast,
        merged,
        added,
        curves,
        baseline_frame_count,
        (grad_added, grad_confirmed, grad_tissue),
        (n_real_voxels, n_null_voxels),
    )

    peak_added = (
        float(np.median(curves["added"][baseline_frame_count:].max()))
        if curves["added"] is not None
        else float("nan")
    )
    peak_conf = (
        float(np.median(curves["confirmed vessel"][baseline_frame_count:].max()))
        if curves["confirmed vessel"] is not None
        else float("nan")
    )
    peak_tissue = (
        float(np.median(curves["breast tissue"][baseline_frame_count:].max()))
        if curves["breast tissue"] is not None
        else float("nan")
    )

    return {
        "exam_id": exam_id,
        "merged_voxels": int(merged.sum()),
        "matlab_skeleton_voxels": int(skeleton.sum()),
        "added_voxels": int(added.sum()),
        "added_branches": int(stats.get("n_kept_components", 0)),
        "null_skeleton_voxels": int(null_skeleton.sum()),
        "null_branches": n_null_branches,
        "null_branches_passing": n_null_pass,
        "null_voxels_passing": n_null_voxels,
        "real_voxels_passing": n_real_voxels,
        "false_positive_ratio": (
            round(n_null_voxels / n_real_voxels, 4) if n_real_voxels else float("nan")
        ),
        "peak_pct_enhancement_added": round(peak_added, 1),
        "peak_pct_enhancement_confirmed": round(peak_conf, 1),
        "peak_pct_enhancement_tissue": round(peak_tissue, 1),
        "median_edge_gradient_added": round(float(np.median(grad_added)), 3)
        if grad_added.size
        else float("nan"),
        "median_edge_gradient_confirmed": round(float(np.median(grad_confirmed)), 3)
        if grad_confirmed.size
        else float("nan"),
        "median_edge_gradient_tissue": round(float(np.median(grad_tissue)), 3)
        if grad_tissue.size
        else float("nan"),
    }


def _write_figures(
    exam_id,
    output_dir,
    enhancement,
    merged,
    added,
    curves,
    baseline_frame_count,
    gradients,
    null_counts,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14, 8.5))
    grid = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.2)

    ax = fig.add_subplot(grid[0, 0])
    _mip_panel(
        ax,
        enhancement,
        [
            (merged, _MUTED, "merge (known)", 0.35),
            (added, _ADDED_COLOUR, "complement added", 1.6),
        ],
        "Enhancement MIP with additions",
    )
    ax.legend(loc="upper right", fontsize=7, framealpha=0.85, markerscale=4)

    ax = fig.add_subplot(grid[0, 1])
    for name, colour, width in (
        ("confirmed vessel", _CONFIRMED_COLOUR, 1.8),
        ("added", _ADDED_COLOUR, 2.4),
        ("mask-edge band", _NEUTRAL, 1.2),
        ("breast tissue", _MUTED, 1.2),
    ):
        curve = curves.get(name)
        if curve is None:
            continue
        ax.plot(curve, color=colour, linewidth=width, label=name)
    ax.axvline(baseline_frame_count - 0.5, color="black", linestyle=":", linewidth=1)
    ax.annotate(
        "contrast arrives",
        xy=(baseline_frame_count - 0.5, ax.get_ylim()[1]),
        fontsize=7,
        ha="left",
        va="top",
    )
    ax.set_xlabel("UFAST phase")
    ax.set_ylabel("% change from own baseline")
    ax.set_title("Check 2: do the additions fill with contrast?", fontsize=9)
    ax.legend(fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[0, 2])
    grad_added, grad_confirmed, grad_tissue = gradients
    data = [g for g in (grad_confirmed, grad_added, grad_tissue) if g.size]
    labels = [
        n
        for n, g in zip(
            ("confirmed\nvessel", "added", "breast\ntissue"),
            (grad_confirmed, grad_added, grad_tissue),
        )
        if g.size
    ]
    if data:
        parts = ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True)
        for patch, colour in zip(
            parts["boxes"], (_CONFIRMED_COLOUR, _ADDED_COLOUR, _MUTED)
        ):
            patch.set_facecolor(colour)
            patch.set_alpha(0.75)
    ax.set_ylabel("baseline edge strength (normalised)")
    ax.set_title("Check 3: do additions sit on motion edges?", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[1, 0])
    real_voxels, null_voxels = null_counts
    ax.bar(
        ["real exam", "no-contrast\ncontrol"],
        [real_voxels, null_voxels],
        color=[_ADDED_COLOUR, _NEUTRAL],
    )
    for i, value in enumerate((real_voxels, null_voxels)):
        ax.annotate(f"{value}", xy=(i, value), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("voxels passing the same bar")
    ax.set_title("Check 1: negative control", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(grid[1, 1:])
    _mip_panel(
        ax,
        enhancement,
        [(added, _ADDED_COLOUR, "complement added", 3.0)],
        "Additions alone, on the enhancement MIP",
    )

    fig.suptitle(f"Complement QC — {exam_id[:52]}", fontsize=11)
    out = output_dir / f"{exam_id}_complement_qc.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocessing-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exam-id", action="append", required=True)
    args = parser.parse_args()

    rows = []
    for exam_id in args.exam_id:
        print(f"=== {exam_id} ===", flush=True)
        try:
            rows.append(qc_exam(args.preprocessing_root, exam_id, args.output_dir))
        except Exception as exc:  # noqa: BLE001 - QC survey, report and continue
            print(f"  FAILED {type(exc).__name__}: {exc}", flush=True)

    if rows:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = args.output_dir / "complement_qc_summary.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {csv_path}")
        for row in rows:
            print(
                f"  {row['exam_id'][:36]}  added={row['added_voxels']:5d}  "
                f"null-passing={row['null_voxels_passing']:5d}  "
                f"FP ratio={row['false_positive_ratio']}"
            )


if __name__ == "__main__":
    main()
