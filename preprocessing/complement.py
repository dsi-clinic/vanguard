"""Add a third, independent vessel source on top of the production merged skeleton.

``preprocessing.merge`` combines the HR-mapped and UFAST-direct TC4D skeletons. This
module adds a further, independent route: the MATLAB-translated SegVessel/Jerman
pipeline (``segmentation.matlab_vessel_segmentation``, ported from
``matlab-conv-2/vessel_pipeline`` and validated there across 6 UChicago exams -- both
tc4d-dense and tc4d-sparse cases).

Combine policy (additive only, same "never override" contract as ``preprocessing.merge``):
  - Run SegVessel on this exam's UFAST peak-enhancement and HR subtraction volumes to get
    an independent Jerman-vesselness skeleton (the "MATLAB route").
  - Split the MATLAB route's skeleton into connected components, and classify each
    whole component by how much of it already lies near the merged skeleton.
  - A component mostly near the merged skeleton is "confirmed": both routes agree it is
    a real vessel, so it becomes part of the CALIBRATION reference for the quality gate
    below.
  - A component carrying enough voxels the merged skeleton does not already cover is a
    complement candidate. These two tests are independent, so a component can be both --
    a real vessel the merge caught most of and this route follows further is a good
    example *and* a genuine find. Candidates are NOT required to touch the existing tree
    -- the merged skeleton is known to be incomplete, and a genuinely new, spatially
    separate vessel is exactly the kind of finding this stage exists to recover. Each
    candidate component (>= ``min_component_voxels``, mostly interior to the breast
    mask) is judged on its own merits against two independent per-component statistics,
    both calibrated against real confirmed vessel segments:
      1. Mean Jerman vesselness over the component. Per-voxel vesselness at a
         1-voxel-wide skeletonized centerline is sparse/noisy (>50% of confirmed real
         vessel voxels have exactly 0 response in practice -- calibrating per-voxel
         percentiles against that distribution floors to 0 and filters nothing), but
         averaged over a whole candidate segment it is a real, well-behaved signal
         (component means for confirmed vessel segments span roughly 0.03-0.3, zero
         components at exactly 0 in the exam checked during design).
      2. Elongation: PCA on the component's physical (mm) coordinates, ratio of the
         largest to second-largest spread eigenvalue. A real vessel segment is long
         and thin (one dominant axis); a noise blob is compact in all three directions
         (eigenvalues closer to equal). This directly operationalises "tubular, not
         blobby" -- the actual claim the Jerman filter is designed to make, and the
         reason this MATLAB pipeline exists rather than a plain intensity threshold.
  - This design supersedes an earlier version validated in
    ``matlab-conv-2/vessel_pipeline/complement_filter.py`` (operating point q=15) that
    required touching the existing tree and gated per-voxel; that version passed large,
    fully isolated noise blobs whenever they happened to be big enough (or later,
    attached by a single-voxel toehold) -- see git history on this file for both bugs
    and the isolated-component QC that found them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import SimpleITK as sitk

from preprocessing.dicom import DicomGeometry
from preprocessing.merge import find_ufast_phases
from preprocessing.spatial import resample_to_geometry
from segmentation.matlab_vessel_segmentation import SegVessel

#: Provenance code for a MATLAB-route voxel added by the quality-gated complement
#: filter. Continues the numbering in preprocessing.merge
#: (LABEL_BACKGROUND=0 .. LABEL_KINETICS_ADJACENCY_ADDED=5).
LABEL_MATLAB_COMPLEMENT_ADDED = 6

#: Quality-gate operating point: keep a candidate branch only if its per-branch
#: enhancement, mean vesselness, and elongation are all >= this percentile of the
#: confirmed segments' own per-branch values. Lower = looser (more kept, more risk).
#:
#: Re-swept 2/5/10/15/25/40 on 4 exams (2026-07-28) after the gating unit changed from
#: whole connected components to branches. That re-sweep was necessary, not routine: a
#: percentile describes a distribution's shape, and the population went from 2-9
#: components per exam to 55-569 branches, so the value inherited from the component-era
#: sweep could not be assumed to still hold.
#:
#: Judged against the no-contrast negative control (analysis/qc_complement_vessels.py),
#: which cannot contain vessels, so anything passing there is a false positive by
#: construction. Pooled over the 4 exams:
#:     pct= 2   2657 real vox / 124 null vox
#:     pct= 5   2195 real vox /  17 null vox   <- already leaking
#:     pct=10   1861 real vox /   0 null vox   <- loosest value with a clean control
#:     pct=15   1544 real vox /   0 null vox
#: 10.0 sits exactly at the boundary: loosening to 5 buys ~18% more yield but starts
#: admitting structure from a vessel-free volume, on the same exam (simbiosys...1182)
#: that the motion and visual checks independently flag as this cohort's worst case.
#:
#: Still only 4 exams, and the negative control uses adjacent baseline frames, so it
#: sees less motion than the real subtraction -- it bounds the false-positive rate from
#: below. Re-check before trusting cohort-wide numbers built on this default.
DEFAULT_QUALITY_PERCENTILE = 10.0

#: A MATLAB-route voxel within this many voxels of the merged skeleton counts as
#: lying "near" it. Whole components are then classified by how many of their voxels
#: are near (see DEFAULT_CONFIRMED_OVERLAP_FRACTION).
DEFAULT_TC_TOLERANCE_VOXELS = 2

#: A MATLAB-route COMPONENT with at least this fraction of its voxels near the merged
#: skeleton is trusted as a real vessel and joins the CALIBRATION reference.
#:
#: Being a good calibration example and having something new to contribute are separate
#: questions, so they get separate rules (see DEFAULT_MIN_NOVEL_VOXELS). Tying both to
#: one threshold has no workable setting: a high bar starves the calibration set, and a
#: low bar files a segment that runs along a known vessel and then continues somewhere
#: new under "already known" and discards the extension -- the exact loss this stage
#: exists to prevent. A component may therefore be both a calibration example and a
#: candidate; with a 10th-percentile operating point over several components, one
#: member barely moves the bar it is judged against.
#:
#: Set from the observed distribution rather than by guess: on a real exam the
#: components' overlap fractions had median 0.63, with none at all above 0.9 (see the
#: 2026-07-27 holistic-review LAB_NOTEBOOK entry). A 0.9 bar left zero calibration
#: components and the stage added nothing.
DEFAULT_CONFIRMED_OVERLAP_FRACTION = 0.5

#: A candidate component needs at least this fraction of its voxels in the eroded
#: breast interior. Judging the fraction rather than clipping each component to the
#: interior matters: clipping would cut components at the margin and shorten them
#: before the size and elongation gates see them, while a boundary-hugging artifact --
#: the thing edge_margin_voxels exists to reject -- fails this outright.
DEFAULT_MIN_INTERIOR_FRACTION = 0.5

#: A complement candidate within this many voxels of the breast-mask boundary is
#: rejected regardless of its quality-gate score -- classical edge/vesselness filters
#: reliably fire on the hard mask boundary next to bright chest signal, not on real
#: vessels (matlab-conv-2's "toward chest not breast" finding).
DEFAULT_EDGE_MARGIN_VOXELS = 3

#: Minimum connected-component size (26-connectivity) for a candidate component to be
#: considered at all -- too few voxels for the elongation PCA to be meaningful, and
#: not enough signal to trust a mean-vesselness/enhancement estimate.
DEFAULT_MIN_COMPONENT_VOXELS = 8

#: A component is a CANDIDATE if it has at least this many voxels the merged skeleton
#: does not already cover. Anything less is either already known or too small to judge,
#: and matching DEFAULT_MIN_COMPONENT_VOXELS keeps it consistent with the size gate.
DEFAULT_MIN_NOVEL_VOXELS = DEFAULT_MIN_COMPONENT_VOXELS

#: Breast-probability threshold applied to the mean HR breast-model output.
BREAST_MASK_PROBABILITY_THRESHOLD = 0.5

#: Below this many confirmed (on-merged-skeleton) voxels, there isn't enough data to
#: decompose into a reliable calibration set of real vessel components, so the
#: complement filter bails out rather than gating on a handful of segments.
MIN_CONFIRMED_VOXELS_FOR_CALIBRATION = 20

#: Below this many confirmed calibration COMPONENTS (as opposed to voxels), the
#: per-component percentile thresholds (elongation, mean vesselness, mean enhancement)
#: are too noisy to trust -- same bail-out rationale as
#: MIN_CONFIRMED_VOXELS_FOR_CALIBRATION, at the component level.
MIN_CONFIRMED_COMPONENTS_FOR_CALIBRATION = 5

#: Elongation of a single-voxel-wide component is undefined (PCA needs >=3 non-
#: collinear points to estimate a meaningful second axis); such components fall back
#: to this value, which is below any realistic tube-vs-blob threshold, so they're
#: rejected by the elongation gate rather than by a NaN/crash.
_MIN_POINTS_FOR_ELONGATION = 3
_UNDEFINED_ELONGATION = 0.0

#: A skeleton voxel with this many or more 26-neighbours is where branches meet, so it
#: ends one branch and starts the next (a plain run of vessel has two neighbours).
_JUNCTION_MIN_NEIGHBOURS = 3

#: The Jerman filter normalises its whole output by a single global maximum Hessian
#: eigenvalue, so one sharp non-vessel structure rescales every real vessel toward
#: zero. Multiplying the volume by a hard breast mask below manufactures exactly such
#: a structure at the mask boundary: measured across four exams and five scales, that
#: boundary won the maximum every time, and 32-66% of the voxels on those exams'
#: independently-derived HR-TC4D skeletons scored exactly zero vesselness as a result.
#: The mask eroded by this many voxels is handed to SegVessel as the region where its
#: normalising maxima may be measured, which keeps the boundary out of the statistic
#: without changing the rule. Chosen a little wider than DEFAULT_EDGE_MARGIN_VOXELS
#: because a single voxel of leftover boundary is enough to set a maximum.
VESSELNESS_NORMALISATION_EROSION_VOXELS = 5

#: HR-branch subtraction is a single peak-minus-phase0 volume. HR model inputs are
#: already z-scored (not raw signal) -- SegVessel's own contrast normalisation
#: (seg_vessel.py) renormalises before the Jerman filter runs, so this is a documented
#: approximation of a raw-signal HR subtraction, same as
#: matlab-conv-2/vessel_pipeline/run_case_uchicago.py used.
_HR_BASELINE_PHASE_INDEX = 0


def _load_dyn_subtraction(
    preprocessing_root: Path, exam_id: str, baseline_frame_count: int
) -> np.ndarray:
    """Peak-minus-baseline UFAST enhancement volume, (y,x,z), clipped at 0.

    Same peak/baseline split as preprocessing.merge.compute_ufast_peak_snr, but returns
    the raw enhancement magnitude (not a per-voxel SNR ratio): SegVessel's own
    background-subtraction + contrast normalisation needs the raw enhancement scale.
    """
    phases = find_ufast_phases(preprocessing_root, exam_id)
    signal_tzyx = np.stack(
        [sitk.GetArrayFromImage(sitk.ReadImage(str(p))) for p in phases], axis=0
    ).astype(np.float32)
    baseline_mean = signal_tzyx[:baseline_frame_count].mean(axis=0)
    peak = signal_tzyx[baseline_frame_count:].max(axis=0)
    sub_zyx = np.clip(peak - baseline_mean, 0, None)
    return np.transpose(sub_zyx, (1, 2, 0))  # (z,y,x) -> (y,x,z)


def _load_hr_subtraction(case_root: Path, provenance: dict) -> np.ndarray:
    """Peak-minus-phase0 HR subtraction, physically resampled onto the UFAST grid, (y,x,z).

    HR model inputs (case_root/hr_model_inputs/hr_phase_NNNN_id.npy) are (y,x,z) on the
    native HR grid (preprocessing.model.prepare_hr_phase_for_model's output convention
    is a transpose + intensity transform only, no resizing -- so this is exactly the HR
    DICOM geometry's shape). SegVessel's own HR-to-DYN step (imresize3_nearest, sized by
    an assumed-1:1 in-plane aspect ratio between the two grids) has no notion of physical
    origin, direction, or per-axis spacing -- it silently assumes both grids are
    corner-anchored with identical in-plane pixel spacing, which does not hold here since
    UFAST is resampled onto a 1-mm isotropic grid (preprocessing.spatial) while native HR
    resolution is finer. Resampling onto the UFAST grid here, the same
    DicomGeometry-driven physical mapping every other HR<->UFAST step in this codebase
    uses (see _breast_mask_ufast_yxz below and preprocessing.spatial), makes that internal
    resize a same-shape no-op instead of a silent misalignment. Linear interpolation
    (not nearest) because this is continuous subtraction signal, not a mask.
    """
    hr_dir = case_root / "hr_model_inputs"
    paths = sorted(hr_dir.glob("hr_phase_*_id.npy"))
    if not paths:
        raise FileNotFoundError(f"no HR model inputs found under {hr_dir}")
    phases_yxzt = np.stack([np.load(p).astype(np.float32) for p in paths], axis=3)
    baseline = phases_yxzt[..., _HR_BASELINE_PHASE_INDEX]
    peak = phases_yxzt.max(axis=3)
    sub_hr_zyx = np.transpose(
        np.clip(peak - baseline, 0, None), (2, 0, 1)
    )  # (y,x,z) -> (z,y,x)

    hr_geometry = DicomGeometry.from_dict(provenance["hr_source"]["geometry"])
    ufast_geometry = DicomGeometry.from_dict(provenance["ufast_output_geometry"])
    mapped_zyx = resample_to_geometry(
        sub_hr_zyx, hr_geometry, ufast_geometry, nearest=False
    )
    return np.transpose(mapped_zyx, (1, 2, 0))  # (z,y,x) -> (y,x,z)


def _breast_mask_ufast_yxz(case_root: Path, provenance: dict) -> np.ndarray:
    """Mean breast probability across HR phases, thresholded, mapped onto the UFAST grid."""
    breast_dir = case_root / "hr_breast_predictions"
    paths = sorted(breast_dir.glob("hr_phase_*_id.npy"))
    if not paths:
        raise FileNotFoundError(f"no HR breast predictions found under {breast_dir}")
    mean_prob_yxz = np.mean([np.load(p).astype(np.float32) for p in paths], axis=0)
    hr_breast_zyx = (
        np.transpose(mean_prob_yxz, (2, 0, 1)) > BREAST_MASK_PROBABILITY_THRESHOLD
    )  # (y,x,z) -> (z,y,x)

    hr_geometry = DicomGeometry.from_dict(provenance["hr_source"]["geometry"])
    ufast_geometry = DicomGeometry.from_dict(provenance["ufast_output_geometry"])
    mapped = resample_to_geometry(
        hr_breast_zyx.astype(np.float32), hr_geometry, ufast_geometry, nearest=True
    )
    return np.transpose(
        mapped > BREAST_MASK_PROBABILITY_THRESHOLD, (1, 2, 0)
    )  # (z,y,x) -> (y,x,z)


def _branch_labels(skeleton_yxz: np.ndarray) -> np.ndarray:
    """Split a skeleton into branches: the runs of voxels between junctions.

    Returns a label image, 0 outside the skeleton.

    THE UNIT MATTERS MORE THAN IT LOOKS
    -----------------------------------
    The gate below compares candidate segments against confirmed ones on per-segment
    statistics, so "segment" has to mean the same thing on both sides and there have to
    be enough of them for a percentile to mean anything. Whole connected components fail
    the second test badly: on four real exams the entire MATLAB skeleton came to just
    2-9 components, because the gap-bridging in vessel_morph fuses the vessel tree into
    a handful of large pieces. A 10th percentile over five numbers is not a calibration.

    Branches are the natural unit instead. There are many per exam, each is one
    anatomical run of vessel, and a candidate branch and a confirmed branch are directly
    comparable. It also makes the gate robust to how aggressively the bridging step
    happens to have fused things together, which is an upstream detail the gate should
    not be sensitive to.

    Junction voxels are folded into an adjacent branch so the labelling covers the whole
    skeleton and no voxel is silently dropped.
    """
    skeleton = skeleton_yxz.astype(bool)
    neighbours = (
        ndi.convolve(
            skeleton.astype(np.int8), np.ones((3, 3, 3), dtype=np.int8), mode="constant"
        )
        - skeleton
    )
    junction = skeleton & (neighbours >= _JUNCTION_MIN_NEIGHBOURS)
    labels, _n = ndi.label(skeleton & ~junction, structure=np.ones((3, 3, 3)))

    # Hand each junction voxel to a touching branch (largest label wins, arbitrary but
    # deterministic). Repeat so junction clusters more than one voxel thick fill in.
    remaining = junction & (labels == 0)
    while remaining.any():
        grown = ndi.grey_dilation(labels, footprint=np.ones((3, 3, 3)))
        newly = remaining & (grown > 0)
        if not newly.any():
            # A junction cluster touching no chain at all: give it its own label.
            leftovers, _m = ndi.label(remaining, structure=np.ones((3, 3, 3)))
            labels = np.where(remaining, leftovers + labels.max(), labels)
            break
        labels = np.where(newly, grown, labels)
        remaining = remaining & ~newly
    return labels


def _component_elongation(coords_mm: np.ndarray) -> float:
    """PCA-based tube-vs-blob score: ratio of the largest to second-largest spread axis.

    Large for a component that's long and thin along one dominant direction (a real
    vessel segment); close to 1 for a component with similar extent in all three
    directions (a noise blob). ``coords_mm`` is the component's voxel centers in
    physical (mm) coordinates, so this is spacing-aware (a component elongated only
    because of anisotropic voxel spacing wouldn't fool it).
    """
    if len(coords_mm) < _MIN_POINTS_FOR_ELONGATION:
        return _UNDEFINED_ELONGATION
    centered = coords_mm - coords_mm.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals = np.sort(np.linalg.eigvalsh(cov))  # ascending: smallest .. largest spread
    return float(eigvals[2] / (eigvals[1] + 1e-6))


def _per_component_stats(
    labeled: np.ndarray,
    vesselness_yxz: np.ndarray,
    enhancement_yxz: np.ndarray,
    spacing_yxz: tuple[float, float, float],
    fraction_masks: dict[str, np.ndarray] | None = None,
) -> dict[int, dict[str, float]]:
    """Per-component size, mean vesselness, mean enhancement, elongation, and fractions.

    Summarises every non-zero label in ``labeled``. Each entry in ``fraction_masks``
    adds a ``"frac_<name>"`` field: the fraction of that component's voxels lying
    inside the given boolean volume.

    Implementation note: the work here is proportional to the number of foreground
    VOXELS, not to voxels times components. Reading out one component at a time with
    ``labeled == label`` rescans the whole grid per component, which on a real
    300x300x160 exam measured 31.6 s for 398 components; pulling the foreground out
    once and grouping it by label gives byte-identical numbers in 0.09 s.
    """
    spacing = np.asarray(spacing_yxz, dtype=np.float64)
    foreground = labeled > 0
    coords = np.argwhere(foreground)  # one pass over the grid
    if coords.size == 0:
        return {}
    labels = labeled[foreground]
    vesselness = vesselness_yxz[foreground]
    enhancement = enhancement_yxz[foreground]
    fractions = {
        name: volume[foreground] for name, volume in (fraction_masks or {}).items()
    }

    # Sort the flat voxel list by label so each component is one contiguous slice.
    order = np.argsort(labels, kind="stable")
    labels = labels[order]
    coords = coords[order]
    vesselness = vesselness[order]
    enhancement = enhancement[order]
    fractions = {name: values[order] for name, values in fractions.items()}

    unique_labels, starts = np.unique(labels, return_index=True)
    bounds = np.append(starts, len(labels))

    stats: dict[int, dict[str, float]] = {}
    for index, label in enumerate(unique_labels):
        lo, hi = bounds[index], bounds[index + 1]
        entry = {
            "size": int(hi - lo),
            "mean_vesselness": float(vesselness[lo:hi].mean()),
            "mean_enhancement": float(enhancement[lo:hi].mean()),
            "elongation": _component_elongation(
                coords[lo:hi].astype(np.float64) * spacing
            ),
        }
        for name, values in fractions.items():
            entry[f"frac_{name}"] = float(values[lo:hi].mean())
        stats[int(label)] = entry
    return stats


def filter_complement(
    *,
    matlab_skeleton_yxz: np.ndarray,
    vesselness_yxz: np.ndarray,
    enhancement_yxz: np.ndarray,
    merged_skeleton_yxz: np.ndarray,
    breast_mask_yxz: np.ndarray,
    spacing_yxz: tuple[float, float, float],
    tc_tolerance_voxels: int = DEFAULT_TC_TOLERANCE_VOXELS,
    quality_percentile: float = DEFAULT_QUALITY_PERCENTILE,
    edge_margin_voxels: int = DEFAULT_EDGE_MARGIN_VOXELS,
    min_component_voxels: int = DEFAULT_MIN_COMPONENT_VOXELS,
    confirmed_overlap_fraction: float = DEFAULT_CONFIRMED_OVERLAP_FRACTION,
    min_interior_fraction: float = DEFAULT_MIN_INTERIOR_FRACTION,
    min_novel_voxels: int = DEFAULT_MIN_NOVEL_VOXELS,
) -> tuple[np.ndarray, dict[str, object]]:
    """Per-component quality-gated complement.

    Keep candidate segments that look like real vessel, independent of whether they
    touch the existing skeleton.

    Candidates are NOT required to be spatially connected to the merged skeleton --
    see the module docstring for why (the merged skeleton is known to be incomplete,
    and a genuinely new, separate vessel is exactly what this stage should recover).
    Instead each candidate component is judged on its own per-component mean
    vesselness, mean enhancement, and elongation (tube-vs-blob shape), each calibrated
    against the same statistics computed over the confirmed segments' own components.

    Both populations are whole connected components of the MATLAB route's own skeleton,
    classified by how much of each one already lies near the merged skeleton, so the
    calibration set and the candidate set are the same kind of object.
    """
    tc_near = ndi.binary_dilation(merged_skeleton_yxz, iterations=tc_tolerance_voxels)
    interior = (
        ndi.binary_erosion(breast_mask_yxz, iterations=edge_margin_voxels)
        if edge_margin_voxels > 0
        else breast_mask_yxz
    )

    # Split the MATLAB route's OWN skeleton into branches first, then decide what each
    # branch is. Splitting the voxels first and labelling afterwards cuts segments
    # wherever they cross the tolerance boundary, which biases the comparison two ways:
    # the calibration set becomes truncated stubs while candidates stay whole, and a
    # segment that runs alongside a known vessel and then continues past it is split,
    # leaving its novel part short enough to fail the size and elongation gates. That
    # split segment is precisely the "vessel the merge only partly caught" case this
    # stage exists to recover. See _branch_labels for why branches, not whole connected
    # components, are the right unit.
    labeled = _branch_labels(matlab_skeleton_yxz)
    n_components = int(labeled.max())
    component_stats = _per_component_stats(
        labeled,
        vesselness_yxz,
        enhancement_yxz,
        spacing_yxz,
        fraction_masks={"near_merged": tc_near, "interior": interior},
    )

    # Two independent questions, two independent rules: is this component trustworthy
    # enough to calibrate against, and does it have anything new to offer? A component
    # can answer yes to both -- a real vessel that the merge caught most of and this
    # route follows further is both a good example and a genuine find.
    confirmed_labels = {
        label
        for label, s in component_stats.items()
        if s["frac_near_merged"] >= confirmed_overlap_fraction
    }
    candidate_labels = {
        label
        for label, s in component_stats.items()
        if s["size"] * (1.0 - s["frac_near_merged"]) >= min_novel_voxels
    }
    confirmed = np.isin(labeled, list(confirmed_labels))
    candidate = np.isin(labeled, list(candidate_labels))

    empty_stats = {
        "n_confirmed": int(confirmed.sum()),
        "n_candidates": int(candidate.sum()),
        "n_kept": 0,
    }
    if confirmed.sum() < MIN_CONFIRMED_VOXELS_FOR_CALIBRATION or not candidate_labels:
        return np.zeros_like(matlab_skeleton_yxz), empty_stats

    # Only components with enough voxels for the elongation PCA to mean anything
    # should inform the calibration thresholds themselves.
    calibration = [
        component_stats[label]
        for label in confirmed_labels
        if component_stats[label]["size"] >= _MIN_POINTS_FOR_ELONGATION
    ]
    if len(calibration) < MIN_CONFIRMED_COMPONENTS_FOR_CALIBRATION:
        empty_stats["n_confirmed_components"] = len(calibration)
        return np.zeros_like(matlab_skeleton_yxz), empty_stats

    vesselness_threshold = float(
        np.percentile([s["mean_vesselness"] for s in calibration], quality_percentile)
    )
    enhancement_threshold = float(
        np.percentile([s["mean_enhancement"] for s in calibration], quality_percentile)
    )
    elongation_threshold = float(
        np.percentile([s["elongation"] for s in calibration], quality_percentile)
    )

    keep_labels = {
        label
        for label in candidate_labels
        if (s := component_stats[label])["size"] >= min_component_voxels
        and s["frac_interior"] >= min_interior_fraction
        and s["mean_vesselness"] >= vesselness_threshold
        and s["mean_enhancement"] >= enhancement_threshold
        and s["elongation"] >= elongation_threshold
    }
    # Voxels of a kept component that the merged skeleton already has add nothing, and
    # the caller unions the result anyway; restricting to the genuinely new voxels here
    # keeps `n_kept` an honest count of added coverage.
    kept = np.isin(labeled, list(keep_labels)) & ~merged_skeleton_yxz

    stats = {
        "n_confirmed": int(confirmed.sum()),
        "n_confirmed_components": len(calibration),
        "n_candidates": int(candidate.sum()),
        "n_candidate_components": len(candidate_labels),
        "n_components": n_components,
        "n_kept": int(kept.sum()),
        "n_kept_components": len(keep_labels),
        "vesselness_threshold": round(vesselness_threshold, 4),
        "enhancement_threshold": round(enhancement_threshold, 2),
        "elongation_threshold": round(elongation_threshold, 3),
        "params": {
            "tc_tolerance_voxels": tc_tolerance_voxels,
            "quality_percentile": quality_percentile,
            "edge_margin_voxels": edge_margin_voxels,
            "min_component_voxels": min_component_voxels,
            "confirmed_overlap_fraction": confirmed_overlap_fraction,
            "min_interior_fraction": min_interior_fraction,
            "min_novel_voxels": min_novel_voxels,
        },
    }
    return kept, stats


def complement_exam(
    *,
    preprocessing_root: Path,
    case_root: Path,
    exam_id: str,
    merged_skeleton_zyx: np.ndarray,
    quality_percentile: float = DEFAULT_QUALITY_PERCENTILE,
    tc_tolerance_voxels: int = DEFAULT_TC_TOLERANCE_VOXELS,
    edge_margin_voxels: int = DEFAULT_EDGE_MARGIN_VOXELS,
    min_component_voxels: int = DEFAULT_MIN_COMPONENT_VOXELS,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Run SegVessel + the quality-gated complement filter for one exam.

    Returns ``(final_skeleton_zyx, kept_complement_zyx, provenance)``, all in vanguard's
    native (z,y,x) convention. ``final_skeleton_zyx`` is ``merged_skeleton_zyx`` unioned
    with the kept MATLAB-route complement voxels.
    """
    provenance = json.loads((case_root / "preprocessing_provenance.json").read_text())
    baseline_frame_count = int(provenance["ufast_source"]["baseline_frame_count"])
    ufast_geometry = DicomGeometry.from_dict(provenance["ufast_output_geometry"])
    spacing_yxz = (
        ufast_geometry.spacing_xyz_mm[1],
        ufast_geometry.spacing_xyz_mm[0],
        ufast_geometry.spacing_xyz_mm[2],
    )

    sub_dyn = _load_dyn_subtraction(preprocessing_root, exam_id, baseline_frame_count)
    sub_hr = _load_hr_subtraction(case_root, provenance)
    breast_mask_yxz = _breast_mask_ufast_yxz(case_root, provenance)

    # Restrict SegVessel to the breast, same as
    # matlab-conv-2/vessel_pipeline/run_case_uchicago.py: without this, the classical
    # edge3/Jerman pass also fires on chest wall, heart, and other high-contrast
    # non-breast anatomy, drowning out the real vessel signal (confirmed here -- omitting
    # this mask inflated the raw MATLAB skeleton ~25x over the matlab-conv-2 validated
    # count for the same exam/threshold). sub_hr is already resampled onto the UFAST
    # grid (_load_hr_subtraction), so no separate resize is needed here.
    sub_dyn = sub_dyn * breast_mask_yxz
    sub_hr = sub_hr * breast_mask_yxz

    # The mask multiplication above leaves a step edge that would otherwise capture the
    # Jerman filter's global normalising maxima -- see
    # VESSELNESS_NORMALISATION_EROSION_VOXELS.
    normalisation_roi = ndi.binary_erosion(
        breast_mask_yxz, iterations=VESSELNESS_NORMALISATION_EROSION_VOXELS
    )
    if not normalisation_roi.any():
        raise ValueError(
            f"breast mask for {exam_id} vanishes when eroded by "
            f"{VESSELNESS_NORMALISATION_EROSION_VOXELS} voxels"
        )

    _vmask, morph = SegVessel(
        sub_dyn, sub_hr, spacing_yxz, verbose=False, normalisation_roi=normalisation_roi
    )
    matlab_skeleton_yxz = morph["skel_label"] > 0
    vesselness_yxz = morph["vness_dyn"]

    merged_skeleton_yxz = np.transpose(merged_skeleton_zyx, (1, 2, 0))
    kept_yxz, filter_stats = filter_complement(
        matlab_skeleton_yxz=matlab_skeleton_yxz,
        vesselness_yxz=vesselness_yxz,
        enhancement_yxz=sub_dyn,
        merged_skeleton_yxz=merged_skeleton_yxz,
        breast_mask_yxz=breast_mask_yxz,
        spacing_yxz=spacing_yxz,
        tc_tolerance_voxels=tc_tolerance_voxels,
        quality_percentile=quality_percentile,
        edge_margin_voxels=edge_margin_voxels,
        min_component_voxels=min_component_voxels,
    )

    final_skeleton_yxz = merged_skeleton_yxz | kept_yxz
    final_skeleton_zyx = np.transpose(final_skeleton_yxz, (2, 0, 1))
    kept_zyx = np.transpose(kept_yxz, (2, 0, 1))

    provenance_out = {
        "exam_id": exam_id,
        "route": "matlab_segvessel_complement (quality-gated additive complement)",
        "matlab_skeleton_voxels": int(matlab_skeleton_yxz.sum()),
        "merged_skeleton_voxels_pre_complement": int(merged_skeleton_yxz.sum()),
        "final_skeleton_voxels": int(final_skeleton_yxz.sum()),
        "voxels_added": int(kept_zyx.sum()),
        "filter_stats": filter_stats,
        "provenance_label_zyx_codes": {
            "matlab_complement_added": LABEL_MATLAB_COMPLEMENT_ADDED
        },
    }
    return final_skeleton_zyx, kept_zyx, provenance_out
