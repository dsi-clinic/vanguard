"""Add a third, independent vessel source on top of the production merged skeleton.

``preprocessing.merge`` combines the HR-mapped and UFAST-direct TC4D skeletons. This
module adds a further, independent route: the MATLAB-translated SegVessel/Jerman
pipeline (``segmentation.matlab_vessel_segmentation``, ported from
``matlab-conv-2/vessel_pipeline`` and validated there across 6 UChicago exams -- both
tc4d-dense and tc4d-sparse cases).

Combine policy (additive only, same "never override" contract as ``preprocessing.merge``):
  - Run SegVessel on this exam's UFAST peak-enhancement and HR subtraction volumes to get
    an independent Jerman-vesselness skeleton (the "MATLAB route").
  - A MATLAB-route voxel already near the merged skeleton (within
    ``tc_tolerance_voxels``) is "confirmed" -- both routes agree, so it adds no new
    coverage, but it becomes part of the CALIBRATION reference for the quality gate
    below (the vessel segments already known to be real).
  - A MATLAB-route voxel far from the merged skeleton is a complement candidate. Candidates
    are NOT required to touch the existing tree -- the merged skeleton is known to be
    incomplete, and a genuinely new, spatially separate vessel is exactly the kind of
    finding this stage exists to recover. Instead, each candidate connected component
    (>= ``min_component_voxels``, interior to the breast mask) is judged on its own
    merits against two independent per-component statistics, both calibrated against
    real confirmed vessel segments (the connected components of the "confirmed" voxels
    above, not individual voxels):
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

#: Quality-gate operating point: keep a candidate component only if its per-component
#: enhancement, mean vesselness, and elongation are all >= this percentile of the
#: confirmed segments' own per-component values. Lower = looser (more kept, more risk).
#: Swept 5/10/15/25/35/50 on 5 exams with the new per-component/elongation design
#: (2026-07-27): pct=5 let a blob through on one exam (elongation_threshold collapsed
#: to 2.84 there, vs. 4-10 everywhere else -- visually confirmed as a dense non-tubular
#: mass, not vessels); pct=10 roughly doubled yield over pct=15 with no such collapse
#: on any of the 5 exams and visually clean (thin, branching additions). Not yet
#: validated at the same n=25 scale as the original q=15 -- re-check before trusting
#: cohort-wide numbers built on this default.
DEFAULT_QUALITY_PERCENTILE = 10.0

#: A MATLAB-route voxel within this many voxels of the merged skeleton counts as
#: "confirmed" (both routes agree), and its connected components become the
#: calibration reference below.
DEFAULT_TC_TOLERANCE_VOXELS = 2

#: A complement candidate within this many voxels of the breast-mask boundary is
#: rejected regardless of its quality-gate score -- classical edge/vesselness filters
#: reliably fire on the hard mask boundary next to bright chest signal, not on real
#: vessels (matlab-conv-2's "toward chest not breast" finding).
DEFAULT_EDGE_MARGIN_VOXELS = 3

#: Minimum connected-component size (26-connectivity) for a candidate component to be
#: considered at all -- too few voxels for the elongation PCA to be meaningful, and
#: not enough signal to trust a mean-vesselness/enhancement estimate.
DEFAULT_MIN_COMPONENT_VOXELS = 8

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
    mask: np.ndarray,
    labeled: np.ndarray,
    label_ids: np.ndarray,
    vesselness_yxz: np.ndarray,
    enhancement_yxz: np.ndarray,
    spacing_yxz: tuple[float, float, float],
) -> dict[int, dict[str, float]]:
    """Per-component size, mean vesselness, mean enhancement, and elongation.

    ``label_ids`` are the label values to summarise (restricted to components that
    actually appear in ``mask`` -- e.g. only within the confirmed set, or only within
    the candidate set -- even though ``labeled`` may have been produced from a
    larger/different array).
    """
    spacing = np.asarray(spacing_yxz, dtype=np.float64)
    stats: dict[int, dict[str, float]] = {}
    for label in label_ids:
        component = mask & (labeled == label)
        size = int(component.sum())
        if size == 0:
            continue
        coords_mm = np.argwhere(component).astype(np.float64) * spacing
        stats[int(label)] = {
            "size": size,
            "mean_vesselness": float(vesselness_yxz[component].mean()),
            "mean_enhancement": float(enhancement_yxz[component].mean()),
            "elongation": _component_elongation(coords_mm),
        }
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
    """
    tc_near = ndi.binary_dilation(merged_skeleton_yxz, iterations=tc_tolerance_voxels)
    confirmed = matlab_skeleton_yxz & tc_near  # confirmed by both routes
    candidate = matlab_skeleton_yxz & ~tc_near  # complement candidates

    empty_stats = {
        "n_confirmed": int(confirmed.sum()),
        "n_candidates": int(candidate.sum()),
        "n_kept": 0,
    }
    if confirmed.sum() < MIN_CONFIRMED_VOXELS_FOR_CALIBRATION or not candidate.any():
        return np.zeros_like(matlab_skeleton_yxz), empty_stats

    # Calibration reference: the confirmed voxels' OWN connected components (not the
    # confirmed voxels as a single pool) -- elongation is only meaningful computed per
    # component, and per-component mean vesselness/enhancement is far better behaved
    # than the per-voxel distribution (see module docstring).
    confirmed_labeled, n_confirmed_components = ndi.label(
        confirmed, structure=np.ones((3, 3, 3))
    )
    confirmed_stats = _per_component_stats(
        confirmed,
        confirmed_labeled,
        np.arange(1, n_confirmed_components + 1),
        vesselness_yxz,
        enhancement_yxz,
        spacing_yxz,
    )
    # Only components with enough voxels for the elongation PCA to mean anything
    # should inform the calibration thresholds themselves.
    calibration = [
        s for s in confirmed_stats.values() if s["size"] >= _MIN_POINTS_FOR_ELONGATION
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

    interior = (
        ndi.binary_erosion(breast_mask_yxz, iterations=edge_margin_voxels)
        if edge_margin_voxels > 0
        else breast_mask_yxz
    )
    candidate_interior = candidate & interior

    candidate_labeled, n_candidate_components = ndi.label(
        candidate_interior, structure=np.ones((3, 3, 3))
    )
    candidate_stats = _per_component_stats(
        candidate_interior,
        candidate_labeled,
        np.arange(1, n_candidate_components + 1),
        vesselness_yxz,
        enhancement_yxz,
        spacing_yxz,
    )
    keep_labels = {
        label
        for label, s in candidate_stats.items()
        if s["size"] >= min_component_voxels
        and s["mean_vesselness"] >= vesselness_threshold
        and s["mean_enhancement"] >= enhancement_threshold
        and s["elongation"] >= elongation_threshold
    }
    kept = candidate_interior & np.isin(candidate_labeled, list(keep_labels))

    stats = {
        "n_confirmed": int(confirmed.sum()),
        "n_confirmed_components": len(calibration),
        "n_candidates": int(candidate.sum()),
        "n_candidate_components": n_candidate_components,
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

    _vmask, morph = SegVessel(sub_dyn, sub_hr, spacing_yxz, verbose=False)
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
