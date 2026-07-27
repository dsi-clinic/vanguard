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
    coverage, but its enhancement and Jerman-vesselness values become the CALIBRATION
    reference for the quality gate below (the voxels already known to be real vessel).
  - A MATLAB-route voxel far from the merged skeleton is a complement candidate. It's
    kept only if its peak enhancement AND Jerman vesselness are both >=
    ``quality_percentile``-th percentile of the confirmed voxels' own values, it's
    interior to the breast mask (>= ``edge_margin_voxels`` from the boundary), and it
    belongs to a connected component (seeded together with the confirmed voxels, so a
    candidate branch attached to a confirmed vessel survives) >=
    ``min_component_voxels`` large.
  - This is the exact method validated in
    ``matlab-conv-2/vessel_pipeline/complement_filter.py`` (operating point q=15):
    97-640 added voxels/exam, ~0% of them within 2 voxels of the breast-mask boundary,
    visually confirmed to thread through real vessel structure rather than form blobs
    or track the chest wall/mask edge.
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

#: Quality-gate operating point: keep a complement voxel only if its enhancement and
#: Jerman vesselness are >= this percentile of the confirmed (on-merged-skeleton)
#: voxels' own values. Validated in matlab-conv-2 across 6 UChicago exams (both
#: tc4d-dense and tc4d-sparse): q=15 was the recommended balance of yield vs purity.
DEFAULT_QUALITY_PERCENTILE = 15.0

#: A MATLAB-route voxel within this many voxels of the merged skeleton counts as
#: "confirmed" (both routes agree) rather than a complement candidate.
DEFAULT_TC_TOLERANCE_VOXELS = 2

#: A complement candidate within this many voxels of the breast-mask boundary is
#: rejected regardless of its quality-gate score -- classical edge/vesselness filters
#: reliably fire on the hard mask boundary next to bright chest signal, not on real
#: vessels (matlab-conv-2's "toward chest not breast" finding).
DEFAULT_EDGE_MARGIN_VOXELS = 3

#: Minimum connected-component size (26-connectivity, seeded together with confirmed
#: voxels so a real complement branch attached to a confirmed vessel survives) for a
#: gated candidate to be kept.
DEFAULT_MIN_COMPONENT_VOXELS = 8

#: Breast-probability threshold applied to the mean HR breast-model output.
BREAST_MASK_PROBABILITY_THRESHOLD = 0.5

#: Below this many confirmed (on-merged-skeleton) voxels, the quality-gate percentile
#: thresholds have too little data to calibrate against, so the complement filter bails
#: out rather than gating on a handful of voxels' values.
MIN_CONFIRMED_VOXELS_FOR_CALIBRATION = 20

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


def filter_complement(
    *,
    matlab_skeleton_yxz: np.ndarray,
    vesselness_yxz: np.ndarray,
    enhancement_yxz: np.ndarray,
    merged_skeleton_yxz: np.ndarray,
    breast_mask_yxz: np.ndarray,
    tc_tolerance_voxels: int = DEFAULT_TC_TOLERANCE_VOXELS,
    quality_percentile: float = DEFAULT_QUALITY_PERCENTILE,
    edge_margin_voxels: int = DEFAULT_EDGE_MARGIN_VOXELS,
    min_component_voxels: int = DEFAULT_MIN_COMPONENT_VOXELS,
) -> tuple[np.ndarray, dict[str, object]]:
    """Quality-gated complement: keep MATLAB-route voxels as good as confirmed ones.

    Direct port of matlab-conv-2/vessel_pipeline/complement_filter.py::filter_complement,
    re-pointed at the production merged skeleton (preprocessing.merge) instead of the
    exploratory combined_tc4d reference it was originally validated against.
    """
    tc_near = ndi.binary_dilation(merged_skeleton_yxz, iterations=tc_tolerance_voxels)
    on_merged = matlab_skeleton_yxz & tc_near  # confirmed by both routes
    candidate = matlab_skeleton_yxz & ~tc_near  # complement candidates

    if on_merged.sum() < MIN_CONFIRMED_VOXELS_FOR_CALIBRATION or not candidate.any():
        return np.zeros_like(matlab_skeleton_yxz), {
            "n_confirmed": int(on_merged.sum()),
            "n_candidates": int(candidate.sum()),
            "n_kept": 0,
        }

    vesselness_threshold = float(
        np.percentile(vesselness_yxz[on_merged], quality_percentile)
    )
    enhancement_threshold = float(
        np.percentile(enhancement_yxz[on_merged], quality_percentile)
    )
    interior = (
        ndi.binary_erosion(breast_mask_yxz, iterations=edge_margin_voxels)
        if edge_margin_voxels > 0
        else breast_mask_yxz
    )
    gated = (
        candidate
        & (vesselness_yxz >= vesselness_threshold)
        & (enhancement_yxz >= enhancement_threshold)
        & interior
    )

    # Seed connectivity with the confirmed voxels too, so a gated candidate branch
    # attached to an already-trusted vessel survives the component-size floor.
    seed = gated | on_merged
    labeled, _ = ndi.label(seed, structure=np.ones((3, 3, 3)))
    keep_labels = {
        label
        for label in np.unique(labeled[gated])
        if label and (labeled == label).sum() >= min_component_voxels
    }
    kept = gated & np.isin(labeled, list(keep_labels))

    stats = {
        "n_confirmed": int(on_merged.sum()),
        "n_candidates": int(candidate.sum()),
        "n_kept": int(kept.sum()),
        "vesselness_threshold": round(vesselness_threshold, 4),
        "enhancement_threshold": round(enhancement_threshold, 2),
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
