"""Merge the HR-mapped skeleton with the UFAST-direct skeleton (production route).

Merge policy (HR is authoritative; UFAST only adds, never overrides):
  - Every HR-mapped skeleton voxel is always kept.
  - A UFAST-direct voxel within ``dilation_radius_voxels`` of an HR voxel is
    "confirmed" (both routes agree on that vessel) but doesn't add any new
    coverage -- it's HR's voxel that gets kept either way.
  - A UFAST-direct voxel with no nearby HR voxel is a merge candidate only if
    it belongs to a connected component at least ``vessel_like_min_voxels``
    large -- this drops UFAST-only noise specks before they can pollute the
    merge.
  - A UFAST-only component that fails the size filter gets a second, additive
    chance (``enable_kinetics_filter``, on by default): it's kept anyway if
    its mean per-voxel contrast-enhancement peak-SNR (computed from the raw
    motion-corrected UFAST signal) clears ``kinetics_min_peak_snr`` AND it's
    spatially continuous with already-trusted vasculature (HR skeleton or a
    size-qualifying UFAST component) within ``kinetics_adjacency_radius_voxels``.
    Both conditions are required: amplitude alone is a weak discriminator in
    this data (typical background tissue commonly shows peak-SNR 3-4+), so
    spatial continuity is what actually keeps this trustworthy.
  - The candidate mask (HR | qualifying UFAST-only) is re-cleaned with TC4D's
    own internal topology finalization (``graph_extraction.tc4d._finalize_exam_mask_topology``).
    That cleanup has no mechanism to preserve trusted voxels on its own (it
    can drop a whole component via its automatic support-score threshold), so
    the raw HR skeleton -- and any kinetics-adjacency-accepted voxels, which
    clear their own independent evidence bar -- are unioned back in
    unconditionally after cleanup runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import SimpleITK as sitk

from graph_extraction.tc4d import _finalize_exam_mask_topology

#: Voxel-radius proximity match for "does this UFAST voxel land near an HR
#: voxel" -- exact voxel equality under-counts real agreement because the
#: HR-mapped skeleton is a nearest-voxel rasterization of a different grid,
#: not an interpolation, so real shared vessels are typically off by a voxel
#: or two even when both routes see the same structure.
DEFAULT_DILATION_RADIUS_VOXELS = 2

#: Connected-component size floor for a UFAST-only candidate to count as
#: vessel-like rather than noise. Same threshold analysis.segment_ufast_vessels
#: uses for its own noise-vs-real connectivity stats.
VESSEL_LIKE_MIN_VOXELS = 10

#: Component-mean peak-SNR bar for the kinetics-adjacency additive filter.
#: Deliberately looser than a strict per-voxel bolus-arrival bar since this is
#: a component MEAN and a tighter bar under-recalls components the size
#: filter already trusts.
DEFAULT_KINETICS_MIN_PEAK_SNR = 2.0

#: Voxel-radius a UFAST-only component must be within of already-trusted
#: vasculature (HR skeleton or a size-qualifying UFAST component) to be
#: accepted via the kinetics path.
DEFAULT_KINETICS_ADJACENCY_RADIUS_VOXELS = 6

#: Guards the peak-SNR denominator: voxels with near-zero baseline variance
#: (e.g. air/background) are EXCLUDED from scoring, not given a floored
#: denominator -- flooring produces spurious SNR values orders of magnitude
#: too large (division by an artificial ~0 std).
_KINETICS_STD_EPS = 1e-8

#: Provenance codes for the merged skeleton's per-voxel origin.
LABEL_BACKGROUND = 0
LABEL_HR_ONLY = 1
LABEL_CONFIRMED_BOTH = 2
LABEL_UFAST_ONLY_ADDED = 3
LABEL_REPAIR_BRIDGE_ADDED = 4
LABEL_KINETICS_ADJACENCY_ADDED = 5


def find_ufast_phases(preprocessing_root: Path, exam_id: str) -> list[Path]:
    """Return the exam's motion-corrected UFAST phase files, in phase order."""
    dce_dir = preprocessing_root / "dce" / exam_id
    phases = sorted(dce_dir.glob(f"{exam_id}_*.nii.gz"))
    if not phases:
        raise FileNotFoundError(f"no UFAST phases found under {dce_dir}")
    return phases


def classify_overlap(
    hr_mask: np.ndarray, ufast_mask: np.ndarray, *, dilation_radius_voxels: int
) -> dict[str, np.ndarray]:
    """Split UFAST-direct voxels into "confirmed by HR proximity" vs "UFAST-only".

    Asymmetric by design for the *merge geometry*: HR is dilated and tested
    against UFAST, not the other way around, since HR is the authoritative
    source -- every HR voxel is kept regardless of what UFAST does, so a
    proximity-confirmed UFAST voxel does not contribute a new merged voxel.
    Provenance labeling is separate: ``confirmed_both`` marks HR voxels near
    the UFAST mask (see ``merge_skeletons``), which is the documented
    two-voxel proximity agreement on the coordinates that survive the merge.
    """
    structure = ndi.generate_binary_structure(3, 3)
    hr_dilated = ndi.binary_dilation(
        hr_mask, structure=structure, iterations=dilation_radius_voxels
    )
    confirmed_ufast = ufast_mask & hr_dilated
    ufast_only_candidate = ufast_mask & ~hr_dilated
    return {
        "confirmed_ufast_mask": confirmed_ufast,
        "ufast_only_candidate_mask": ufast_only_candidate,
    }


def filter_vessel_like_components(
    mask: np.ndarray, *, min_voxels: int = VESSEL_LIKE_MIN_VOXELS
) -> tuple[np.ndarray, dict[str, int]]:
    """Keep only connected components (26-connectivity) at least ``min_voxels`` large."""
    if not np.any(mask):
        return mask, {
            "n_components_total": 0,
            "n_components_kept": 0,
            "voxels_dropped_as_noise": 0,
        }
    structure = ndi.generate_binary_structure(3, 3)
    labeled, n_components = ndi.label(mask, structure=structure)
    sizes = np.bincount(labeled.ravel())
    keep_ids = np.flatnonzero(sizes >= min_voxels)
    keep_ids = keep_ids[keep_ids != 0]
    kept = np.isin(labeled, keep_ids)
    stats = {
        "n_components_total": int(n_components),
        "n_components_kept": int(len(keep_ids)),
        "voxels_dropped_as_noise": int(mask.sum() - kept.sum()),
    }
    return kept, stats


def compute_ufast_peak_snr(preprocessing_root: Path, exam_id: str) -> np.ndarray:
    """Per-voxel contrast-enhancement peak-SNR from the raw motion-corrected UFAST signal.

    ``(peak post-baseline signal - baseline mean) / baseline std``. Voxels
    with near-zero baseline variance (``_KINETICS_STD_EPS``) are left at 0.0
    rather than given a floored denominator.
    """
    phases = find_ufast_phases(preprocessing_root, exam_id)
    signal_tzyx = np.stack(
        [sitk.GetArrayFromImage(sitk.ReadImage(str(p))) for p in phases], axis=0
    ).astype(np.float32)
    provenance = json.loads(
        (
            preprocessing_root / "work" / exam_id / "preprocessing_provenance.json"
        ).read_text()
    )
    baseline_frame_count = int(provenance["ufast_source"]["baseline_frame_count"])
    baseline = signal_tzyx[:baseline_frame_count]
    baseline_mean = baseline.mean(axis=0)
    baseline_std = baseline.std(axis=0)
    peak = signal_tzyx[baseline_frame_count:].max(axis=0)

    peak_snr = np.zeros_like(baseline_mean)
    ok_std = baseline_std > _KINETICS_STD_EPS
    peak_snr[ok_std] = (peak[ok_std] - baseline_mean[ok_std]) / baseline_std[ok_std]
    return peak_snr


def kinetics_adjacency_candidates(
    preprocessing_root: Path,
    exam_id: str,
    *,
    candidate_mask: np.ndarray,
    already_added_mask: np.ndarray,
    trusted_mask: np.ndarray,
    min_peak_snr: float = DEFAULT_KINETICS_MIN_PEAK_SNR,
    adjacency_radius_voxels: int = DEFAULT_KINETICS_ADJACENCY_RADIUS_VOXELS,
) -> tuple[np.ndarray, dict[str, int]]:
    """Second, additive chance for UFAST-only candidates the size filter rejected.

    A candidate connected component is accepted if BOTH its mean per-voxel
    peak-SNR clears ``min_peak_snr`` AND it is spatially continuous with
    ``trusted_mask`` within ``adjacency_radius_voxels``. Only considers
    components NOT already in ``already_added_mask``, so this never
    duplicates or overrides the size filter's own result -- it only adds.
    """
    remaining_candidate = candidate_mask & ~already_added_mask
    if not np.any(remaining_candidate):
        return (
            np.zeros_like(candidate_mask),
            {
                "n_components_considered": 0,
                "n_components_accepted": 0,
                "voxels_added": 0,
            },
        )

    structure = ndi.generate_binary_structure(3, 3)
    labeled, n_components = ndi.label(remaining_candidate, structure=structure)
    if n_components == 0:
        return (
            np.zeros_like(candidate_mask),
            {
                "n_components_considered": 0,
                "n_components_accepted": 0,
                "voxels_added": 0,
            },
        )

    peak_snr_zyx = compute_ufast_peak_snr(preprocessing_root, exam_id)
    sizes = np.bincount(labeled.ravel())[1 : n_components + 1]
    snr_sums = ndi.sum(peak_snr_zyx, labeled, index=np.arange(1, n_components + 1))
    comp_mean_snr = snr_sums / np.maximum(sizes, 1)
    snr_pass_ids = set((np.flatnonzero(comp_mean_snr >= min_peak_snr) + 1).tolist())

    trusted_dilated = ndi.binary_dilation(
        trusted_mask, structure=structure, iterations=adjacency_radius_voxels
    )
    adjacent_ids = {
        component_id
        for component_id in range(1, n_components + 1)
        if np.any((labeled == component_id) & trusted_dilated)
    }

    accepted_ids = snr_pass_ids & adjacent_ids
    accepted_mask = (
        np.isin(labeled, list(accepted_ids))
        if accepted_ids
        else np.zeros_like(candidate_mask)
    )
    stats = {
        "n_components_considered": int(n_components),
        "n_components_accepted": int(len(accepted_ids)),
        "voxels_added": int(accepted_mask.sum()),
    }
    return accepted_mask, stats


def merge_skeletons(
    *,
    preprocessing_root: Path,
    exam_id: str,
    hr_skeleton: np.ndarray,
    hr_support: np.ndarray,
    ufast_skeleton: np.ndarray,
    ufast_support: np.ndarray,
    dilation_radius_voxels: int = DEFAULT_DILATION_RADIUS_VOXELS,
    vessel_like_min_voxels: int = VESSEL_LIKE_MIN_VOXELS,
    enable_kinetics_filter: bool = True,
    kinetics_min_peak_snr: float = DEFAULT_KINETICS_MIN_PEAK_SNR,
    kinetics_adjacency_radius_voxels: int = DEFAULT_KINETICS_ADJACENCY_RADIUS_VOXELS,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Merge one exam's HR-mapped and UFAST-direct skeletons (both same grid).

    Returns ``(merged_skeleton, merged_support, provenance)``, all ``(z,y,x)``
    except ``provenance``, which is a JSON-serializable summary dict.
    """
    overlap = classify_overlap(
        hr_skeleton, ufast_skeleton, dilation_radius_voxels=dilation_radius_voxels
    )
    ufast_only_candidate = overlap["ufast_only_candidate_mask"]
    ufast_size_added_mask, filter_stats = filter_vessel_like_components(
        ufast_only_candidate, min_voxels=vessel_like_min_voxels
    )

    if enable_kinetics_filter:
        kinetics_added_mask, kinetics_stats = kinetics_adjacency_candidates(
            preprocessing_root,
            exam_id,
            candidate_mask=ufast_only_candidate,
            already_added_mask=ufast_size_added_mask,
            trusted_mask=hr_skeleton | ufast_size_added_mask,
            min_peak_snr=kinetics_min_peak_snr,
            adjacency_radius_voxels=kinetics_adjacency_radius_voxels,
        )
    else:
        kinetics_added_mask = np.zeros_like(ufast_only_candidate)
        kinetics_stats = {
            "n_components_considered": 0,
            "n_components_accepted": 0,
            "voxels_added": 0,
        }

    ufast_added_mask = ufast_size_added_mask | kinetics_added_mask
    candidate_merged = hr_skeleton | ufast_added_mask

    # Reconstruct a pseudo temporal-support count: neither route persists the
    # raw integer count TC4D's cleanup stage was designed around, only a
    # thresholded boolean. Summing the two boolean supports gives values in
    # {0,1,2} -- voxels backed by neither/one/both routes -- which preserves
    # the intended ordinal "more agreement -> more supported" signal that the
    # reused cleanup's component scoring depends on.
    support_count_zyx = hr_support.astype(np.int32) + ufast_support.astype(np.int32)
    cleaned_candidate = _finalize_exam_mask_topology(
        candidate_merged, support_count_zyx=support_count_zyx
    )

    # _finalize_exam_mask_topology is TC4D's own from-scratch skeleton
    # derivation and has no mechanism to preserve voxels already in
    # hr_skeleton -- HR is authoritative (see module docstring), so re-union
    # it back in unconditionally: cleanup only gets to decide what UFAST-only/
    # repair-bridge additions survive on top, never whether an HR voxel stays.
    hr_voxels_rescued = int((hr_skeleton & ~cleaned_candidate).sum())
    # kinetics_added_mask voxels hit the same problem: cleanup's micro-
    # component pruning floor scales with total exam size, but a kinetics-
    # accepted component is often only a few voxels -- smaller than that
    # floor. Since peak-SNR + spatial continuity is what actually validates
    # these voxels, force-reunion them too. ufast_size_added_mask is NOT
    # force-kept -- those voxels only cleared a bare size filter and are
    # meant to still be subject to cleanup's own topology/support judgment.
    kinetics_voxels_rescued = int((kinetics_added_mask & ~cleaned_candidate).sum())
    merged_skeleton = hr_skeleton | cleaned_candidate | kinetics_added_mask
    merged_support = hr_support | ufast_support | merged_skeleton

    # Provenance only: proximity-confirmed UFAST voxels are left out of the
    # merged mask in favor of the authoritative HR coordinates, so intersecting
    # merged_skeleton with confirmed_ufast_mask would count mostly exact
    # overlaps. Label the HR voxels near the UFAST mask instead -- that is the
    # documented two-voxel proximity agreement, without changing the merge.
    structure = ndi.generate_binary_structure(3, 3)
    ufast_dilated = ndi.binary_dilation(
        ufast_skeleton.astype(bool),
        structure=structure,
        iterations=dilation_radius_voxels,
    )
    confirmed_final = merged_skeleton & hr_skeleton & ufast_dilated
    hr_only_final = merged_skeleton & hr_skeleton & ~confirmed_final
    ufast_size_added_final = merged_skeleton & ufast_size_added_mask & ~hr_skeleton
    kinetics_added_final = (
        merged_skeleton & kinetics_added_mask & ~hr_skeleton & ~ufast_size_added_mask
    )
    repair_bridge_final = merged_skeleton & ~(hr_skeleton | ufast_skeleton)

    provenance_label = np.zeros(merged_skeleton.shape, dtype=np.uint8)
    provenance_label[hr_only_final] = LABEL_HR_ONLY
    provenance_label[confirmed_final] = LABEL_CONFIRMED_BOTH
    provenance_label[ufast_size_added_final] = LABEL_UFAST_ONLY_ADDED
    provenance_label[kinetics_added_final] = LABEL_KINETICS_ADJACENCY_ADDED
    provenance_label[repair_bridge_final] = LABEL_REPAIR_BRIDGE_ADDED

    provenance = {
        "exam_id": exam_id,
        "route": "hr_ufast_skeleton_merge (HR-authoritative + filtered UFAST additions)",
        "dilation_radius_voxels": dilation_radius_voxels,
        "vessel_like_min_voxels": vessel_like_min_voxels,
        "support_count_reconstruction": "hr_support.astype(int) + ufast_support.astype(int) -> {0,1,2}",
        "hr_skeleton_voxels": int(hr_skeleton.sum()),
        "ufast_skeleton_voxels": int(ufast_skeleton.sum()),
        "confirmed_ufast_voxels": int(overlap["confirmed_ufast_mask"].sum()),
        "ufast_only_candidate_voxels": int(ufast_only_candidate.sum()),
        "ufast_size_added_voxels_pre_cleanup": int(ufast_size_added_mask.sum()),
        "filter_stats": filter_stats,
        "kinetics_filter": {
            "enabled": enable_kinetics_filter,
            "min_peak_snr": kinetics_min_peak_snr,
            "adjacency_radius_voxels": kinetics_adjacency_radius_voxels,
            "voxels_rescued_from_cleanup": kinetics_voxels_rescued,
            **kinetics_stats,
        },
        "ufast_added_voxels_pre_cleanup": int(ufast_added_mask.sum()),
        "merged_voxels_pre_cleanup": int(candidate_merged.sum()),
        "hr_voxels_rescued_from_cleanup": hr_voxels_rescued,
        "merged_voxels_final": int(merged_skeleton.sum()),
        "label_voxel_counts": {
            "hr_only": int((provenance_label == LABEL_HR_ONLY).sum()),
            "confirmed_both": int((provenance_label == LABEL_CONFIRMED_BOTH).sum()),
            "ufast_only_added": int((provenance_label == LABEL_UFAST_ONLY_ADDED).sum()),
            "kinetics_adjacency_added": int(
                (provenance_label == LABEL_KINETICS_ADJACENCY_ADDED).sum()
            ),
            "repair_bridge_added": int(
                (provenance_label == LABEL_REPAIR_BRIDGE_ADDED).sum()
            ),
        },
        "provenance_label_zyx_codes": {
            "background": LABEL_BACKGROUND,
            "hr_only": LABEL_HR_ONLY,
            "confirmed_both": LABEL_CONFIRMED_BOTH,
            "ufast_only_added": LABEL_UFAST_ONLY_ADDED,
            "repair_bridge_added": LABEL_REPAIR_BRIDGE_ADDED,
            "kinetics_adjacency_added": LABEL_KINETICS_ADJACENCY_ADDED,
        },
    }
    return merged_skeleton, merged_support, provenance, provenance_label
