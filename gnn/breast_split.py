"""Split a bilateral DCE-MRI skeleton down to its tumor-bearing breast.

Generalizes the inner-wall symmetry-line method from
``tabular/duke_final_symmetry_lines.py`` (Aakrithi's DUKE splitter,
``git log`` credits ``aakrithiram``, 2026-07-09) into a reusable, dataset-
agnostic function. That script computes the midline and infers tumor side
but only ever writes QC images and summary counts -- it never applies the
side mask to the skeleton and saves a filtered array. This module adds that
missing step: :func:`split_case` returns a boolean skeleton restricted to
the tumor-bearing breast, in the same coordinate system as the input (no
recrop/recenter), plus an explicit include/exclude decision.

The geometric primitives (`_inner_wall_midline_from_projection`,
`_project_to_yx`, `_axis_values_for_mask`) are imported directly from
``tabular.duke_final_symmetry_lines`` rather than reimplemented -- they are
pure array functions with no DUKE-specific coupling.

Exclusion policy (see ``AUDITING_RESULTS.md`` for the full writeup): a case
is excluded, not silently included with a best-guess side, whenever the
tumor-bearing side can't be determined with reasonable confidence. This
covers three real failure modes: no tumor mask (unknown tumor side), a
tumor mask that barely overlaps the breast mask at all (misregistration),
and a tumor that meaningfully overlaps both sides of the midline (genuinely
bilateral disease or an ambiguous/near-midline tumor). One case is never
split into two independently labeled samples -- only the retained side's
skeleton is produced.

Precondition: only call :func:`split_case` on cases already known to be
bilateral (the case-level ``bilateral`` clinical flag). A native unilateral
image has breast tissue on only one side, so the inner-wall method -- which
needs a wall on *both* sides of the midline to pair against -- correctly
fails to find one and excludes the case (see
``tests/test_gnn_breast_split.py::test_native_unilateral_input_cannot_find_a_midline``).
Per the harmonized-cohort plan, native unilateral cases must be routed
around this module entirely and used unchanged, not passed through it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tabular.duke_final_symmetry_lines import (
    _axis_values_for_mask,
    _detect_coord_to_array_axis,
    _inner_wall_midline_from_projection,
    _iter_segment_entries,
    _preferred_coord_to_array_axis,
    _project_to_yx,
    _read_json,
)

#: Root of the frozen-fold centerline pipeline's per-case output directories
#: (``<root>/<dataset>/<case_id>/``), shared with the skeleton/morphometry
#: files this module reads.
CENTERLINE_STUDIES_ROOT = Path(
    "/gpfs/data/karczmar-lab/workspaces/saritbose/centerlines_tc4d/studies"
)


def resolve_study_dir(
    dataset: str, case_id: str, *, root: Path = CENTERLINE_STUDIES_ROOT
) -> Path:
    """Return a case's per-case output directory under the centerline tree."""
    return root / dataset / case_id


def detect_case_coord_axes(
    case_id: str, study_dir: Path, array_shape: tuple[int, int, int]
) -> tuple[int, int, int]:
    """Detect a case's ``(z, y, x)`` coordinate-to-array-axis mapping.

    Assuming a fixed axis order (e.g. always ``(0, 1, 2)``) is wrong for at
    least some real cases -- confirmed empirically (`LAB_NOTEBOOK.md`,
    2026-07-14) -- so this reuses
    ``duke_final_symmetry_lines._detect_coord_to_array_axis``, which infers
    the mapping per-case from morphometry segment coordinates, the same way
    that script resolves it for its own tumor-side inference.

    Raises if no valid mapping can be found -- there is no reasonable
    fallback (any wrong guess would silently corrupt the split).
    """
    summary = _read_json(study_dir / "run_summary.json")
    tumor_context = summary.get("tumor_context", {})
    tumor_layout = (
        str(tumor_context.get("layout", "") or "")
        if isinstance(tumor_context, dict)
        else ""
    )
    morphometry = _read_json(study_dir / f"{case_id}_morphometry.json")
    segments = _iter_segment_entries(morphometry)
    preferred = _preferred_coord_to_array_axis(tumor_layout)
    coord_to_axis, assumption = _detect_coord_to_array_axis(
        segments=segments, array_shape=array_shape, preferred=preferred
    )
    if coord_to_axis is None:
        raise ValueError(
            f"case={case_id}: could not detect coordinate-axis mapping ({assumption})"
        )
    return coord_to_axis


#: Shared breast-mask root confirmed (2026-07-14, per Aakrithi) to cover all
#: four MAMA-MIA datasets under identical per-case subdirectories -- unlike
#: ``duke_final_symmetry_lines.py``'s ``DEFAULT_DUKE_BREAST_ROOTS``, this is
#: not DUKE-specific: ``<root>/<dataset>/<case_id>/images/`` holds
#: ``<case_id>_0001_breast_mask.npy`` (or another timepoint suffix) for
#: DUKE, ISPY1, ISPY2, and NACT alike (verified by directory listing against
#: real data, not assumed from the DUKE convention alone).
BREAST_MASK_ROOT = Path("/gpfs/data/karczmar-lab/workspaces/saritbose/breast_masks")


def resolve_breast_mask_path(
    dataset: str, case_id: str, *, root: Path = BREAST_MASK_ROOT
) -> Path | None:
    """Find a case's breast-mask ``.npy`` file under the shared mask root.

    Mirrors ``duke_final_symmetry_lines.py::_resolve_breast_mask_path``'s
    preferred-timepoint-then-glob logic, generalized from DUKE-only to any
    dataset present under ``root`` (same directory convention for all four).

    Returns ``None`` if no mask file is found -- callers treat that as
    ``breast_mask_missing``, not an error, since a case can genuinely lack a
    breast mask.
    """
    images_dir = root / dataset / case_id / "images"
    preferred = images_dir / f"{case_id}_0001_breast_mask.npy"
    if preferred.exists():
        return preferred
    candidates = sorted(images_dir.glob(f"{case_id}_*_breast_mask.npy"))
    return candidates[0] if candidates else None


def resolve_tumor_mask_orientation(
    tumor_mask: np.ndarray, *, breast_mask: np.ndarray, flip_axis: int = 0
) -> tuple[np.ndarray, bool]:
    """Disambiguate a tumor mask's axis-0 direction against the breast mask.

    ``tabular.duke_final_symmetry_lines._load_mask_matching_shape`` only
    searches axis *permutations* to match the expected array shape, never
    axis *flips* -- so it can't distinguish two orientations that both
    happen to produce the right shape. In the real MAMA-MIA cohort, the
    tumor mask NIfTI files (a different export pipeline than the skeleton
    and breast-mask ``.npy`` arrays) exhibit exactly this ambiguity on
    array axis 0 for a meaningful fraction of cases -- confirmed empirically
    (2026-07-14, `LAB_NOTEBOOK.md`) on 11 real cases spanning DUKE and
    ISPY2: flipping axis 0 took several cases from ~0% tumor-in-breast
    overlap to ~100%, while leaving already-good cases mostly unchanged (one
    case got very slightly worse under the flip, which is why this compares
    rather than always flipping).

    A real tumor is, by definition, inside breast tissue -- so "which
    orientation puts the tumor inside the breast mask" is a physically
    grounded disambiguator, not an arbitrary tie-break. Ambiguity is never
    silent: the caller receives which orientation won and can log/audit it.

    Returns:
        ``(resolved_mask, was_flipped)``.
    """
    unflipped_overlap = np.count_nonzero(tumor_mask & breast_mask)
    flipped = np.flip(tumor_mask, axis=flip_axis)
    flipped_overlap = np.count_nonzero(flipped & breast_mask)
    if flipped_overlap > unflipped_overlap:
        return flipped, True
    return tumor_mask, False


#: Minimum fraction of projected-mask rows that must yield a clean inner-wall
#: pair for the midline to be trusted, mirroring
#: ``duke_final_symmetry_lines.py``'s ``--x-support-fraction`` default.
MIN_INNER_WALL_ROW_FRACTION = 0.05

#: A case is excluded unless at least this fraction of tumor voxels that
#: overlap either breast mask fall on one side of the midline. Below this,
#: the tumor meaningfully straddles both sides (near-midline tumor or
#: genuinely bilateral disease) and side assignment isn't trustworthy.
MIN_TUMOR_SIDE_VOXEL_FRACTION = 0.9

#: A case is excluded if fewer than this fraction of the tumor mask's raw
#: voxels land inside either breast mask at all -- protects against a
#: breast/tumor mask pair that isn't actually co-registered.
MIN_TUMOR_IN_BREAST_FRACTION = 0.5


@dataclass(frozen=True)
class BreastSplitResult:
    """Outcome of attempting to split one case to its tumor-bearing breast."""

    case_id: str
    included: bool
    exclude_reason: str
    retained_side: str | None
    midline_x: float | None
    tumor_side_voxel_fraction: float | None
    left_breast_voxels: int | None
    right_breast_voxels: int | None
    skeleton_single_breast: np.ndarray | None
    #: Split the same way as ``skeleton_single_breast``, ``None`` unless a
    #: ``support_mask`` was passed to :func:`split_case`. The GNN graph
    #: builder (``gnn/data_loader.py::_build_case``) reads the support mask
    #: from the same directory as the skeleton it's given, so a substituted
    #: single-breast skeleton needs a matching substituted support mask
    #: alongside it -- see ``save_split_support_mask``.
    support_single_breast: np.ndarray | None


def _side_breast_masks(
    breast_mask: np.ndarray, *, x_axis: int, midline_x: float
) -> tuple[np.ndarray, np.ndarray]:
    axis_values = _axis_values_for_mask(breast_mask.shape, x_axis)
    left = breast_mask & (axis_values < midline_x)
    right = breast_mask & (axis_values >= midline_x)
    return left, right


def split_case(
    case_id: str,
    *,
    skeleton: np.ndarray,
    breast_mask: np.ndarray,
    tumor_mask: np.ndarray,
    support_mask: np.ndarray | None = None,
    x_axis: int = 2,
    y_axis: int = 1,
    z_axis: int = 0,
    min_inner_wall_row_fraction: float = MIN_INNER_WALL_ROW_FRACTION,
    min_tumor_side_voxel_fraction: float = MIN_TUMOR_SIDE_VOXEL_FRACTION,
    min_tumor_in_breast_fraction: float = MIN_TUMOR_IN_BREAST_FRACTION,
) -> BreastSplitResult:
    """Restrict ``skeleton`` to its tumor-bearing breast, or exclude the case.

    ``skeleton``, ``breast_mask``, and ``tumor_mask`` must be boolean arrays
    of identical shape, in the same coordinate system the skeleton was built
    in (the retained-side skeleton is returned in that same coordinate
    system, unmodified). ``support_mask``, if given, must match that shape
    too and is split the same way -- the GNN graph builder needs both the
    skeleton and its support mask from the same directory (see
    ``BreastSplitResult.support_single_breast``).

    Args:
        case_id: Case identifier, carried through into the result for logging.
        skeleton: Boolean skeleton mask to split.
        breast_mask: Boolean whole-breast mask (both breasts).
        tumor_mask: Boolean tumor mask.
        support_mask: Optional boolean support mask (the neighborhood around
            the skeleton used for radius/distance-transform features) to
            split the same way as ``skeleton``.
        x_axis: Which array axis is left-right -- the axis the midline
            splits on. Default matches the DUKE convention in
            ``duke_final_symmetry_lines.py``.
        y_axis: Which array axis is superior-inferior.
        z_axis: Which array axis is anterior-posterior.
        min_inner_wall_row_fraction: See module docstring.
        min_tumor_side_voxel_fraction: See module docstring.
        min_tumor_in_breast_fraction: See module docstring.

    Returns:
        A :class:`BreastSplitResult`. ``skeleton_single_breast`` (and
        ``support_single_breast``, if ``support_mask`` was given) are
        ``None`` whenever ``included`` is ``False``.
    """

    def _excluded(reason: str) -> BreastSplitResult:
        return BreastSplitResult(
            case_id=case_id,
            included=False,
            exclude_reason=reason,
            retained_side=None,
            midline_x=None,
            tumor_side_voxel_fraction=None,
            left_breast_voxels=None,
            right_breast_voxels=None,
            skeleton_single_breast=None,
            support_single_breast=None,
        )

    if skeleton.shape != breast_mask.shape or skeleton.shape != tumor_mask.shape:
        raise ValueError(
            f"case={case_id}: shape mismatch skeleton={skeleton.shape} "
            f"breast_mask={breast_mask.shape} tumor_mask={tumor_mask.shape}"
        )
    if support_mask is not None and support_mask.shape != skeleton.shape:
        raise ValueError(
            f"case={case_id}: shape mismatch support_mask={support_mask.shape} "
            f"skeleton={skeleton.shape}"
        )

    breast_mask = np.asarray(breast_mask, dtype=bool)
    tumor_mask = np.asarray(tumor_mask, dtype=bool)
    skeleton = np.asarray(skeleton, dtype=bool)
    if support_mask is not None:
        support_mask = np.asarray(support_mask, dtype=bool)

    if not np.any(breast_mask):
        return _excluded("breast_mask_empty")
    if not np.any(tumor_mask):
        return _excluded("tumor_mask_empty")

    coord_to_axis = (z_axis, y_axis, x_axis)
    breast_yx = _project_to_yx(breast_mask, coord_to_axis=coord_to_axis)
    image_center_x = float(breast_mask.shape[x_axis]) / 2.0
    midline_x, failure_reason = _inner_wall_midline_from_projection(
        breast_yx,
        image_midline_x=image_center_x,
        min_pair_fraction=min_inner_wall_row_fraction,
    )
    if midline_x is None:
        return _excluded(f"midline_detection_failed:{failure_reason}")

    left_breast, right_breast = _side_breast_masks(
        breast_mask, x_axis=x_axis, midline_x=midline_x
    )

    tumor_in_breast = tumor_mask & (left_breast | right_breast)
    raw_tumor_voxels = int(np.count_nonzero(tumor_mask))
    in_breast_voxels = int(np.count_nonzero(tumor_in_breast))
    if in_breast_voxels < min_tumor_in_breast_fraction * raw_tumor_voxels:
        return _excluded("tumor_mostly_outside_breast_mask")

    tumor_left = int(np.count_nonzero(tumor_mask & left_breast))
    tumor_right = int(np.count_nonzero(tumor_mask & right_breast))
    total = tumor_left + tumor_right
    side_fraction = max(tumor_left, tumor_right) / total
    if side_fraction < min_tumor_side_voxel_fraction:
        return _excluded("tumor_overlap_both_sides")

    retained_side = "left" if tumor_left >= tumor_right else "right"
    retained_breast_mask = left_breast if retained_side == "left" else right_breast

    return BreastSplitResult(
        case_id=case_id,
        included=True,
        exclude_reason="",
        retained_side=retained_side,
        midline_x=midline_x,
        tumor_side_voxel_fraction=side_fraction,
        left_breast_voxels=int(np.count_nonzero(left_breast)),
        right_breast_voxels=int(np.count_nonzero(right_breast)),
        skeleton_single_breast=skeleton & retained_breast_mask,
        support_single_breast=(
            support_mask & retained_breast_mask if support_mask is not None else None
        ),
    )


#: Filename suffix for a saved single-breast skeleton, appended to
#: ``case_id`` -- shared with ``gnn/data_loader.py``'s
#: ``breast_split_mode="single"`` resolution so the writer
#: (``save_split_skeleton``, ``gnn.build_single_breast_skeletons``) and the
#: reader agree on one convention.
SINGLE_BREAST_SKELETON_SUFFIX = "_skeleton_4d_single_breast_mask.npy"


def single_breast_skeleton_path(root: Path, dataset: str, case_id: str) -> Path:
    """Return the expected single-breast skeleton path for one case.

    Mirrors the layout ``gnn.build_single_breast_skeletons`` writes to:
    ``<root>/<dataset>/<case_id>/<case_id><SINGLE_BREAST_SKELETON_SUFFIX>``.
    """
    return root / dataset / case_id / f"{case_id}{SINGLE_BREAST_SKELETON_SUFFIX}"


def save_split_skeleton(result: BreastSplitResult, out_dir: Path) -> Path:
    """Save ``result.skeleton_single_breast`` next to the original skeleton naming.

    Raises if ``result`` was excluded -- callers must check ``included``
    first; there is no filtered skeleton to save for an excluded case.
    """
    if not result.included or result.skeleton_single_breast is None:
        raise ValueError(
            f"case={result.case_id}: cannot save, excluded ({result.exclude_reason})"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{result.case_id}{SINGLE_BREAST_SKELETON_SUFFIX}"
    np.save(out_path, result.skeleton_single_breast)
    return out_path


#: Support-mask filename pattern -- deliberately identical to the *original*
#: exam-level support mask's own name (not a "single_breast"-suffixed
#: variant), because ``gnn/data_loader.py::_build_case`` looks up the
#: support mask via a fixed pattern relative to the skeleton path's parent
#: directory, unaware of breast-split substitution. Must match
#: ``config.DEFAULT_CONFIG["feature_toggles"]["deepsets_support_mask_pattern"]``
#: (``gnn/data_loader.py``'s ``_SUPPORT_PATTERN``).
SUPPORT_MASK_FILENAME_PATTERN = "{case_id}_skeleton_4d_exam_support_mask.npy"


def save_split_support_mask(result: BreastSplitResult, out_dir: Path) -> Path:
    """Save ``result.support_single_breast`` using the original support-mask filename.

    Raises if ``result`` was excluded, or if ``split_case`` wasn't given a
    ``support_mask`` to split in the first place -- callers must check both
    before calling this.
    """
    if not result.included or result.support_single_breast is None:
        raise ValueError(
            f"case={result.case_id}: cannot save, excluded ({result.exclude_reason}) "
            "or no support_mask was passed to split_case"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / SUPPORT_MASK_FILENAME_PATTERN.format(case_id=result.case_id)
    np.save(out_path, result.support_single_breast)
    return out_path
