"""Precompute single-breast skeletons for every bilateral case in the cohort.

Phase 4 of the harmonized single-breast dataset plan: for every case flagged
``bilateral`` in the clinical metadata, runs ``gnn.breast_split.split_case``
(with ``resolve_tumor_mask_orientation`` applied first -- see
``AUDITING_RESULTS.md``) and, if included, saves the resulting single-breast
skeleton and its matching single-breast support mask, plus a copy of
``run_summary.json``, under ``--out-root`` in the same
``<dataset>/<case_id>/`` layout as the source centerline tree. All three are
required: ``gnn/data_loader.py::_build_case`` resolves the support mask and
``study_timepoints`` (from ``run_summary.json``) relative to whichever
skeleton path it's given, unaware of breast-split substitution --
``run_summary.json`` is copied unmodified since study timing doesn't depend
on which breast is retained. Writes a per-case manifest CSV
(``breast_split_manifest.csv``) that is the authoritative record of which
bilateral cases are in the harmonized single-breast cohort and why any
others were excluded -- ``gnn/data_loader.py``'s
``breast_split_mode="single"`` consumes this manifest's presence (a saved
skeleton file) directly, not a re-run of the splitter.

Writes to ``--out-root`` (a workspace path), never back into the shared
``centerlines_tc4d/studies`` tree, so the source data stays read-only.

Usage::

    python -m gnn.build_single_breast_skeletons \
        --out-root /gpfs/data/karczmar-lab/workspaces/spencervenancio/breast_split_skeletons
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from clinical_features import load_clinical_from_patient_info
from gnn.breast_split import (
    SUPPORT_MASK_FILENAME_PATTERN,
    detect_case_coord_axes,
    resolve_breast_mask_path,
    resolve_study_dir,
    resolve_tumor_mask_orientation,
    save_split_skeleton,
    save_split_support_mask,
    split_case,
)
from tabular.duke_final_symmetry_lines import _load_array, _load_mask_matching_shape

PATIENT_INFO_DIR = Path(
    "/gpfs/data/karczmar-lab/MAMA-MIA-syn60868042/patient_info_files"
)
TUMOR_MASK_DIR = Path(
    "/gpfs/data/karczmar-lab/MAMA-MIA-syn60868042/segmentations/expert"
)

_MANIFEST_NAME = "breast_split_manifest.csv"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--cases",
        type=str,
        default=None,
        help="Optional comma-separated case-ID whitelist (restricts to bilateral "
        "cases already in this list); omit to process every bilateral case.",
    )
    return parser.parse_args()


def discover_bilateral_cases(cases: list[str] | None = None) -> pd.DataFrame:
    """Return ``(case_id, dataset)`` for every case flagged bilateral in clinical metadata."""
    clinical = load_clinical_from_patient_info(PATIENT_INFO_DIR)
    bilateral = clinical[clinical["bilateral"] == True]  # noqa: E712
    if cases is not None:
        bilateral = bilateral[bilateral["case_id"].isin(cases)]
    return (
        bilateral[["case_id", "dataset"]].sort_values("case_id").reset_index(drop=True)
    )


def process_one_case(
    case_id: str, dataset: str, *, out_root: Path
) -> dict[str, object]:
    """Run the splitter on one bilateral case; returns a manifest row.

    Raises if the skeleton file itself is missing -- a bilateral case with
    no skeleton at all is a build-time problem upstream of this script, not
    something this script should paper over. Missing breast/tumor masks are
    real, expected data gaps (e.g. ISPY1's bilateral subset, 2026-07-14) and
    are recorded as exclusions, not raised.
    """
    study_dir = resolve_study_dir(dataset, case_id)
    skeleton_path = study_dir / f"{case_id}_skeleton_4d_exam_mask.npy"
    if not skeleton_path.exists():
        raise FileNotFoundError(f"case={case_id}: no skeleton at {skeleton_path}")
    skeleton = np.asarray(_load_array(skeleton_path), dtype=bool)

    support_mask_path = study_dir / SUPPORT_MASK_FILENAME_PATTERN.format(
        case_id=case_id
    )
    breast_mask_path = resolve_breast_mask_path(dataset, case_id)
    tumor_mask_path = TUMOR_MASK_DIR / f"{case_id}.nii.gz"
    if (
        breast_mask_path is None
        or not tumor_mask_path.exists()
        or not support_mask_path.exists()
    ):
        if breast_mask_path is None:
            reason = "breast_mask_missing"
        elif not tumor_mask_path.exists():
            reason = "tumor_mask_missing"
        else:
            reason = "support_mask_missing"
        return {
            "case_id": case_id,
            "dataset": dataset,
            "included": False,
            "exclude_reason": reason,
            "tumor_mask_was_flipped": None,
            "retained_side": None,
            "tumor_side_voxel_fraction": None,
            "skeleton_voxels_before": int(skeleton.sum()),
            "skeleton_voxels_after": None,
            "output_path": None,
            "support_output_path": None,
        }
    support_mask = np.asarray(_load_array(support_mask_path), dtype=bool)

    z_axis, y_axis, x_axis = detect_case_coord_axes(case_id, study_dir, skeleton.shape)
    breast_mask, _ = _load_mask_matching_shape(
        breast_mask_path, expected_shape=skeleton.shape, threshold=0.5
    )
    tumor_mask, _ = _load_mask_matching_shape(
        tumor_mask_path, expected_shape=skeleton.shape, threshold=0.5
    )
    tumor_mask, tumor_was_flipped = resolve_tumor_mask_orientation(
        tumor_mask, breast_mask=breast_mask
    )

    result = split_case(
        case_id,
        skeleton=skeleton,
        breast_mask=breast_mask,
        tumor_mask=tumor_mask,
        support_mask=support_mask,
        x_axis=x_axis,
        y_axis=y_axis,
        z_axis=z_axis,
    )

    output_path = None
    support_output_path = None
    if result.included:
        case_out_dir = out_root / dataset / case_id
        output_path = save_split_skeleton(result, case_out_dir)
        support_output_path = save_split_support_mask(result, case_out_dir)
        # gnn/data_loader.py::_build_case reads study_timepoints from
        # run_summary.json in the same directory as the skeleton it's given;
        # copied unmodified since acquisition timing doesn't depend on which
        # breast is retained.
        shutil.copy2(study_dir / "run_summary.json", case_out_dir / "run_summary.json")

    return {
        "case_id": case_id,
        "dataset": dataset,
        "included": result.included,
        "exclude_reason": result.exclude_reason,
        "tumor_mask_was_flipped": tumor_was_flipped,
        "retained_side": result.retained_side,
        "tumor_side_voxel_fraction": result.tumor_side_voxel_fraction,
        "skeleton_voxels_before": int(skeleton.sum()),
        "skeleton_voxels_after": int(result.skeleton_single_breast.sum())
        if result.included
        else None,
        "output_path": str(output_path) if output_path is not None else None,
        "support_output_path": str(support_output_path)
        if support_output_path is not None
        else None,
    }


def main() -> None:
    """Discover the full bilateral cohort, split each case, write the manifest."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = _parse_args()
    cases = args.cases.split(",") if args.cases else None

    cohort = discover_bilateral_cases(cases)
    logging.info("Processing %d bilateral case(s)", len(cohort))

    args.out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (case_id, dataset) in enumerate(zip(cohort["case_id"], cohort["dataset"])):
        if i % 25 == 0:
            logging.info("  %d/%d: %s (%s)", i, len(cohort), case_id, dataset)
        rows.append(process_one_case(case_id, dataset, out_root=args.out_root))

    manifest = pd.DataFrame(rows)
    manifest_path = args.out_root / _MANIFEST_NAME
    manifest.to_csv(manifest_path, index=False)

    n_included = int(manifest["included"].sum())
    logging.info("Done: %d/%d included", n_included, len(manifest))
    logging.info(
        manifest.groupby(["dataset", "exclude_reason"], dropna=False).size().to_string()
    )
    logging.info("Wrote %s", manifest_path)


if __name__ == "__main__":
    main()
