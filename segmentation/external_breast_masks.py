"""Inject external (ground-truth) whole-breast masks into the vessel-seg pipeline.

The segmentation pipeline (``batch_segmentation.py``) normally computes breast
masks itself (STEP-2, a U-Net) and feeds them as an extra input channel to the
vessel stage (STEP-3). This module lets STEP-3 instead consume the
externally-provided BreastDivider "whole-breast" masks shipped with the
UChicago manifest, so we can test whether real breast masks change the vessel
segmentation outcome.

Source data (see the manifest data dictionary):
  - manifest CSV ``..._with_whole_breast_masks.csv`` -- one row per exam_id,
    with ``whole_breast_mask_archive_path`` / ``whole_breast_mask_archive_member``
    naming the mask's NIfTI member inside a shared TAR.
  - ``whole_breast_masks.tar`` -- one NIfTI per exam at
    ``whole_breast_masks/<dataset>/<exam_id>.nii.gz``, labels
    ``0=background / 1=left / 2=right``. Use ``mask > 0`` for a whole-breast
    union.

Key facts (verified against real data, not just the dictionary):
  - The mask is stored in the SAME raw array space / affine as the exam's
    ``phase_0000.nii.gz`` (``exact_source_grid_match`` for the smoke-test
    patients). So applying the SAME geometric transform the adapter applies to
    images (``UChicagoDataset.preprocess``) lands the mask in STEP-1 space with
    no resize. We apply ONLY that geometric transform -- NOT
    ``normalize_image``/``zscore_image``, which are intensity ops meaningless
    for a 0/1 label map.
  - There is ONE mask per exam, generated from phase 0 only. This build did NOT
    include motion correction, so reusing the phase-0 mask for every dynamic
    phase of an exam is a documented approximation (it assumes the breast does
    not move across phases), not an oversight. Callers that fan the mask out to
    all phases must surface this limitation.
"""

from __future__ import annotations

import csv
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import SimpleITK as sitk

if TYPE_CHECKING:
    from cohorts.base import DatasetAdapter

DEFAULT_MANIFEST_CSV = (
    "/gpfs/data/karczmar-lab/vanguard/dce2d_internal_ultrafast_manifest/"
    "dce2d_internal_ultrafast_manifest_with_whole_breast_masks.csv"
)


def load_manifest(manifest_csv: str | Path) -> dict[str, dict[str, str]]:
    """Load the mask-augmented manifest, indexed by ``exam_id`` (181 small rows)."""
    manifest_csv = Path(manifest_csv)
    with manifest_csv.open(newline="") as f:
        rows = {row["exam_id"]: row for row in csv.DictReader(f)}
    return rows


def _resolve_mask_member(
    exam_id: str, manifest: dict[str, dict[str, str]]
) -> tuple[str, str]:
    """Return ``(archive_path, archive_member)`` for an exam, or raise clearly."""
    if exam_id not in manifest:
        raise KeyError(f"exam_id not in mask manifest: {exam_id}")
    row = manifest[exam_id]
    archive_path = row.get("whole_breast_mask_archive_path", "").strip()
    archive_member = row.get("whole_breast_mask_archive_member", "").strip()
    if not archive_path or not archive_member:
        raise ValueError(
            f"exam_id {exam_id} has no whole_breast_mask archive path/member in manifest"
        )
    return archive_path, archive_member


def validate_masks_available(
    case_ids: list[str], manifest: dict[str, dict[str, str]]
) -> None:
    """Fail-closed check: raise if ANY case lacks a resolvable mask member.

    Never silently skip a patient -- surface every missing case at once so the
    caller can bail before spending GPU time (matches this project's
    fail-closed data-join review standard).
    """
    missing = []
    for case_id in case_ids:
        try:
            _resolve_mask_member(case_id, manifest)
        except (KeyError, ValueError) as exc:
            missing.append(f"{case_id}: {exc}")
    if missing:
        raise ValueError(
            "External breast masks missing for {} of {} case(s):\n  {}".format(
                len(missing), len(case_ids), "\n  ".join(missing)
            )
        )


def build_whole_breast_mask(
    exam_id: str,
    manifest: dict[str, dict[str, str]],
    extract_dir: Path,
    adapter: DatasetAdapter,
    target_shape: tuple[int, int, int] | None = None,
) -> np.ndarray:
    """Extract + binarize + reorient one exam's whole-breast mask into STEP-1 space.

    Returns a ``float16`` array in the same geometry/shape as the exam's STEP-1
    preprocessed image. Applies the adapter's GEOMETRIC transform only (no
    intensity normalization). If ``target_shape`` is given and does not match
    after the transform, falls back to a ``torchio.Resize`` (the same convention
    ``predict_breast_batched`` uses) and warns -- not expected for the
    exact-grid-match smoke-test patients, but keeps the one documented
    repaired-geometry manifest case safe.
    """
    archive_path, archive_member = _resolve_mask_member(exam_id, manifest)

    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path) as tf:
        tf.extract(archive_member, path=extract_dir)
    mask_path = extract_dir / archive_member

    mask_raw = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))
    binary = (mask_raw > 0).astype(np.float32)
    reoriented = adapter.preprocess(binary)

    if target_shape is not None and reoriented.shape != tuple(target_shape):
        import torchio as tio

        print(
            f"  [warn] {exam_id}: mask shape {reoriented.shape} != target "
            f"{tuple(target_shape)}; resizing (torchio.Resize)."
        )
        resized = tio.Resize(tuple(target_shape))(reoriented[np.newaxis])
        reoriented = np.squeeze(resized)

    return reoriented.astype(np.float16)


def build_external_step2_dir(
    base_name_to_case: dict[str, str],
    step1_dir: Path,
    manifest_csv: str | Path,
    out_dir: Path,
    adapter: DatasetAdapter,
    persist_copy_dir: Path | None = None,
) -> dict[str, dict]:
    """Populate ``out_dir`` (a step2-shaped dir) from external whole-breast masks.

    ``base_name_to_case`` maps each STEP-1 basename (``{case_id}_{phase_stem}``)
    to its exam ``case_id`` -- exactly what ``preprocess_parallel`` returns. One
    mask is built per exam (from phase 0) and written to EVERY phase's STEP-1
    basename for that exam (the no-motion-correction reuse-across-phases
    approximation; see module docstring).

    Fail-closed: validates all case_ids up front. Writes ``{basename}.npy``
    (float16, native STEP-1 shape) matching STEP-3's ``additional_input_dir``
    contract. If ``persist_copy_dir`` is given, also writes a copy there
    (outside any --cleanup temp dir) so a later comparison can read the exact
    masks that were used.
    """
    step1_dir = Path(step1_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if persist_copy_dir is not None:
        persist_copy_dir = Path(persist_copy_dir)
        persist_copy_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(manifest_csv)
    case_ids = sorted(set(base_name_to_case.values()))
    validate_masks_available(case_ids, manifest)

    # Group STEP-1 basenames by exam so each exam's mask is built exactly once.
    phases_by_case: dict[str, list[str]] = {}
    for base_name, case_id in base_name_to_case.items():
        phases_by_case.setdefault(case_id, []).append(base_name)

    extract_dir = out_dir / "_extracted_masks"
    report: dict[str, dict] = {}
    for case_id in case_ids:
        target_shape = None
        # native STEP-1 shape for this exam (any of its phases; all identical)
        for base_name in phases_by_case[case_id]:
            step1_file = step1_dir / f"{base_name}.npy"
            if step1_file.exists():
                target_shape = np.load(step1_file, mmap_mode="r").shape
                break

        mask = build_whole_breast_mask(
            case_id, manifest, extract_dir, adapter, target_shape=target_shape
        )

        n_written = 0
        for base_name in phases_by_case[case_id]:
            np.save(out_dir / f"{base_name}.npy", mask)
            if persist_copy_dir is not None:
                np.save(persist_copy_dir / f"{base_name}.npy", mask)
            n_written += 1

        _, archive_member = _resolve_mask_member(case_id, manifest)
        report[case_id] = {
            "n_phases": n_written,
            "mask_shape": tuple(int(s) for s in mask.shape),
            "mask_voxel_count": int((mask > 0).sum()),
            "source_member": archive_member,
        }
        print(
            f"  [ok] {case_id}: mask {report[case_id]['mask_shape']} "
            f"({report[case_id]['mask_voxel_count']} breast voxels) "
            f"-> {n_written} phase(s)"
        )

    return report


def _cli() -> None:
    """Standalone conversion, for debugging against an existing STEP-1 dir."""
    import argparse
    import sys

    here = Path(__file__).resolve().parent
    repo_root = here.parent
    for p in (str(repo_root), str(here)):
        if p not in sys.path:
            sys.path.insert(0, p)
    from cohorts.uchicago import UChicagoDataset

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--step1-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--manifest-csv", default=DEFAULT_MANIFEST_CSV)
    ap.add_argument(
        "--dataset-root",
        default="/gpfs/data/karczmar-lab/vanguard/dce2d_internal_ultrafast_manifest",
    )
    args = ap.parse_args()

    # Reconstruct base_name -> case_id from the STEP-1 dir. UChicago basenames
    # are {case_id}_phase_XXXX; strip the trailing _phase_XXXX to get case_id.
    base_name_to_case = {}
    for npy in sorted(args.step1_dir.glob("*.npy")):
        base = npy.stem
        case_id = base.rsplit("_phase_", 1)[0]
        base_name_to_case[base] = case_id

    adapter = UChicagoDataset(root=args.dataset_root)
    report = build_external_step2_dir(
        base_name_to_case, args.step1_dir, args.manifest_csv, args.out_dir, adapter
    )
    print(f"\n[done] {len(report)} exam(s) -> {args.out_dir}")


if __name__ == "__main__":
    _cli()
