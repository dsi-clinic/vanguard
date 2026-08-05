"""MAMA-MIA nnU-Net tumor-mask workflow for paired UChicago HR/UFAST DCE.

The released model and preprocessing plan are from Garrucho et al., Scientific
Data 12, 453 (2025), https://doi.org/10.1038/s41597-025-04707-4, and the
official MAMA-MIA nnU-Net fork.

Aakrithi Ram designed and implemented the initial UChicago HR/UFAST tumor
segmentation and shared-window visual-QC workflow and validated it on a 30-case
pilot. Anna Woodard generalized that implementation for restartable
cohort-scale execution, multi-source publication, and longitudinal laterality
provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

from preprocessing.cases import select_case
from preprocessing.dicom import (
    DicomGeometry,
    geometry_alignment_checks,
    load_dicom_series,
)
from preprocessing.motion import DEFAULT_MOTION_SETTINGS, correct_phase
from preprocessing.pipeline import _identity_alignment_qc
from preprocessing.spatial import (
    isotropic_geometry,
    resample_to_geometry,
    save_nifti_xyz,
)

POLICY_NAME = "mama_mia_first_postcontrast_hr_tumor_v1"
FIRST_POSTCONTRAST_PHASE = 1
TEMPORAL_INSTABILITY_DICE_THRESHOLD = 0.60
TINY_VOLUME_ML_THRESHOLD = 0.10
BILATERAL_MIN_OPPOSITE_VOLUME_ML = 0.10
BILATERAL_MIN_OPPOSITE_TO_PRIMARY_RATIO = 0.10
COMPONENT_CONNECTIVITY = 3
BINARY_THRESHOLD = 0.5

DEFAULT_INVENTORY = Path(
    "/gpfs/data/karczmar-lab/vanguard/dce2d_internal_ultrafast_manifest/"
    "paired_hr_ufast_source_dicom/dicom_file_manifest.parquet"
)
DEFAULT_CASE_MANIFEST = Path(
    "/gpfs/data/karczmar-lab/vanguard/dce2d_internal_ultrafast_manifest/"
    "paired_hr_ufast_source_dicom/paired_preprocessing_case_manifest.csv"
)
DEFAULT_COHORT_MANIFEST = Path(
    "/gpfs/data/karczmar-lab/vanguard/dce2d_internal_ultrafast_manifest/"
    "dce2d_internal_ultrafast_with_high_resolution_manifest.csv"
)
DEFAULT_MODEL_DIR = Path(
    "/gpfs/data/karczmar-lab/MAMA-MIA-syn60868042/"
    "nnUNet_pretrained_weights/full_image_dce_mri_tumor_segmentation"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _shared_window(signal_tzyx: np.ndarray) -> list[float]:
    values = np.asarray(signal_tzyx)[np.asarray(signal_tzyx) > 0]
    if not values.size:
        raise ValueError("HR DCE sequence has no positive values")
    return [float(value) for value in np.percentile(values, [0.5, 99.5])]


def _motion_correct_hr(signal_tzyx: np.ndarray, geometry: DicomGeometry) -> tuple[np.ndarray, list[dict[str, Any]]]:
    signal_tzyx = np.asarray(signal_tzyx, dtype=np.float32)
    fixed_xyz = np.transpose(signal_tzyx[0], (2, 1, 0))
    corrected = [signal_tzyx[0]]
    metrics: list[dict[str, Any]] = [
        {
            "phase_index": 0,
            "transform_accepted": True,
            "transform_rejection_reason": "reference_phase",
            "translation_voxels": [0.0, 0.0, 0.0],
        }
    ]
    for phase_index, moving_zyx in enumerate(signal_tzyx[1:], start=1):
        corrected_xyz, _, phase_metrics = correct_phase(
            fixed_xyz,
            np.transpose(moving_zyx, (2, 1, 0)),
            support=None,
            spacing_xyz_mm=np.asarray(geometry.spacing_xyz_mm),
            settings=DEFAULT_MOTION_SETTINGS,
        )
        corrected.append(np.transpose(corrected_xyz, (2, 1, 0)))
        metrics.append({"phase_index": phase_index, **phase_metrics})
    return np.stack(corrected, axis=0), metrics


def _case_root(output_root: Path, exam_id: str) -> Path:
    return output_root.expanduser().resolve() / "work" / exam_id


def _safe_symlink(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        if link.resolve() != target.resolve():
            raise FileExistsError(f"{link} points to {link.resolve()}, not {target}")
        return
    link.symlink_to(target.resolve())


def prepare_case_for_tumor(
    *,
    inventory: Path,
    case_manifest: Path,
    exam_id: str,
    output_root: Path,
) -> dict[str, Any]:
    """Prepare one exact paired case for MAMA-MIA tumor nnU-Net inference."""
    record = select_case(case_manifest, exam_id)
    case_root = _case_root(output_root, exam_id)
    provenance_path = case_root / "tumor_preprocessing_provenance.json"
    if provenance_path.exists():
        return json.loads(provenance_path.read_text())
    if case_root.exists():
        raise FileExistsError(f"refusing to reuse partial case directory: {case_root}")

    case_root.mkdir(parents=True)
    try:
        hr = load_dicom_series(
            inventory,
            study_uid=record.study_instance_uid,
            series_uid=record.hr_series_instance_uid,
        )
        ufast = load_dicom_series(
            inventory,
            study_uid=record.study_instance_uid,
            series_uid=record.ufast_series_instance_uid,
        )
        if hr.signal_tzyx.shape[0] <= FIRST_POSTCONTRAST_PHASE:
            raise ValueError(f"{exam_id}: HR series has no first postcontrast phase")
        if not np.all(np.diff(hr.times_seconds) > 0):
            raise ValueError(f"{exam_id}: HR DICOM times are not increasing")

        geometry_checks = geometry_alignment_checks(hr.geometry, ufast.geometry)
        alignment_qc = _identity_alignment_qc(
            hr,
            ufast,
            baseline_frame_count=record.ufast_baseline_frame_count,
        )
        corrected_hr_tzyx, motion_metrics = _motion_correct_hr(hr.signal_tzyx, hr.geometry)
        hr_dir = case_root / "hr_images"
        nnunet_input_dir = case_root / "nnunet_postcontrast_inputs"
        rows = []
        for phase_index, phase_zyx in enumerate(corrected_hr_tzyx):
            image_path = hr_dir / f"{exam_id}__hr_tp{phase_index:04d}_0000.nii.gz"
            save_nifti_xyz(image_path, np.transpose(phase_zyx, (2, 1, 0)), hr.geometry)
            prediction_path = (
                case_root
                / "nnunet_postcontrast_predictions"
                / f"{exam_id}__hr_tp{phase_index:04d}.nii.gz"
            )
            if phase_index >= FIRST_POSTCONTRAST_PHASE:
                input_link = nnunet_input_dir / image_path.name
                _safe_symlink(image_path, input_link)
            else:
                input_link = Path()
            rows.append(
                {
                    "exam_id": exam_id,
                    "dataset": record.dataset,
                    "phase_index": phase_index,
                    "time_seconds": float(hr.times_seconds[phase_index]),
                    "role": (
                        "precontrast"
                        if phase_index == 0
                        else (
                            "final_candidate"
                            if phase_index == FIRST_POSTCONTRAST_PHASE
                            else "temporal_qc"
                        )
                    ),
                    "hr_image": str(image_path),
                    "nnunet_input": str(input_link) if phase_index >= FIRST_POSTCONTRAST_PHASE else "",
                    "prediction": str(prediction_path) if phase_index >= FIRST_POSTCONTRAST_PHASE else "",
                }
            )

        pd.DataFrame(rows).to_csv(case_root / "hr_phase_manifest.csv", index=False)
        provenance = {
            "status": "prepared",
            "policy": {
                "name": POLICY_NAME,
                "tumor_source": "first postcontrast HR image",
                "first_postcontrast_phase": FIRST_POSTCONTRAST_PHASE,
                "precontrast_phase": 0,
                "later_phases": "temporal QC only",
                "breast_mask": "not used as nnU-Net input and not intersected with tumor output",
                "hr_motion_reference_phase": 0,
                "hr_motion_output_interpolator": "linear",
                "nnunet_normalization": "handled by released nnU-Net 3d_fullres plan",
                "nnunet_resampling": "handled by released nnU-Net 3d_fullres plan",
            },
            "case": record.__dict__,
            "inventory_path": str(inventory.expanduser().resolve()),
            "case_manifest_path": str(case_manifest.expanduser().resolve()),
            "hr_source": {
                "source_kind": hr.source_kind,
                "source_location": str(hr.archive_path),
                "archive_path": (
                    str(hr.archive_path) if hr.source_kind == "zip" else None
                ),
                "selected_dicom_sha256": hr.source_sha256,
                "series_instance_uid": record.hr_series_instance_uid,
                "geometry": hr.geometry.to_dict(),
                "times_seconds": hr.times_seconds.tolist(),
                "temporal_positions": list(hr.temporal_positions),
                "shared_4d_window": _shared_window(hr.signal_tzyx),
            },
            "ufast_source": {
                "source_kind": ufast.source_kind,
                "source_location": str(ufast.archive_path),
                "archive_path": (
                    str(ufast.archive_path) if ufast.source_kind == "zip" else None
                ),
                "selected_dicom_sha256": ufast.source_sha256,
                "series_instance_uid": record.ufast_series_instance_uid,
                "geometry": ufast.geometry.to_dict(),
                "times_seconds": ufast.times_seconds.tolist(),
                "temporal_positions": list(ufast.temporal_positions),
                "baseline_frame_count": record.ufast_baseline_frame_count,
            },
            "geometry_alignment": geometry_checks,
            "identity_alignment_qc": alignment_qc,
            "hr_motion_qc": {"reference_phase": 0, "metrics": motion_metrics},
            "outputs": {
                "hr_images": str(hr_dir),
                "nnunet_postcontrast_inputs": str(nnunet_input_dir),
                "nnunet_postcontrast_predictions": str(case_root / "nnunet_postcontrast_predictions"),
            },
        }
        _write_json(provenance_path, provenance)
        return provenance
    except Exception:
        import shutil

        shutil.rmtree(case_root, ignore_errors=True)
        raise


def _selected_cases(args: argparse.Namespace) -> pd.DataFrame:
    """Return the requested cases, or every runnable manifest row."""
    case_manifest = pd.read_csv(args.case_manifest, dtype=str)
    if args.selected_cases is None:
        return case_manifest.sort_values("exam_id").reset_index(drop=True)
    selected = pd.read_csv(args.selected_cases, dtype=str)
    cases = selected[["exam_id", "dataset"]].merge(
        case_manifest,
        on=["exam_id", "dataset"],
        how="inner",
        validate="one_to_one",
    )
    if len(cases) != len(selected):
        missing = selected.loc[~selected["exam_id"].isin(cases["exam_id"])]
        raise RuntimeError(f"selected cases missing runnable HR rows:\n{missing}")
    return cases.sort_values("exam_id").reset_index(drop=True)


def prepare_case_command(args: argparse.Namespace) -> None:
    """Prepare one stable case-manifest row for use in a Slurm array."""
    if args.exam_id is not None:
        exam_id = args.exam_id
    else:
        index = args.array_index
        if index is None:
            import os

            index = int(os.environ["SLURM_ARRAY_TASK_ID"])
        cases = _selected_cases(args)
        if index < 0 or index >= len(cases):
            raise IndexError(f"array index {index} outside [0, {len(cases) - 1}]")
        exam_id = str(cases.iloc[index]["exam_id"])
    provenance = prepare_case_for_tumor(
        inventory=args.inventory,
        case_manifest=args.case_manifest,
        exam_id=exam_id,
        output_root=args.output_root,
    )
    print(f"prepared {exam_id}: {provenance['outputs']['nnunet_postcontrast_inputs']}")


def write_cohort_manifests(args: argparse.Namespace) -> None:
    """Index already prepared cases and pin the released model provenance."""
    cases = _selected_cases(args)

    all_rows = []
    job_rows = []
    for row in cases.sort_values("exam_id").itertuples(index=False):
        case_root = _case_root(args.output_root, str(row.exam_id))
        provenance_path = case_root / "tumor_preprocessing_provenance.json"
        if not provenance_path.is_file():
            raise FileNotFoundError(f"case is not prepared: {provenance_path}")
        provenance = json.loads(provenance_path.read_text())
        phase_manifest = pd.read_csv(case_root / "hr_phase_manifest.csv")
        all_rows.append(phase_manifest)
        job_rows.append(
            {
                "exam_id": row.exam_id,
                "dataset": row.dataset,
                "hr_series_instance_uid": row.hr_series_instance_uid,
                "ufast_series_instance_uid": row.ufast_series_instance_uid,
                "input_dir": provenance["outputs"]["nnunet_postcontrast_inputs"],
                "output_dir": provenance["outputs"]["nnunet_postcontrast_predictions"],
                "n_postcontrast_phases": int((phase_manifest["phase_index"] >= FIRST_POSTCONTRAST_PHASE).sum()),
            }
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(job_rows).to_csv(args.output_root / "tumor_case_manifest.csv", index=False)
    pd.concat(all_rows, ignore_index=True).to_csv(args.output_root / "tumor_phase_manifest.csv", index=False)
    _write_json(
        args.output_root / "tumor_run_provenance.json",
        {
            "policy": POLICY_NAME,
            "inventory": str(args.inventory.expanduser().resolve()),
            "case_manifest": str(args.case_manifest.expanduser().resolve()),
            "selected_cases": (
                "all runnable case-manifest rows"
                if args.selected_cases is None
                else str(args.selected_cases.expanduser().resolve())
            ),
            "model_dir": str(args.model_dir.expanduser().resolve()),
            "model_dir_sha256s": {
                str(path.relative_to(args.model_dir)): _sha256(path)
                for path in sorted(args.model_dir.glob("fold_*/checkpoint_final.pth"))
            },
            "dataset_json": json.loads((args.model_dir / "dataset.json").read_text()),
            "plans_json_sha256": _sha256(args.model_dir / "plans.json"),
            "folds": [0, 1, 2, 3, 4],
            "first_postcontrast_phase": FIRST_POSTCONTRAST_PHASE,
            "note": "nnU-Net handles 3d_fullres 1 mm resampling and ZScoreNormalization internally.",
        },
    )
    print(f"wrote {args.output_root / 'tumor_case_manifest.csv'}")
    print(f"wrote {args.output_root / 'tumor_phase_manifest.csv'}")
    print(f"n_cases={len(job_rows)}")
    print(f"n_postcontrast_inputs={sum(row['n_postcontrast_phases'] for row in job_rows)}")


def prepare_cohort(args: argparse.Namespace) -> None:
    """Prepare cases sequentially, then write the cohort manifests."""
    for row in _selected_cases(args).itertuples(index=False):
        prepare_case_for_tumor(
            inventory=args.inventory,
            case_manifest=args.case_manifest,
            exam_id=str(row.exam_id),
            output_root=args.output_root,
        )
    write_cohort_manifests(args)


def _load_mask(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    image = nib.load(path)
    return np.asanyarray(image.dataobj) > 0, image


def _dice(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    if not a.any() and not b.any():
        return None
    denom = int(a.sum() + b.sum())
    return float((2.0 * np.logical_and(a, b).sum()) / denom) if denom else None


def _component_outputs(
    mask: np.ndarray, img: nib.Nifti1Image, out_dir: Path
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_path = out_dir / "tumor_all_components_hr.nii.gz"
    primary_path = out_dir / "tumor_primary_hr.nii.gz"
    labeled, n_components = ndimage.label(mask, structure=np.ones((3, 3, 3), dtype=np.uint8))
    if n_components:
        sizes = ndimage.sum(mask, labeled, index=np.arange(1, n_components + 1))
        primary_label = int(np.argmax(sizes) + 1)
        primary = labeled == primary_label
    else:
        primary_label = 0
        primary = np.zeros_like(mask, dtype=bool)
    voxel_ml = float(abs(np.linalg.det(img.affine[:3, :3])) / 1000.0)
    field_center_xyz = (np.asarray(mask.shape, dtype=float) - 1.0) / 2.0
    field_center_ras_x = float(
        nib.affines.apply_affine(img.affine, field_center_xyz)[0]
    )
    components = []
    for label in range(1, n_components + 1):
        component = labeled == label
        centroid_xyz = np.asarray(ndimage.center_of_mass(component), dtype=float)
        centroid_ras = nib.affines.apply_affine(img.affine, centroid_xyz)
        components.append(
            {
                "label": label,
                "volume_ml": float(component.sum() * voxel_ml),
                "centroid_xyz": centroid_xyz.tolist(),
                "centroid_ras_mm": np.asarray(centroid_ras, dtype=float).tolist(),
                "laterality": (
                    "right"
                    if float(centroid_ras[0]) > field_center_ras_x
                    else "left"
                ),
            }
        )
    for path, arr in ((all_path, mask), (primary_path, primary)):
        out = nib.Nifti1Image(arr.astype(np.uint8), img.affine, img.header)
        out.set_data_dtype(np.uint8)
        nib.save(out, path)
    return {
        "all_components_path": str(all_path),
        "primary_path": str(primary_path),
        "n_components": int(n_components),
        "primary_label": primary_label,
        "all_components_volume_ml": float(mask.sum() * voxel_ml),
        "primary_component_volume_ml": float(primary.sum() * voxel_ml),
        "field_center_ras_x_mm": field_center_ras_x,
        "components": components,
        "primary_mask": primary,
    }


def finalize_cohort(args: argparse.Namespace) -> None:
    """Select the primary mask, map it to UFAST, and write the cohort manifest."""
    cases = pd.read_csv(args.output_root / "tumor_case_manifest.csv", dtype=str)
    phase_manifest = pd.read_csv(args.output_root / "tumor_phase_manifest.csv")
    rows = []
    for case in cases.itertuples(index=False):
        exam_id = str(case.exam_id)
        case_root = _case_root(args.output_root, exam_id)
        provenance = json.loads((case_root / "tumor_preprocessing_provenance.json").read_text())
        case_phases = phase_manifest.loc[phase_manifest["exam_id"].astype(str).eq(exam_id)].copy()
        first_row = case_phases.loc[case_phases["phase_index"].eq(FIRST_POSTCONTRAST_PHASE)].iloc[0]
        first_pred = Path(first_row["prediction"])
        first_mask, first_img = _load_mask(first_pred)
        component_info = _component_outputs(first_mask, first_img, case_root / "tumor_masks")

        later_dice = []
        for _, later in case_phases.loc[case_phases["phase_index"].gt(FIRST_POSTCONTRAST_PHASE)].iterrows():
            later_pred = Path(later["prediction"])
            if not later_pred.exists():
                continue
            later_mask, _ = _load_mask(later_pred)
            value = _dice(first_mask, later_mask)
            if value is not None:
                later_dice.append(value)
        median_later_dice = float(np.median(later_dice)) if later_dice else None
        tumor_empty = not bool(first_mask.any())
        primary_volume_ml = float(component_info["primary_component_volume_ml"])
        primary_component = next(
            (
                component
                for component in component_info["components"]
                if component["label"] == component_info["primary_label"]
            ),
            None,
        )
        tumor_laterality_direct = (
            "unknown"
            if primary_component is None
            else str(primary_component["laterality"])
        )
        opposite_components = [
            component
            for component in component_info["components"]
            if component["laterality"] != tumor_laterality_direct
        ]
        opposite_volume_ml = max(
            (float(component["volume_ml"]) for component in opposite_components),
            default=0.0,
        )
        opposite_ratio = (
            opposite_volume_ml / primary_volume_ml
            if primary_volume_ml > 0
            else None
        )
        bilateral_candidate = bool(
            opposite_volume_ml >= BILATERAL_MIN_OPPOSITE_VOLUME_ML
            and opposite_ratio is not None
            and opposite_ratio >= BILATERAL_MIN_OPPOSITE_TO_PRIMARY_RATIO
        )
        review_temporal = bool(
            tumor_empty
            or median_later_dice is None
            or median_later_dice < TEMPORAL_INSTABILITY_DICE_THRESHOLD
        )
        review_tiny = bool(primary_volume_ml < TINY_VOLUME_ML_THRESHOLD)

        primary_hr_zyx = np.transpose(
            component_info["primary_mask"].astype(np.float32), (2, 1, 0)
        )
        hr_geometry = DicomGeometry.from_dict(provenance["hr_source"]["geometry"])
        ufast_geometry = DicomGeometry.from_dict(provenance["ufast_source"]["geometry"])
        # Vanguard's published UFAST phases and mapped vessel skeletons use the
        # source-aligned 1 mm output grid, not the native DICOM voxel grid.  Keep
        # the tumor mask on that exact downstream grid so array indices, shape,
        # and affine all agree without a second consumer-side resampling step.
        ufast_output_geometry = isotropic_geometry(ufast_geometry, spacing_mm=1.0)
        primary_ufast = (
            resample_to_geometry(
                primary_hr_zyx,
                hr_geometry,
                ufast_output_geometry,
                nearest=True,
            )
            > BINARY_THRESHOLD
        )
        primary_ufast_path = case_root / "tumor_masks" / "tumor_primary_ufast.nii.gz"
        save_nifti_xyz(
            primary_ufast_path,
            np.transpose(primary_ufast.astype(np.float32), (2, 1, 0)),
            ufast_output_geometry,
        )

        provenance["tumor_mapping"] = {
            "source": "motion-corrected HR first-postcontrast primary component",
            "target": "Vanguard source-aligned 1 mm UFAST output grid",
            "transform": "physical-coordinate mapping in shared DICOM frame",
            "interpolator": "nearest_neighbor",
            "target_geometry": ufast_output_geometry.to_dict(),
            "output_path": str(primary_ufast_path),
            "output_voxels": int(primary_ufast.sum()),
        }
        _write_json(
            case_root / "tumor_preprocessing_provenance.json",
            provenance,
        )

        alignment_status = str(provenance["identity_alignment_qc"]["status"])
        mapping_status = "pass" if alignment_status == "pass" else "review_required"
        exclusion_reasons = []
        if mapping_status != "pass":
            exclusion_reasons.append("alignment_review_required")
        if bilateral_candidate:
            exclusion_reasons.append("bilateral_candidate_review_required")
        exclude_downstream = bool(exclusion_reasons)
        exclude_reason = ";".join(exclusion_reasons)

        rows.append(
            {
                "exam_id": exam_id,
                "dataset": case.dataset,
                "hr_series_instance_uid": case.hr_series_instance_uid,
                "ufast_series_instance_uid": case.ufast_series_instance_uid,
                "first_postcontrast_phase": FIRST_POSTCONTRAST_PHASE,
                "first_postcontrast_time_seconds": float(first_row["time_seconds"]),
                "n_later_postcontrast_phases": int(case_phases["phase_index"].gt(FIRST_POSTCONTRAST_PHASE).sum()),
                "median_first_to_later_dice": median_later_dice,
                "all_components_volume_ml": component_info["all_components_volume_ml"],
                "primary_component_volume_ml": primary_volume_ml,
                "tumor_laterality_direct": tumor_laterality_direct,
                "laterality_method": (
                    "primary-component centroid relative to the HR physical "
                    "field center; NIfTI RAS x increases toward patient right"
                ),
                "opposite_breast_component_ml": opposite_volume_ml,
                "opposite_to_primary_volume_ratio": opposite_ratio,
                "tumor_mask_empty": tumor_empty,
                "review_tiny": review_tiny,
                "review_temporal_instability": review_temporal,
                "bilateral_candidate": bilateral_candidate,
                "bilateral_review_status": (
                    "review_required" if bilateral_candidate else "not_suspected"
                ),
                "alignment_qc_status": alignment_status,
                "mapping_status": mapping_status,
                "primary_ufast_grid": "vanguard_source_aligned_1mm_output",
                "exclude_downstream": exclude_downstream,
                "exclude_reason": exclude_reason,
                "primary_hr_mask": component_info["primary_path"],
                "primary_ufast_mask": str(primary_ufast_path),
                "all_components_hr_mask": component_info["all_components_path"],
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(args.output_root / "tumor_mask_manifest.csv", index=False)
    print(f"wrote {args.output_root / 'tumor_mask_manifest.csv'}")
    _write_longitudinal_manifest(args, manifest)


def _write_longitudinal_manifest(
    args: argparse.Namespace, direct_manifest: pd.DataFrame
) -> None:
    """Add every cohort exam and propagate an unambiguous patient tumor side."""
    cohort = pd.read_csv(args.cohort_manifest, dtype=str)
    direct = direct_manifest.copy()
    for column in ("tumor_mask_empty", "review_tiny", "bilateral_candidate"):
        direct[column] = direct[column].astype(bool)
    usable = direct.loc[
        direct["tumor_laterality_direct"].isin(["left", "right"])
        & ~direct["tumor_mask_empty"]
        & ~direct["review_tiny"]
        & ~direct["bilateral_candidate"]
    ]
    patient_lookup = cohort[["exam_id", "patient_id"]].merge(
        usable[["exam_id", "tumor_laterality_direct"]],
        on="exam_id",
        how="inner",
        validate="one_to_one",
    )
    consensus: dict[str, tuple[str, str, bool]] = {}
    for patient_id, group in patient_lookup.groupby("patient_id", sort=False):
        sides = sorted(set(group["tumor_laterality_direct"]))
        source_exams = ";".join(sorted(group["exam_id"]))
        consensus[str(patient_id)] = (
            sides[0] if len(sides) == 1 else "unknown",
            source_exams,
            len(sides) > 1,
        )

    duplicate_source_columns = [
        column
        for column in direct.columns
        if column != "exam_id" and column in cohort.columns
    ]
    direct_for_join = direct.drop(columns=duplicate_source_columns)
    combined = cohort.merge(
        direct_for_join, on="exam_id", how="left", validate="one_to_one"
    )
    final_side = []
    side_source = []
    source_exams = []
    conflicts = []
    for row in combined.itertuples(index=False):
        direct_side = getattr(row, "tumor_laterality_direct", None)
        direct_valid = bool(
            direct_side in {"left", "right"}
            and not bool(getattr(row, "tumor_mask_empty", True))
            and not bool(getattr(row, "review_tiny", True))
            and not bool(getattr(row, "bilateral_candidate", True))
        )
        patient_side, patient_sources, conflict = consensus.get(
            str(row.patient_id), ("unknown", "", False)
        )
        if conflict:
            final_side.append("unknown")
            side_source.append("unknown_patient_conflict")
            source_exams.append(patient_sources)
        elif direct_valid:
            final_side.append(direct_side)
            side_source.append("direct_segmentation")
            source_exams.append(str(row.exam_id))
        elif patient_side in {"left", "right"}:
            final_side.append(patient_side)
            side_source.append("same_patient_consensus")
            source_exams.append(patient_sources)
        else:
            final_side.append("unknown")
            side_source.append("unknown_no_unambiguous_valid_source")
            source_exams.append(patient_sources)
        conflicts.append(conflict)
    combined["tumor_laterality"] = final_side
    combined["tumor_laterality_source"] = side_source
    combined["tumor_laterality_source_exam_ids"] = source_exams
    combined["patient_laterality_conflict"] = conflicts
    direct_exclusion = combined["exclude_downstream"].eq(True)
    combined["exclude_downstream"] = direct_exclusion | combined[
        "patient_laterality_conflict"
    ]
    direct_reasons = combined["exclude_reason"].fillna("").astype(str)
    combined["exclude_reason"] = [
        ";".join(
            reason
            for reason in (
                direct_reason,
                "patient_laterality_conflict" if conflict else "",
            )
            if reason
        )
        for direct_reason, conflict in zip(
            direct_reasons,
            combined["patient_laterality_conflict"],
            strict=True,
        )
    ]
    combined["tumor_segmentation_status"] = np.where(
        combined["primary_hr_mask"].notna(),
        "segmented",
        combined.get("hr_selection_status", pd.Series("not_runnable", index=combined.index)),
    )
    if args.centerline_root is not None:
        centerline_rows = []
        for case_dir in sorted(args.centerline_root.expanduser().resolve().glob("*/*")):
            exam_id = case_dir.name
            skeleton = case_dir / f"{exam_id}_skeleton_4d_exam_mask.npy"
            support = case_dir / f"{exam_id}_skeleton_4d_exam_support_mask.npy"
            morphometry = case_dir / f"{exam_id}_morphometry.json"
            if skeleton.is_file() or support.is_file() or morphometry.is_file():
                centerline_rows.append(
                    {
                        "exam_id": exam_id,
                        "skeleton_path": str(skeleton) if skeleton.is_file() else "",
                        "support_path": str(support) if support.is_file() else "",
                        "morphometry_path": (
                            str(morphometry) if morphometry.is_file() else ""
                        ),
                        "centerline_status": (
                            "complete"
                            if all(
                                path.is_file()
                                for path in (skeleton, support, morphometry)
                            )
                            else "incomplete"
                        ),
                    }
                )
        centerlines = pd.DataFrame(centerline_rows)
        if centerlines["exam_id"].duplicated().any():
            raise ValueError("centerline root contains duplicate exam IDs")
        combined = combined.merge(
            centerlines, on="exam_id", how="left", validate="one_to_one"
        )
        combined["centerline_status"] = combined["centerline_status"].fillna(
            "missing"
        )
    output_path = args.output_root / "longitudinal_tumor_manifest.csv"
    combined.to_csv(output_path, index=False)
    print(f"wrote {output_path}")


def publish_cohort(args: argparse.Namespace) -> None:
    """Publish direct masks on the exact student-facing UFAST cohort grid."""
    if args.centerline_root is None:
        raise ValueError("publish-cohort requires --centerline-root")
    cohort_path = args.cohort_manifest.expanduser().resolve()
    cohort = pd.read_csv(cohort_path, dtype=str)
    if cohort["exam_id"].duplicated().any():
        raise ValueError("cohort manifest repeats exam_id")
    cohort_by_exam = cohort.set_index("exam_id")

    parts = []
    for path in args.direct_manifest:
        resolved = path.expanduser().resolve()
        part = pd.read_csv(resolved)
        part["source_direct_manifest"] = str(resolved)
        parts.append(part)
    direct = pd.concat(parts, ignore_index=True)
    if direct["exam_id"].astype(str).duplicated().any():
        repeated = direct.loc[
            direct["exam_id"].astype(str).duplicated(keep=False), "exam_id"
        ].tolist()
        raise ValueError(f"direct tumor manifests repeat exam IDs: {repeated[:3]}")
    direct = direct.loc[
        direct["exam_id"].astype(str).isin(cohort_by_exam.index)
    ].copy()
    if direct.empty:
        raise ValueError("no direct tumor masks overlap the cohort")
    if not direct["primary_ufast_grid"].eq(
        "vanguard_source_aligned_1mm_output"
    ).all():
        raise ValueError("a direct tumor mask isn't on the Vanguard UFAST output grid")

    args.output_root.mkdir(parents=True, exist_ok=True)
    mask_dir = args.output_root / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    checksums = []
    canonical_paths = []
    source_paths = []
    normalized_datasets = []
    centerline_root = args.centerline_root.expanduser().resolve()
    for row in direct.itertuples(index=False):
        exam_id = str(row.exam_id)
        cohort_row = cohort_by_exam.loc[exam_id]
        dataset = str(cohort_row["dataset"])
        source_mask = Path(row.primary_ufast_mask).expanduser().resolve()
        if not source_mask.is_file():
            raise FileNotFoundError(source_mask)
        canonical_mask = mask_dir / f"{exam_id}.nii.gz"
        _safe_symlink(source_mask, canonical_mask)

        phase_paths = [Path(value) for value in json.loads(cohort_row["phase_files"])]
        if not phase_paths or not phase_paths[0].is_file():
            raise FileNotFoundError(f"{exam_id}: published UFAST phase is missing")
        mask_image = nib.load(source_mask)
        phase_image = nib.load(phase_paths[0])
        skeleton_path = (
            centerline_root
            / dataset
            / exam_id
            / f"{exam_id}_skeleton_4d_exam_mask.npy"
        )
        if not skeleton_path.is_file():
            raise FileNotFoundError(skeleton_path)
        skeleton_shape_xyz = tuple(reversed(np.load(skeleton_path, mmap_mode="r").shape))
        if mask_image.shape != phase_image.shape or mask_image.shape != skeleton_shape_xyz:
            raise ValueError(
                f"{exam_id}: tumor/image/skeleton shapes disagree: "
                f"{mask_image.shape}, {phase_image.shape}, {skeleton_shape_xyz}"
            )
        if not np.allclose(mask_image.affine, phase_image.affine):
            raise ValueError(f"{exam_id}: tumor and UFAST image affines disagree")

        source_paths.append(str(source_mask))
        canonical_paths.append(str(canonical_mask))
        normalized_datasets.append(dataset)
        checksums.append(
            {
                "exam_id": exam_id,
                "primary_ufast_mask": str(canonical_mask),
                "sha256": _sha256(source_mask),
            }
        )

    direct["source_dataset"] = direct["dataset"]
    direct["dataset"] = normalized_datasets
    direct["source_primary_ufast_mask"] = source_paths
    direct["primary_ufast_mask"] = canonical_paths
    direct = direct.sort_values("exam_id").reset_index(drop=True)
    expected_mask_names = {Path(path).name for path in canonical_paths}
    for published_mask in mask_dir.glob("*.nii.gz"):
        if published_mask.is_symlink() and published_mask.name not in expected_mask_names:
            published_mask.unlink()
    direct_path = args.output_root / "tumor_mask_manifest.csv"
    direct.to_csv(direct_path, index=False)
    pd.DataFrame(checksums).sort_values("exam_id").to_csv(
        args.output_root / "mask_checksums.csv", index=False
    )
    _write_longitudinal_manifest(args, direct)

    missing = sorted(set(cohort["exam_id"].astype(str)) - set(direct["exam_id"]))
    provenance = {
        "policy": POLICY_NAME,
        "cohort_manifest": str(cohort_path),
        "cohort_manifest_sha256": _sha256(cohort_path),
        "direct_manifests": [
            {
                "path": str(path.expanduser().resolve()),
                "sha256": _sha256(path.expanduser().resolve()),
            }
            for path in args.direct_manifest
        ],
        "centerline_root": str(centerline_root),
        "cohort_exams": int(len(cohort)),
        "direct_masks": int(len(direct)),
        "missing_direct_exam_ids": missing,
        "spatial_validation": (
            "every direct mask has the exact published UFAST phase shape and "
            "affine and the exact published skeleton shape"
        ),
    }
    _write_json(args.output_root / "provenance.json", provenance)
    (args.output_root / "README.md").write_text(
        "# UChicago tumor masks\n\n"
        f"This directory contains {len(direct)} direct primary-tumor masks for "
        f"the {len(cohort)}-exam cohort. Each mask is the largest connected "
        "component from MAMA-MIA inference on the first postcontrast "
        "higher-spatial-resolution image, mapped in physical coordinates to "
        "the exact 1 mm Vanguard UFAST image and vessel-skeleton grid. "
        "Possible bilateral cases remain flagged in the manifests; the flat "
        "`masks/` directory always contains only the primary component.\n\n"
        "`tumor_mask_manifest.csv` lists direct segmentations, "
        "`longitudinal_tumor_manifest.csv` joins tumor status and unambiguous "
        "same-patient laterality to every cohort row, and "
        "`mask_checksums.csv` records mask content checksums.\n",
        encoding="utf-8",
    )
    checksum_paths = [
        args.output_root / "tumor_mask_manifest.csv",
        args.output_root / "longitudinal_tumor_manifest.csv",
        args.output_root / "mask_checksums.csv",
        args.output_root / "provenance.json",
        args.output_root / "README.md",
    ]
    (args.output_root / "SHA256SUMS").write_text(
        "".join(f"{_sha256(path)}  {path.name}\n" for path in checksum_paths),
        encoding="utf-8",
    )
    print(
        f"published {len(direct)}/{len(cohort)} direct masks under "
        f"{args.output_root}; missing_direct={len(missing)}"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output-root", type=Path, required=True)
    common.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    common.add_argument("--case-manifest", type=Path, default=DEFAULT_CASE_MANIFEST)
    common.add_argument(
        "--cohort-manifest", type=Path, default=DEFAULT_COHORT_MANIFEST
    )
    common.add_argument("--centerline-root", type=Path)
    common.add_argument("--selected-cases", type=Path)
    common.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    sub.add_parser("prepare-cohort", parents=[common])
    prepare_case_parser = sub.add_parser("prepare-case", parents=[common])
    prepare_case_parser.add_argument("--exam-id")
    prepare_case_parser.add_argument("--array-index", type=int)
    sub.add_parser("index-cohort", parents=[common])
    sub.add_parser("finalize-cohort", parents=[common])
    publish_parser = sub.add_parser("publish-cohort", parents=[common])
    publish_parser.add_argument(
        "--direct-manifest", action="append", required=True, type=Path
    )
    return parser


def main() -> None:
    """Run one tumor-segmentation workflow stage."""
    args = _parser().parse_args()
    if args.command == "prepare-cohort":
        prepare_cohort(args)
    elif args.command == "prepare-case":
        prepare_case_command(args)
    elif args.command == "index-cohort":
        write_cohort_manifests(args)
    elif args.command == "finalize-cohort":
        finalize_cohort(args)
    elif args.command == "publish-cohort":
        publish_cohort(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
