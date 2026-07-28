"""MAMA-MIA nnU-Net tumor-mask workflow for paired UChicago HR/UFAST DCE."""

from __future__ import annotations

import argparse
import csv
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
from preprocessing.spatial import resample_to_geometry, save_nifti_xyz

POLICY_NAME = "mama_mia_first_postcontrast_hr_tumor_v1"
FIRST_POSTCONTRAST_PHASE = 1
TEMPORAL_INSTABILITY_DICE_THRESHOLD = 0.60
TINY_VOLUME_ML_THRESHOLD = 0.10
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
DEFAULT_SELECTED_CASES = Path(
    "results/uchicago_nnunet_timepoint_pilot/"
    "all_datasets_random10_per_dataset_seed20260727/selected_cases.csv"
)
DEFAULT_MODEL_DIR = Path(
    "/ess/scratch/scratch1/aakrithiram/nnunet_vanguard/results/"
    "full_image_dce_mri_tumor_segmentation"
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
                input_link = Path("")
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
                "archive_path": str(hr.archive_path),
                "selected_dicom_sha256": hr.source_sha256,
                "series_instance_uid": record.hr_series_instance_uid,
                "geometry": hr.geometry.to_dict(),
                "times_seconds": hr.times_seconds.tolist(),
                "temporal_positions": list(hr.temporal_positions),
                "shared_4d_window": _shared_window(hr.signal_tzyx),
            },
            "ufast_source": {
                "archive_path": str(ufast.archive_path),
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


def prepare_cohort(args: argparse.Namespace) -> None:
    selected = pd.read_csv(args.selected_cases, dtype=str)
    case_manifest = pd.read_csv(args.case_manifest, dtype=str)
    cases = selected[["exam_id", "dataset"]].merge(
        case_manifest,
        on=["exam_id", "dataset"],
        how="inner",
        validate="one_to_one",
    )
    if len(cases) != len(selected):
        missing = selected.loc[~selected["exam_id"].isin(cases["exam_id"])]
        raise RuntimeError(f"selected cases missing runnable HR rows:\n{missing}")

    all_rows = []
    job_rows = []
    for row in cases.sort_values("exam_id").itertuples(index=False):
        provenance = prepare_case_for_tumor(
            inventory=args.inventory,
            case_manifest=args.case_manifest,
            exam_id=str(row.exam_id),
            output_root=args.output_root,
        )
        case_root = _case_root(args.output_root, str(row.exam_id))
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
            "selected_cases": str(args.selected_cases.expanduser().resolve()),
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


def _component_outputs(mask: np.ndarray, img: nib.Nifti1Image, out_dir: Path) -> dict[str, Any]:
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
        "primary_mask": primary,
    }


def finalize_cohort(args: argparse.Namespace) -> None:
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
        primary_ufast = (
            resample_to_geometry(primary_hr_zyx, hr_geometry, ufast_geometry, nearest=True)
            > BINARY_THRESHOLD
        )
        primary_ufast_path = case_root / "tumor_masks" / "tumor_primary_ufast.nii.gz"
        save_nifti_xyz(primary_ufast_path, np.transpose(primary_ufast.astype(np.float32), (2, 1, 0)), ufast_geometry)

        alignment_status = str(provenance["identity_alignment_qc"]["status"])
        mapping_status = "pass" if alignment_status == "pass" else "review_required"
        exclude_downstream = mapping_status != "pass"
        exclude_reason = "alignment_review_required" if exclude_downstream else ""

        rows.append(
            {
                "exam_id": exam_id,
                "hr_series_instance_uid": case.hr_series_instance_uid,
                "ufast_series_instance_uid": case.ufast_series_instance_uid,
                "first_postcontrast_phase": FIRST_POSTCONTRAST_PHASE,
                "first_postcontrast_time_seconds": float(first_row["time_seconds"]),
                "n_later_postcontrast_phases": int(case_phases["phase_index"].gt(FIRST_POSTCONTRAST_PHASE).sum()),
                "median_first_to_later_dice": median_later_dice,
                "all_components_volume_ml": component_info["all_components_volume_ml"],
                "primary_component_volume_ml": primary_volume_ml,
                "opposite_breast_component_ml": None,
                "opposite_to_primary_volume_ratio": None,
                "tumor_mask_empty": tumor_empty,
                "review_tiny": review_tiny,
                "review_temporal_instability": review_temporal,
                "bilateral_candidate": None,
                "bilateral_review_status": "not_evaluated_no_aligned_bilateral_hr_mask",
                "alignment_qc_status": alignment_status,
                "mapping_status": mapping_status,
                "exclude_downstream": exclude_downstream,
                "exclude_reason": exclude_reason,
                "primary_hr_mask": component_info["primary_path"],
                "primary_ufast_mask": str(primary_ufast_path),
                "all_components_hr_mask": component_info["all_components_path"],
                "qc_panel": "",
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(args.output_root / "tumor_mask_manifest.csv", index=False)
    print(f"wrote {args.output_root / 'tumor_mask_manifest.csv'}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output-root", type=Path, required=True)
    common.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    common.add_argument("--case-manifest", type=Path, default=DEFAULT_CASE_MANIFEST)
    common.add_argument("--selected-cases", type=Path, default=DEFAULT_SELECTED_CASES)
    common.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    sub.add_parser("prepare-cohort", parents=[common])
    sub.add_parser("finalize-cohort", parents=[common])
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "prepare-cohort":
        prepare_cohort(args)
    elif args.command == "finalize-cohort":
        finalize_cohort(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
