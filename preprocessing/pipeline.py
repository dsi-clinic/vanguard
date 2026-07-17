"""Vanguard-owned paired HR/UFAST preprocessing and skeleton handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from preprocessing.cases import CaseRecord, select_case
from preprocessing.dicom import (
    DicomGeometry,
    geometry_alignment_checks,
    load_dicom_series,
)
from preprocessing.model import model_subject_id, prepare_hr_phase_for_model
from preprocessing.motion import (
    DEFAULT_MOTION_SETTINGS,
    correct_phase,
    correlation_in_support,
)
from preprocessing.qc import write_mapping_qc
from preprocessing.spatial import (
    isotropic_geometry,
    rasterize_skeleton_identity,
    resample_to_geometry,
    save_nifti_xyz,
)

POLICY_NAME = "vanguard_spgr_raw_signal_v3"
TARGET_SPACING_MM = 1.0
BINARY_THRESHOLD = 0.5
INTERSERIES_REVIEW_TRANSLATION_MM = 2.0
INTERSERIES_MEANINGFUL_NCC_GAIN = 0.02


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _case_root(output_root: Path, exam_id: str) -> Path:
    return output_root.expanduser().resolve() / "work" / exam_id


def _shared_window(signal_tzyx: np.ndarray) -> list[float]:
    values = np.asarray(signal_tzyx)[np.asarray(signal_tzyx) > 0]
    if not values.size:
        raise ValueError("4D signal has no positive values for a shared window")
    return [float(value) for value in np.percentile(values, [0.5, 99.5])]


def _validate_pair(record: CaseRecord, hr: Any, ufast: Any) -> dict[str, object]:
    if not 1 <= record.ufast_baseline_frame_count < ufast.signal_tzyx.shape[0]:
        raise ValueError("UFAST baseline count must be in [1, n_phases)")
    checks = geometry_alignment_checks(hr.geometry, ufast.geometry)
    if not checks["same_frame_of_reference_uid"]:
        raise ValueError("HR and UFAST do not share a DICOM FrameOfReferenceUID")
    return checks


def _identity_alignment_qc(
    hr: Any, ufast: Any, *, baseline_frame_count: int
) -> dict[str, object]:
    """Check whether image content supports the DICOM-identity mapping."""
    hr_phase0_on_ufast = resample_to_geometry(
        hr.signal_tzyx[0], hr.geometry, ufast.geometry
    )
    ufast_baseline = np.asarray(
        ufast.signal_tzyx[:baseline_frame_count].mean(axis=0), dtype=np.float32
    )
    overlap = (hr_phase0_on_ufast > 0) & (ufast_baseline > 0)
    identity_ncc = correlation_in_support(
        hr_phase0_on_ufast,
        ufast_baseline,
        overlap,
        maximum_voxels=DEFAULT_MOTION_SETTINGS.maximum_correlation_voxels,
    )
    try:
        _, _, proposal = correct_phase(
            np.transpose(ufast_baseline, (2, 1, 0)),
            np.transpose(hr_phase0_on_ufast, (2, 1, 0)),
            support=np.transpose(overlap, (2, 1, 0)),
            spacing_xyz_mm=np.asarray(ufast.geometry.spacing_xyz_mm),
            settings=DEFAULT_MOTION_SETTINGS,
        )
        proposed_norm_mm = float(proposal["proposed_translation_norm_mm"])
        proposed_ncc_gain = float(proposal["corr_delta"])
        content_disagrees = bool(
            proposed_norm_mm > INTERSERIES_REVIEW_TRANSLATION_MM
            and proposed_ncc_gain >= INTERSERIES_MEANINGFUL_NCC_GAIN
        )
        status = "review_required" if content_disagrees else "pass"
        reason = (
            "meaningful_translation_would_improve_alignment"
            if content_disagrees
            else "no_meaningful_translation_improvement"
        )
    except ValueError as error:
        proposal = None
        proposed_norm_mm = None
        proposed_ncc_gain = None
        status = "review_required"
        reason = f"translation_diagnostic_failed: {error}"
    if not np.isfinite(identity_ncc):
        status = "review_required"
        reason = "identity_ncc_not_finite"
    return {
        "metric": (
            "HR phase 0 versus mean protocol UFAST baseline under DICOM identity"
        ),
        "baseline_frame_count": baseline_frame_count,
        "identity_ncc": float(identity_ncc) if np.isfinite(identity_ncc) else None,
        "overlap_voxels": int(overlap.sum()),
        "proposed_translation_norm_mm": proposed_norm_mm,
        "proposed_translation_ncc_gain": proposed_ncc_gain,
        "review_translation_threshold_mm": INTERSERIES_REVIEW_TRANSLATION_MM,
        "meaningful_ncc_gain_threshold": INTERSERIES_MEANINGFUL_NCC_GAIN,
        "status": status,
        "reason": reason,
        "translation_diagnostic": proposal,
        "interpretation": (
            "FrameOfReferenceUID establishes a coordinate system, not absence of "
            "inter-series motion. A meaningful translation proposal triggers review; "
            "the proposal is never applied automatically."
        ),
    }


def prepare_case(
    *,
    inventory_path: Path,
    case_manifest: Path,
    exam_id: str,
    output_root: Path,
) -> Path:
    """Load exact DICOM series and create raw UFAST and native-HR derivatives."""
    record = select_case(case_manifest, exam_id)
    case_root = _case_root(output_root, exam_id)
    if case_root.exists():
        raise FileExistsError(
            f"refusing to mix preprocessing runs in existing directory: {case_root}"
        )
    dce_dir = output_root.expanduser().resolve() / "dce" / exam_id
    if dce_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing UFAST derivative: {dce_dir}"
        )
    case_root.mkdir(parents=True)
    try:
        hr = load_dicom_series(
            inventory_path,
            study_uid=record.study_instance_uid,
            series_uid=record.hr_series_instance_uid,
        )
        ufast = load_dicom_series(
            inventory_path,
            study_uid=record.study_instance_uid,
            series_uid=record.ufast_series_instance_uid,
        )
        geometry_checks = _validate_pair(record, hr, ufast)
        identity_alignment_qc = _identity_alignment_qc(
            hr,
            ufast,
            baseline_frame_count=record.ufast_baseline_frame_count,
        )
        model_dir = case_root / "hr_model_inputs"
        model_dir.mkdir(parents=True)
        dce_dir.mkdir(parents=True)

        for phase_index, phase_zyx in enumerate(hr.signal_tzyx):
            np.save(
                model_dir / f"{model_subject_id(phase_index)}.npy",
                prepare_hr_phase_for_model(phase_zyx),
            )
        np.save(case_root / "hr_times_seconds.npy", hr.times_seconds)
        np.save(dce_dir / "ufast_times_seconds.npy", ufast.times_seconds)

        target_geometry = isotropic_geometry(
            ufast.geometry, spacing_mm=TARGET_SPACING_MM
        )
        resampled_txyz = np.stack(
            [
                np.transpose(
                    resample_to_geometry(phase, ufast.geometry, target_geometry),
                    (2, 1, 0),
                )
                for phase in ufast.signal_tzyx
            ],
            axis=0,
        )
        fixed = resampled_txyz[0]
        corrected = [fixed]
        motion_metrics: list[dict[str, object]] = [
            {
                "phase_index": 0,
                "time_seconds": float(ufast.times_seconds[0]),
                "transform_accepted": True,
                "transform_rejection_reason": "reference_phase",
                "translation_voxels": [0.0, 0.0, 0.0],
            }
        ]
        for phase_index, moving in enumerate(resampled_txyz[1:], start=1):
            _, shift, metrics = correct_phase(
                fixed,
                moving,
                support=None,
                spacing_xyz_mm=np.asarray(target_geometry.spacing_xyz_mm),
                settings=DEFAULT_MOTION_SETTINGS,
            )
            if metrics["transform_accepted"]:
                output = np.transpose(
                    resample_to_geometry(
                        ufast.signal_tzyx[phase_index],
                        ufast.geometry,
                        target_geometry,
                        output_shift_xyz_voxels=shift,
                    ),
                    (2, 1, 0),
                )
            else:
                output = moving
            corrected.append(output)
            motion_metrics.append(
                {
                    "phase_index": phase_index,
                    "time_seconds": float(ufast.times_seconds[phase_index]),
                    **metrics,
                }
            )
        corrected_txyz = np.stack(corrected, axis=0)
        if corrected_txyz.shape[0] != ufast.times_seconds.size:
            raise RuntimeError("motion correction changed the number of timepoints")
        if np.any(corrected_txyz < 0) or not np.all(np.isfinite(corrected_txyz)):
            raise RuntimeError("motion output is not finite nonnegative raw signal")
        for phase_index, phase_xyz in enumerate(corrected_txyz):
            save_nifti_xyz(
                dce_dir / f"{exam_id}_{phase_index:04d}.nii.gz",
                phase_xyz,
                target_geometry,
            )

        provenance: dict[str, Any] = {
            "status": "prepared",
            "policy": {
                "name": POLICY_NAME,
                "ufast_clip": "none",
                "ufast_normalization": "none",
                "ufast_resample_spacing_mm": [1.0, 1.0, 1.0],
                "ufast_resample_interpolator": "linear",
                "motion_reference_phase": 0,
                "motion_output_interpolator": "linear",
                "motion_composed_with_spatial_resample": True,
                "saved_phase_interpolation_count": 1,
            },
            "case": record.__dict__,
            "inventory_path": str(inventory_path.expanduser().resolve()),
            "case_manifest_path": str(case_manifest.expanduser().resolve()),
            "case_manifest_sha256": _sha256(case_manifest.expanduser().resolve()),
            "hr_source": {
                "archive_path": str(hr.archive_path),
                "selected_dicom_sha256": hr.source_sha256,
                "geometry": hr.geometry.to_dict(),
                "times_seconds": hr.times_seconds.tolist(),
                "temporal_positions": list(hr.temporal_positions),
                "shared_4d_window": _shared_window(hr.signal_tzyx),
            },
            "ufast_source": {
                "archive_path": str(ufast.archive_path),
                "selected_dicom_sha256": ufast.source_sha256,
                "geometry": ufast.geometry.to_dict(),
                "times_seconds": ufast.times_seconds.tolist(),
                "temporal_positions": list(ufast.temporal_positions),
                "baseline_frame_count": record.ufast_baseline_frame_count,
                "shared_4d_window": _shared_window(ufast.signal_tzyx),
            },
            "ufast_output_geometry": target_geometry.to_dict(),
            "geometry_alignment": geometry_checks,
            "identity_alignment_qc": identity_alignment_qc,
            "model_contract": {
                "series": "native higher-spatial-resolution HR",
                "temporal": "every HR phase in physical acquisition order",
                "array_order": "DICOM z,y,x -> model y,x,z",
                "intensity": "0.1% tail clip then per-volume z-score",
                "spatial": "native HR grid; no resampling",
            },
            "motion_settings": DEFAULT_MOTION_SETTINGS.to_dict(),
            "motion_metrics": motion_metrics,
            "outputs": {
                "hr_model_inputs": str(model_dir),
                "ufast_motion_corrected_images": str(dce_dir),
            },
        }
        _write_json(case_root / "preprocessing_provenance.json", provenance)
    except Exception:
        shutil.rmtree(case_root, ignore_errors=True)
        shutil.rmtree(dce_dir, ignore_errors=True)
        raise
    return case_root


def infer_case(
    *, case_root: Path, breast_model: Path, vessel_model: Path, batch_size: int
) -> None:
    """Run the frozen breast and vessel models over every prepared HR phase."""
    import torch

    from segmentation.batch_segmentation import run_inference_in_process

    if not torch.cuda.is_available():
        raise RuntimeError("vessel inference requires a GPU allocation")
    step1_dir = case_root / "hr_model_inputs"
    breast_dir = case_root / "hr_breast_predictions"
    vessel_dir = case_root / "hr_vessel_predictions"
    if breast_dir.exists() or vessel_dir.exists():
        raise FileExistsError("refusing to overwrite existing model predictions")
    breast_dir.mkdir()
    vessel_dir.mkdir()
    try:
        run_inference_in_process(
            step1_dir,
            breast_dir,
            vessel_dir,
            str(breast_model),
            str(vessel_model),
            batch_size,
            3,
            True,
        )
        provenance_path = case_root / "preprocessing_provenance.json"
        provenance = json.loads(provenance_path.read_text())
        expected = len(provenance["hr_source"]["times_seconds"])
        outputs = sorted(vessel_dir.glob("*.npz"))
        if len(outputs) != expected:
            raise RuntimeError(
                f"expected {expected} vessel phases, found {len(outputs)}"
            )
        provenance["inference"] = {
            "breast_model": str(breast_model.resolve()),
            "breast_model_sha256": _sha256(breast_model),
            "vessel_model": str(vessel_model.resolve()),
            "vessel_model_sha256": _sha256(vessel_model),
            "device": torch.cuda.get_device_name(0),
            "phases": expected,
        }
        _write_json(provenance_path, provenance)
    except Exception:
        shutil.rmtree(breast_dir, ignore_errors=True)
        shutil.rmtree(vessel_dir, ignore_errors=True)
        raise


def tc4d_case(*, case_root: Path) -> None:
    """Run unmodified TC4D over all native-HR vessel probability phases."""
    from graph_extraction.tc4d import run_tc4d_from_priority

    provenance_path = case_root / "preprocessing_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    n_phases = len(provenance["hr_source"]["times_seconds"])
    vessel_dir = case_root / "hr_vessel_predictions"
    paths = [vessel_dir / f"{model_subject_id(index)}.npz" for index in range(n_phases)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing HR vessel phases: {missing[:3]}")
    probabilities = []
    for path in paths:
        with np.load(path, allow_pickle=False) as loaded:
            probabilities.append(np.asarray(loaded["vessel"], dtype=np.float32))
    priority_tyxz = np.stack(probabilities, axis=0)
    result, params, diagnostics = run_tc4d_from_priority(priority_tyxz)
    output_dir = case_root / "hr_tc4d"
    output_dir.mkdir(parents=True, exist_ok=False)
    skeleton_yxz = np.asarray(result["exam_mask"], dtype=bool)
    support_yxz = np.asarray(result["support_mask"], dtype=bool)
    np.save(
        output_dir / "exam_skeleton_zyx.npy",
        np.transpose(skeleton_yxz, (2, 0, 1)).astype(np.uint8),
    )
    np.save(
        output_dir / "exam_support_zyx.npy",
        np.transpose(support_yxz, (2, 0, 1)).astype(np.uint8),
    )
    np.save(
        output_dir / "center_manifold_4d_yxz.npy",
        np.asarray(result["mask_4d"], dtype=np.uint8),
    )
    provenance["tc4d"] = {
        "input_phases": n_phases,
        "input_array_order": "time,y,x,z",
        "skeleton_voxels": int(skeleton_yxz.sum()),
        "support_voxels": int(support_yxz.sum()),
        "effective_min_temporal_support": int(result["effective_min_temporal_support"]),
        "params": params,
        "diagnostics": diagnostics,
    }
    _write_json(provenance_path, provenance)


def map_case(*, case_root: Path) -> None:
    """Map static HR TC4D skeleton/support to the exact 1-mm UFAST grid."""
    provenance_path = case_root / "preprocessing_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    checks = provenance["geometry_alignment"]
    if not checks["same_frame_of_reference_uid"]:
        raise ValueError("physical mapping is not justified by DICOM frame geometry")
    hr_geometry = DicomGeometry.from_dict(provenance["hr_source"]["geometry"])
    target_geometry = DicomGeometry.from_dict(provenance["ufast_output_geometry"])
    skeleton = np.load(case_root / "hr_tc4d" / "exam_skeleton_zyx.npy").astype(bool)
    support = np.load(case_root / "hr_tc4d" / "exam_support_zyx.npy").astype(bool)
    mapped_skeleton, metrics = rasterize_skeleton_identity(
        skeleton, hr_geometry, target_geometry
    )
    mapped_support = (
        resample_to_geometry(
            support.astype(np.float32), hr_geometry, target_geometry, nearest=True
        )
        > BINARY_THRESHOLD
    )
    outside = int(np.logical_and(mapped_skeleton > 0, ~mapped_support).sum())
    mapped_support |= mapped_skeleton > 0
    output_root = case_root.parents[1]
    dataset = str(provenance["case"]["dataset"])
    output_dir = output_root / "centerlines" / dataset / case_root.name
    output_dir.mkdir(parents=True, exist_ok=False)
    exam_id = case_root.name
    np.save(output_dir / f"{exam_id}_skeleton_4d_exam_mask.npy", mapped_skeleton)
    np.save(
        output_dir / f"{exam_id}_skeleton_4d_exam_support_mask.npy",
        mapped_support.astype(np.uint8),
    )
    ufast_times = np.asarray(provenance["ufast_source"]["times_seconds"], dtype=float)
    _write_json(
        output_dir / "run_summary.json",
        {
            "case_id": exam_id,
            "study_timepoints": list(range(int(ufast_times.size))),
            "physical_times_seconds": ufast_times.tolist(),
            "alignment_qc_status": provenance["identity_alignment_qc"]["status"],
            "kinetic_feature_policy": {
                "baseline_frame_count": int(
                    provenance["ufast_source"]["baseline_frame_count"]
                ),
                "baseline_estimator": "mean",
                "enhancement": "relative_signal_change",
                "time_axis": "physical_seconds",
            },
            "skeleton_source": "all native-HR vessel phases via TC4D",
            "kinetic_signal_source": "motion-corrected raw UFAST signal",
        },
    )
    provenance["mapping"] = {
        "transform": "physical-coordinate mapping in shared DICOM FrameOfReferenceUID",
        "interpolation": {
            "skeleton": "physical point rasterization",
            "support": "nearest neighbor",
        },
        "skeleton_outside_support_before_repair": outside,
        **metrics,
    }
    _write_json(provenance_path, provenance)


def qc_case(*, case_root: Path) -> Path:
    """Write a shared-window visual check of the mapped UFAST skeleton."""
    provenance = json.loads((case_root / "preprocessing_provenance.json").read_text())
    output_root = case_root.parents[1]
    exam_id = case_root.name
    dataset = str(provenance["case"]["dataset"])
    centerline_dir = output_root / "centerlines" / dataset / exam_id
    output_path = centerline_dir / "mapping_qc.png"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing QC panel: {output_path}")
    skeleton = np.load(centerline_dir / f"{exam_id}_skeleton_4d_exam_mask.npy").astype(
        bool
    )
    window = tuple(
        float(value) for value in provenance["ufast_source"]["shared_4d_window"]
    )
    write_mapping_qc(
        phase0_nifti=output_root / "dce" / exam_id / f"{exam_id}_0000.nii.gz",
        skeleton_zyx=skeleton,
        output_path=output_path,
        shared_window=window,
    )
    return output_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "infer", "tc4d", "map", "qc", "all")
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--exam-id", required=True)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--case-manifest", type=Path)
    model_root = (
        Path(__file__).resolve().parents[1]
        / "vanguard-blood-vessel-segmentation"
        / "trained_models"
    )
    parser.add_argument(
        "--breast-model", type=Path, default=model_root / "breast_model.pth"
    )
    parser.add_argument(
        "--vessel-model", type=Path, default=model_root / "dv_model.pth"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    return parser


def main() -> None:
    """Run one restartable pipeline stage for one exact case."""
    args = _parser().parse_args()
    case_root = _case_root(args.output_root, args.exam_id)
    if args.stage in {"prepare", "all"}:
        if args.inventory is None or args.case_manifest is None:
            raise ValueError("prepare requires --inventory and --case-manifest")
        case_root = prepare_case(
            inventory_path=args.inventory,
            case_manifest=args.case_manifest,
            exam_id=args.exam_id,
            output_root=args.output_root,
        )
    if args.stage in {"infer", "all"}:
        infer_case(
            case_root=case_root,
            breast_model=args.breast_model,
            vessel_model=args.vessel_model,
            batch_size=args.batch_size,
        )
    if args.stage in {"tc4d", "all"}:
        tc4d_case(case_root=case_root)
    if args.stage in {"map", "all"}:
        map_case(case_root=case_root)
    if args.stage in {"qc", "all"}:
        qc_case(case_root=case_root)


if __name__ == "__main__":
    main()
