#!/usr/bin/env python3
"""Audit Sarit's expanded pCR cohort for duplicate imaging exams.

The audit compares image content rather than relying on patient, study, or series
identifiers.  It uses the released UFAST phase volumes when they exist, reconstructs
baseline/late UFAST volumes from raw DICOM for cases awaiting phase export, and
reconstructs an HR baseline volume for every exam.  Known same-patient technical
re-export exclusions are included as positive controls but never as cohort members.

Raw imaging is read-only.  The only writes are to a new, atomically published audit
directory containing pseudonymous fingerprints, pair scores, QC panels, and
provenance.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import pydicom
from skimage.transform import resize

SCHEMA = "sarit-exam-image-duplicate-audit-v1"
DESCRIPTOR_SHAPE_ZYX = (12, 32, 32)
QC_SLICE_SIZE = 96
QC_Z_FRACTIONS = (0.30, 0.50, 0.70)
VOLUME_DIMENSIONS = 3
SLICE_DIMENSIONS = 2
MINIMUM_SLICE_COUNT = 1
MINIMUM_IN_PLANE_SIZE = 2
MINIMUM_FOREGROUND_VOXELS = 100
EXPECTED_SELECTED_EXAMS = 341
EXPECTED_POSITIVE_CONTROLS = 4
PROBABLE_ANATOMY_CORRELATION = 0.995
PROBABLE_ENHANCEMENT_CORRELATION = 0.980
REVIEW_ANATOMY_CORRELATION = 0.980
REVIEW_COMBINED_SCORE = 0.970
GIT_EXECUTABLE = "/usr/bin/git"

DEFAULT_COHORT_ROOT = Path(
    "/gpfs/data/karczmar-lab/vanguard/dce2d_internal_ultrafast_pretreatment_cohort_v2"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/gpfs/data/karczmar-lab/vanguard/"
    "dce2d_internal_ultrafast_pretreatment_cohort_v2_duplicate_audit_v1"
)
DEFAULT_LEGACY_DICOM = Path(
    "/gpfs/data/karczmar-lab/vanguard/dce2d_internal_ultrafast_manifest/"
    "paired_hr_ufast_source_dicom/dicom_file_manifest.parquet"
)
DEFAULT_ZHEN_DICOM = Path(
    "/gpfs/data/karczmar-lab/vanguard/"
    "uchicago_ultrafast_longitudinal_cohort_v1/_build/zhen_extension/"
    "selected_dicom_file_manifest.parquet"
)
DEFAULT_STAGING_DICOM = Path(
    "/gpfs/data/karczmar-lab/vanguard/"
    "uchicago_ultrafast_longitudinal_cohort_v1/_build/zhen_staging_extension/"
    "selected_dicom_file_manifest.parquet"
)


@dataclass(frozen=True)
class ExtractionResult:
    """Pseudonymous image fingerprints and comparison representations."""

    exam_id: str
    cohort_component: str
    is_positive_control: bool
    expected_partner_exam_id: str
    ufast_source_kind: str
    ufast_baseline_shape: tuple[int, int, int]
    ufast_late_shape: tuple[int, int, int]
    hr_baseline_shape: tuple[int, int, int]
    ufast_baseline_sha256: str
    ufast_late_sha256: str
    hr_baseline_sha256: str
    ufast_baseline_descriptor: np.ndarray
    ufast_enhancement_descriptor: np.ndarray
    hr_baseline_descriptor: np.ndarray
    ufast_baseline_qc: np.ndarray
    ufast_enhancement_qc: np.ndarray
    hr_baseline_qc: np.ndarray


def _clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _truthy(value: object) -> bool:
    return _clean(value).lower() in {"1", "true", "t", "yes", "y"}


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(array, dtype="<f4"))
    digest = hashlib.sha256()
    digest.update(json.dumps(list(canonical.shape), separators=(",", ":")).encode())
    digest.update(memoryview(canonical))
    return digest.hexdigest()


def _finite_volume(array: np.ndarray, *, label: str) -> np.ndarray:
    volume = np.asarray(array, dtype=np.float32)
    valid_shape = (
        volume.ndim == VOLUME_DIMENSIONS
        and volume.shape[0] >= MINIMUM_SLICE_COUNT
        and min(volume.shape[1:]) >= MINIMUM_IN_PLANE_SIZE
    )
    if not valid_shape:
        raise ValueError(f"{label} is not a valid 3D volume")
    if not np.isfinite(volume).all():
        raise ValueError(f"{label} contains non-finite values")
    return volume


def _load_nifti_zyx(path: str) -> np.ndarray:
    image = nib.as_closest_canonical(nib.load(path))
    xyz = _finite_volume(np.asanyarray(image.dataobj), label="NIfTI phase")
    return np.transpose(xyz, (2, 1, 0)).copy()


def _foreground_bbox(volume: np.ndarray) -> tuple[slice, slice, slice]:
    positive = volume[volume > 0]
    if positive.size < MINIMUM_FOREGROUND_VOXELS:
        return tuple(slice(0, size) for size in volume.shape)  # type: ignore[return-value]
    high = float(np.percentile(positive, 99.0))
    threshold = max(float(np.percentile(positive, 1.0)), high * 0.025)
    coordinates = np.argwhere(volume > threshold)
    if len(coordinates) < MINIMUM_FOREGROUND_VOXELS:
        return tuple(slice(0, size) for size in volume.shape)  # type: ignore[return-value]
    starts = coordinates.min(axis=0)
    stops = coordinates.max(axis=0) + 1
    spans = stops - starts
    padding = np.maximum(2, np.ceil(spans * 0.04).astype(int))
    starts = np.maximum(0, starts - padding)
    stops = np.minimum(np.asarray(volume.shape), stops + padding)
    return tuple(slice(int(start), int(stop)) for start, stop in zip(starts, stops))  # type: ignore[return-value]


def _normalization_limits(volume: np.ndarray) -> tuple[float, float]:
    positive = volume[volume > 0]
    values = (
        positive if positive.size >= MINIMUM_FOREGROUND_VOXELS else volume.reshape(-1)
    )
    low, high = (float(value) for value in np.percentile(values, (1.0, 99.5)))
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        low = float(np.min(values))
        high = float(np.max(values))
    if high <= low:
        high = low + 1.0
    return low, high


def _descriptor_and_qc(
    volume: np.ndarray,
    *,
    bbox: tuple[slice, slice, slice] | None = None,
    limits: tuple[float, float] | None = None,
    signed: bool = False,
) -> tuple[np.ndarray, np.ndarray, tuple[slice, slice, slice], tuple[float, float]]:
    volume = _finite_volume(volume, label="descriptor input")
    chosen_bbox = bbox if bbox is not None else _foreground_bbox(volume)
    cropped = volume[chosen_bbox]
    low, high = limits if limits is not None else _normalization_limits(cropped)
    scale = max(high - low, np.finfo(np.float32).eps)
    if signed:
        normalized = np.clip(cropped / scale, -2.0, 2.0)
    else:
        normalized = np.clip((cropped - low) / scale, 0.0, 1.0)
    descriptor = resize(
        normalized,
        DESCRIPTOR_SHAPE_ZYX,
        order=1,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float32)
    slices: list[np.ndarray] = []
    for fraction in QC_Z_FRACTIONS:
        index = min(
            normalized.shape[0] - 1,
            max(0, int(round((normalized.shape[0] - 1) * fraction))),
        )
        slices.append(
            resize(
                cropped[index],
                (QC_SLICE_SIZE, QC_SLICE_SIZE),
                order=1,
                mode="reflect",
                anti_aliasing=True,
                preserve_range=True,
            ).astype(np.float32)
        )
    qc = np.concatenate(slices, axis=1)
    return descriptor, qc, chosen_bbox, (low, high)


def _dicom_sort_order(datasets: list[Any]) -> list[int]:
    if not datasets:
        raise ValueError("empty DICOM phase")
    try:
        orientation = np.asarray(datasets[0].ImageOrientationPatient, dtype=float)
        row_direction = orientation[:3]
        column_direction = orientation[3:]
        normal = np.cross(row_direction, column_direction)
        positions = np.stack(
            [
                np.asarray(dataset.ImagePositionPatient, dtype=float)
                for dataset in datasets
            ]
        )
        projections = positions @ normal
        if np.ptp(projections) > 0:
            return np.argsort(projections, kind="stable").tolist()
    except (AttributeError, TypeError, ValueError):
        pass
    instance_numbers = [
        float(getattr(dataset, "InstanceNumber", index))
        for index, dataset in enumerate(datasets)
    ]
    return np.argsort(instance_numbers, kind="stable").tolist()


def _stack_dicom_datasets(datasets: list[Any]) -> np.ndarray:
    order = _dicom_sort_order(datasets)
    frames: list[np.ndarray] = []
    for index in order:
        pixels = np.asarray(datasets[index].pixel_array)
        if pixels.ndim != SLICE_DIMENSIONS:
            raise ValueError("selected DICOM instance is not single-frame 2D")
        frames.append(pixels)
    shapes = {frame.shape for frame in frames}
    if len(shapes) != 1:
        raise ValueError("DICOM slices do not share one pixel shape")
    return _finite_volume(np.stack(frames).astype(np.float32), label="DICOM phase")


def _load_inventory_phase(spec: dict[str, Any]) -> np.ndarray:
    datasets: list[Any] = []
    if spec["source_kind"] == "zip":
        with zipfile.ZipFile(spec["archive_path"]) as archive:
            for member in spec["locations"]:
                datasets.append(
                    pydicom.dcmread(io.BytesIO(archive.read(member)), force=True)
                )
    elif spec["source_kind"] == "filesystem":
        for source_path in spec["locations"]:
            datasets.append(pydicom.dcmread(source_path, force=True))
    else:
        raise ValueError("unsupported instance inventory source kind")
    return _stack_dicom_datasets(datasets)


def _retro_series_headers(spec: dict[str, Any]) -> list[dict[str, Any]]:
    representative = Path(spec["representative_path"])
    files = sorted(path for path in representative.parent.iterdir() if path.is_file())
    rows: list[dict[str, Any]] = []
    matched = 0
    tags = [
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "TemporalPositionIdentifier",
        "InstanceNumber",
        "ImageOrientationPatient",
        "ImagePositionPatient",
        "Rows",
        "Columns",
    ]
    for source_path in files:
        dataset = pydicom.dcmread(
            source_path,
            stop_before_pixels=True,
            force=True,
            specific_tags=tags,
        )
        if (
            str(getattr(dataset, "StudyInstanceUID", "")) != spec["study_uid"]
            or str(getattr(dataset, "SeriesInstanceUID", "")) != spec["series_uid"]
        ):
            continue
        matched += 1
        temporal = getattr(dataset, "TemporalPositionIdentifier", None)
        if temporal is None:
            continue
        rows.append(
            {
                "path": str(source_path),
                "temporal": float(temporal),
                "instance": float(getattr(dataset, "InstanceNumber", len(rows))),
            }
        )
    if matched != int(spec["expected_instances"]):
        raise ValueError("raw series file count does not match frozen inventory")
    if not rows:
        raise ValueError("raw series has no temporally indexed image instances")
    return rows


def _load_retro_phases(
    spec: dict[str, Any], *, include_late: bool
) -> tuple[np.ndarray, np.ndarray | None]:
    rows = _retro_series_headers(spec)
    temporal_positions = sorted({row["temporal"] for row in rows})
    selected_positions = [temporal_positions[0]]
    if include_late:
        selected_positions.append(temporal_positions[-1])
    volumes: list[np.ndarray] = []
    for temporal in selected_positions:
        phase_rows = [row for row in rows if row["temporal"] == temporal]
        phase_rows.sort(key=lambda row: row["instance"])
        datasets = [pydicom.dcmread(row["path"], force=True) for row in phase_rows]
        volumes.append(_stack_dicom_datasets(datasets))
    if include_late:
        return volumes[0], volumes[1]
    return volumes[0], None


def _extract_record(record: dict[str, Any]) -> ExtractionResult:
    if record["ufast_source_kind"] == "nifti":
        phase_files = record["phase_files"]
        baseline = _load_nifti_zyx(phase_files[0])
        late = _load_nifti_zyx(phase_files[-1])
    elif record["ufast_source_kind"] == "raw_retro_dicom":
        baseline, late_optional = _load_retro_phases(
            record["ufast_spec"], include_late=True
        )
        if late_optional is None:
            raise ValueError("late UFAST phase was not reconstructed")
        late = late_optional
    else:
        raise ValueError("unsupported UFAST source kind")
    if baseline.shape != late.shape:
        raise ValueError("UFAST baseline and late phase shapes differ")

    if record["hr_spec"]["kind"] == "inventory":
        hr = _load_inventory_phase(record["hr_spec"])
    elif record["hr_spec"]["kind"] == "raw_retro_dicom":
        hr, _ = _load_retro_phases(record["hr_spec"], include_late=False)
    else:
        raise ValueError("unsupported HR source kind")

    baseline_descriptor, baseline_qc, bbox, limits = _descriptor_and_qc(baseline)
    enhancement = late - baseline
    enhancement_descriptor, enhancement_qc, _, _ = _descriptor_and_qc(
        enhancement,
        bbox=bbox,
        limits=(0.0, limits[1] - limits[0]),
        signed=True,
    )
    hr_descriptor, hr_qc, _, _ = _descriptor_and_qc(hr)
    return ExtractionResult(
        exam_id=record["exam_id"],
        cohort_component=record["cohort_component"],
        is_positive_control=bool(record["is_positive_control"]),
        expected_partner_exam_id=record["expected_partner_exam_id"],
        ufast_source_kind=record["ufast_source_kind"],
        ufast_baseline_shape=tuple(int(value) for value in baseline.shape),
        ufast_late_shape=tuple(int(value) for value in late.shape),
        hr_baseline_shape=tuple(int(value) for value in hr.shape),
        ufast_baseline_sha256=_array_sha256(baseline),
        ufast_late_sha256=_array_sha256(late),
        hr_baseline_sha256=_array_sha256(hr),
        ufast_baseline_descriptor=baseline_descriptor,
        ufast_enhancement_descriptor=enhancement_descriptor,
        hr_baseline_descriptor=hr_descriptor,
        ufast_baseline_qc=baseline_qc,
        ufast_enhancement_qc=enhancement_qc,
        hr_baseline_qc=hr_qc,
    )


def _prepare_inventory_specs(
    records: list[dict[str, Any]], inventory_paths: list[Path]
) -> None:
    targets = {
        (record["study_uid"], record["hr_series_uid"]): record
        for record in records
        if record["source_category"] == "canonical"
    }
    found: set[tuple[str, str]] = set()
    required_columns = [
        "study_instance_uid",
        "series_instance_uid",
        "read_ok",
        "temporal_position_identifier",
        "instance_number",
        "archive_path",
        "archive_member",
        "source_path",
        "sop_instance_uid",
    ]
    for inventory_path in inventory_paths:
        available = set(pd.read_parquet(inventory_path).columns)
        columns = [column for column in required_columns if column in available]
        frame = pd.read_parquet(inventory_path, columns=columns)
        frame["study_instance_uid"] = frame["study_instance_uid"].astype(str)
        frame["series_instance_uid"] = frame["series_instance_uid"].astype(str)
        mask = [
            (study, series) in targets
            for study, series in zip(
                frame["study_instance_uid"], frame["series_instance_uid"], strict=True
            )
        ]
        frame = frame.loc[mask].copy()
        if frame.empty:
            continue
        for key, block in frame.groupby(
            ["study_instance_uid", "series_instance_uid"], sort=False
        ):
            if key in found:
                continue
            readable = block["read_ok"].fillna(False).isin((True, 1, "true", "True"))
            block = block.loc[readable].copy()
            block["temporal_position_identifier"] = pd.to_numeric(
                block["temporal_position_identifier"], errors="coerce"
            )
            block["instance_number"] = pd.to_numeric(
                block["instance_number"], errors="coerce"
            )
            block = block.dropna(
                subset=["temporal_position_identifier", "instance_number"]
            )
            location_columns = [
                column
                for column in ("archive_path", "archive_member", "source_path")
                if column in block
            ]
            dedup_columns = [
                column
                for column in ("sop_instance_uid", *location_columns)
                if column in block
            ]
            block = block.drop_duplicates(dedup_columns, keep="first")
            baseline_position = block["temporal_position_identifier"].min()
            phase = block.loc[
                block["temporal_position_identifier"].eq(baseline_position)
            ].sort_values("instance_number", kind="stable")
            record = targets[key]
            expected_text = _clean(record["hr_baseline_frame_count"])
            expected = int(float(expected_text)) if expected_text else None
            if expected is not None and len(phase) != expected:
                raise ValueError(
                    f"canonical HR baseline count mismatch for {record['exam_id']}"
                )
            archive_rows = (
                phase["archive_path"].fillna("").astype(str).str.strip().ne("")
                if "archive_path" in phase
                else pd.Series(False, index=phase.index)
            )
            file_rows = (
                phase["source_path"].fillna("").astype(str).str.strip().ne("")
                if "source_path" in phase
                else pd.Series(False, index=phase.index)
            )
            if archive_rows.all() and not file_rows.any():
                archives = sorted(set(phase["archive_path"].astype(str)))
                if len(archives) != 1:
                    raise ValueError(
                        f"canonical HR spans archives for {record['exam_id']}"
                    )
                spec = {
                    "kind": "inventory",
                    "source_kind": "zip",
                    "archive_path": archives[0],
                    "locations": phase["archive_member"].astype(str).tolist(),
                }
            elif file_rows.all() and not archive_rows.any():
                spec = {
                    "kind": "inventory",
                    "source_kind": "filesystem",
                    "locations": phase["source_path"].astype(str).tolist(),
                }
            else:
                raise ValueError(
                    f"canonical HR has mixed source kinds for {record['exam_id']}"
                )
            record["hr_spec"] = spec
            found.add(key)
    missing = set(targets).difference(found)
    if missing:
        missing_exam_ids = sorted(targets[key]["exam_id"] for key in missing)
        raise ValueError(
            "canonical HR inventory coverage incomplete for pseudonymous exams: "
            + ",".join(missing_exam_ids)
        )


def _retro_spec(
    series_lookup: pd.DataFrame,
    *,
    study_uid: str,
    series_uid: str,
    kind: str,
) -> dict[str, Any]:
    row = series_lookup.loc[series_uid]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    if _clean(row["study_instance_uid"]) != study_uid:
        raise ValueError("Retro series does not belong to its selected study")
    return {
        "kind": kind,
        "representative_path": _clean(row["source_path"]),
        "study_uid": study_uid,
        "series_uid": series_uid,
        "expected_instances": int(row["n_instances"]),
    }


def _build_records(cohort_root: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    eligible_path = cohort_root / "source_eligible_cohort_manifest.csv"
    pair_path = cohort_root / "paired_source_manifest.csv"
    exclusion_path = cohort_root / "retro_patient_deduplication_exclusions.csv"
    snapshot_root = cohort_root / "_build" / "input_snapshots"
    retro_inventory_path = snapshot_root / "retro_dicom_series_inventory.parquet"
    retro_gate_path = snapshot_root / "retro_gate_readiness.csv"
    input_paths = [
        eligible_path,
        pair_path,
        exclusion_path,
        retro_inventory_path,
        retro_gate_path,
    ]
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    eligible = pd.read_csv(eligible_path, dtype=str).fillna("")
    pairs = pd.read_csv(pair_path, dtype=str).fillna("")
    exclusions = pd.read_csv(exclusion_path, dtype=str).fillna("")
    gate = pd.read_csv(retro_gate_path, dtype=str).fillna("")
    if (
        len(eligible) != EXPECTED_SELECTED_EXAMS
        or len(pairs) != EXPECTED_SELECTED_EXAMS
    ):
        raise ValueError("expected the frozen 341-exam source-eligible cohort")
    if (
        eligible["exam_id"].duplicated().any()
        or eligible["patient_key"].duplicated().any()
    ):
        raise ValueError("selected cohort pseudonymous identities are not unique")
    pair_columns = [
        "study_instance_uid",
        "ufast_series_instance_uid",
        "ufast_baseline_frame_count",
        "hr_series_instance_uid",
        "hr_precontrast_series_instance_uid",
        "hr_baseline_frame_count",
        "hr_partner_layout",
        "cohort_component",
    ]
    selected = eligible.merge(
        pairs[pair_columns],
        on="study_instance_uid",
        how="left",
        validate="one_to_one",
        suffixes=("", "_pair"),
    )
    records: list[dict[str, Any]] = []
    for row in selected.to_dict(orient="records"):
        phase_files = json.loads(_clean(row["phase_files"]) or "[]")
        source_category = (
            "retro"
            if _clean(row["cohort_component_pair"]) == "retro_caps"
            else "canonical"
        )
        hr_pre = _clean(row["hr_precontrast_series_instance_uid"])
        records.append(
            {
                "exam_id": _clean(row["exam_id"]),
                "patient_key": _clean(row["patient_key"]),
                "cohort_component": _clean(row["cohort_component"]),
                "source_category": source_category,
                "study_uid": _clean(row["study_instance_uid"]),
                "ufast_series_uid": _clean(row["ufast_series_instance_uid"]),
                "ufast_baseline_frame_count": _clean(row["ufast_baseline_frame_count"]),
                "hr_series_uid": hr_pre or _clean(row["hr_series_instance_uid"]),
                "hr_baseline_frame_count": _clean(row["hr_baseline_frame_count"]),
                "phase_files": phase_files,
                "ufast_source_kind": "nifti" if phase_files else "raw_retro_dicom",
                "is_positive_control": False,
                "expected_partner_exam_id": "",
            }
        )

    selected_by_patient = {
        record["patient_key"]: record["exam_id"] for record in records
    }
    control_gate = exclusions.merge(
        gate[
            [
                "study_instance_uid",
                "ultrafast_series_instance_uid",
                "high_resolution_series_instance_uid",
                "high_resolution_precontrast_series_instance_uid",
                "high_resolution_baseline_frame_count",
            ]
        ],
        on="study_instance_uid",
        how="left",
        validate="one_to_one",
    )
    if (
        len(control_gate) != EXPECTED_POSITIVE_CONTROLS
        or control_gate["ultrafast_series_instance_uid"].eq("").any()
    ):
        raise ValueError("expected four resolvable technical duplicate controls")
    for row in control_gate.to_dict(orient="records"):
        patient_key = _clean(row["patient_key"])
        hr_pre = _clean(row["high_resolution_precontrast_series_instance_uid"])
        records.append(
            {
                "exam_id": _clean(row["exam_id"]),
                "patient_key": patient_key,
                "cohort_component": "retro_technical_reexport_control",
                "source_category": "retro",
                "study_uid": _clean(row["study_instance_uid"]),
                "ufast_series_uid": _clean(row["ultrafast_series_instance_uid"]),
                "ufast_baseline_frame_count": "",
                "hr_series_uid": hr_pre
                or _clean(row["high_resolution_series_instance_uid"]),
                "hr_baseline_frame_count": _clean(
                    row["high_resolution_baseline_frame_count"]
                ),
                "phase_files": [],
                "ufast_source_kind": "raw_retro_dicom",
                "is_positive_control": True,
                "expected_partner_exam_id": selected_by_patient[patient_key],
            }
        )

    series = pd.read_parquet(retro_inventory_path)
    series["series_instance_uid"] = series["series_instance_uid"].astype(str)
    series_lookup = series.set_index("series_instance_uid", drop=False)
    for record in records:
        if record["source_category"] != "retro":
            continue
        if record["ufast_source_kind"] == "raw_retro_dicom":
            record["ufast_spec"] = _retro_spec(
                series_lookup,
                study_uid=record["study_uid"],
                series_uid=record["ufast_series_uid"],
                kind="raw_retro_dicom",
            )
        record["hr_spec"] = _retro_spec(
            series_lookup,
            study_uid=record["study_uid"],
            series_uid=record["hr_series_uid"],
            kind="raw_retro_dicom",
        )
    return records, input_paths


def _normalized_rows(descriptors: np.ndarray) -> np.ndarray:
    flat = descriptors.reshape(len(descriptors), -1).astype(np.float32)
    flat -= flat.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(flat, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(np.float32).eps):
        raise ValueError("a comparison descriptor has zero variance")
    return flat / norms


def _flip_invariant_correlations(descriptors: np.ndarray) -> np.ndarray:
    reference = _normalized_rows(descriptors)
    scores = np.full((len(descriptors), len(descriptors)), -np.inf, dtype=np.float32)
    for flip_z in (False, True):
        for flip_y in (False, True):
            for flip_x in (False, True):
                axes = tuple(
                    axis
                    for axis, enabled in enumerate((flip_z, flip_y, flip_x), start=1)
                    if enabled
                )
                candidate = np.flip(descriptors, axis=axes) if axes else descriptors
                candidate_rows = _normalized_rows(candidate)
                scores = np.maximum(scores, reference @ candidate_rows.T)
    scores = np.maximum(scores, scores.T)
    np.fill_diagonal(scores, 1.0)
    return scores


def _pair_table(results: list[ExtractionResult]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ufast = np.stack([result.ufast_baseline_descriptor for result in results])
    enhancement = np.stack([result.ufast_enhancement_descriptor for result in results])
    hr = np.stack([result.hr_baseline_descriptor for result in results])
    ufast_scores = _flip_invariant_correlations(ufast)
    enhancement_scores = _flip_invariant_correlations(enhancement)
    hr_scores = _flip_invariant_correlations(hr)
    combined = 0.50 * ufast_scores + 0.20 * enhancement_scores + 0.30 * hr_scores

    selected_indices = [
        index for index, result in enumerate(results) if not result.is_positive_control
    ]
    rows: list[dict[str, Any]] = []
    for offset, first in enumerate(selected_indices):
        for second in selected_indices[offset + 1 :]:
            left = results[first]
            right = results[second]
            exact_ufast = (
                left.ufast_baseline_sha256 == right.ufast_baseline_sha256
                and left.ufast_late_sha256 == right.ufast_late_sha256
            )
            exact_hr = left.hr_baseline_sha256 == right.hr_baseline_sha256
            ufast_value = float(ufast_scores[first, second])
            enhancement_value = float(enhancement_scores[first, second])
            hr_value = float(hr_scores[first, second])
            combined_value = float(combined[first, second])
            rows.append(
                {
                    "exam_id_a": left.exam_id,
                    "exam_id_b": right.exam_id,
                    "cohort_component_a": left.cohort_component,
                    "cohort_component_b": right.cohort_component,
                    "ufast_baseline_correlation": ufast_value,
                    "ufast_enhancement_correlation": enhancement_value,
                    "hr_baseline_correlation": hr_value,
                    "combined_image_score": combined_value,
                    "minimum_anatomy_correlation": min(ufast_value, hr_value),
                    "exact_ufast_phase_pair": exact_ufast,
                    "exact_hr_baseline": exact_hr,
                    "exact_exam_pixel_match": exact_ufast and exact_hr,
                    "automatic_probable_duplicate": (
                        ufast_value >= PROBABLE_ANATOMY_CORRELATION
                        and hr_value >= PROBABLE_ANATOMY_CORRELATION
                        and enhancement_value >= PROBABLE_ENHANCEMENT_CORRELATION
                    ),
                    "automatic_review_candidate": (
                        (
                            ufast_value >= REVIEW_ANATOMY_CORRELATION
                            and hr_value >= REVIEW_ANATOMY_CORRELATION
                        )
                        or combined_value >= REVIEW_COMBINED_SCORE
                    ),
                }
            )
    pairs = pd.DataFrame(rows).sort_values(
        ["combined_image_score", "minimum_anatomy_correlation"],
        ascending=False,
        kind="stable",
    )

    index_by_exam = {result.exam_id: index for index, result in enumerate(results)}
    control_rows: list[dict[str, Any]] = []
    for control_index, control in enumerate(results):
        if not control.is_positive_control:
            continue
        partner_index = index_by_exam[control.expected_partner_exam_id]
        selected_order = sorted(
            selected_indices,
            key=lambda index: float(combined[control_index, index]),
            reverse=True,
        )
        partner_rank = selected_order.index(partner_index) + 1
        partner = results[partner_index]
        control_rows.append(
            {
                "control_exam_id": control.exam_id,
                "expected_partner_exam_id": partner.exam_id,
                "expected_partner_rank_among_341": partner_rank,
                "ufast_baseline_correlation": float(
                    ufast_scores[control_index, partner_index]
                ),
                "ufast_enhancement_correlation": float(
                    enhancement_scores[control_index, partner_index]
                ),
                "hr_baseline_correlation": float(
                    hr_scores[control_index, partner_index]
                ),
                "combined_image_score": float(combined[control_index, partner_index]),
                "exact_ufast_phase_pair": (
                    control.ufast_baseline_sha256 == partner.ufast_baseline_sha256
                    and control.ufast_late_sha256 == partner.ufast_late_sha256
                ),
                "exact_hr_baseline": (
                    control.hr_baseline_sha256 == partner.hr_baseline_sha256
                ),
            }
        )
    return pairs, pd.DataFrame(control_rows)


def _shared_limits(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([first.reshape(-1), second.reshape(-1)])
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = (float(value) for value in np.percentile(finite, (1.0, 99.0)))
    if high <= low:
        high = low + 1.0
    return low, high


def _write_pair_panels(
    pairs: pd.DataFrame,
    results: list[ExtractionResult],
    output_dir: Path,
    *,
    maximum: int,
    prefix: str,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=False)
    lookup = {result.exam_id: result for result in results}
    paths: list[str] = []
    for rank, row in enumerate(pairs.head(maximum).itertuples(index=False), start=1):
        if hasattr(row, "exam_id_a"):
            left_id, right_id = row.exam_id_a, row.exam_id_b
        else:
            left_id, right_id = row.control_exam_id, row.expected_partner_exam_id
        left, right = lookup[left_id], lookup[right_id]
        figure, axes = plt.subplots(2, 3, figsize=(13.5, 4.8), constrained_layout=True)
        modalities = [
            (left.ufast_baseline_qc, right.ufast_baseline_qc, "UFAST baseline", False),
            (
                left.ufast_enhancement_qc,
                right.ufast_enhancement_qc,
                "UFAST late - baseline",
                True,
            ),
            (left.hr_baseline_qc, right.hr_baseline_qc, "HR baseline", False),
        ]
        for column, (left_image, right_image, title, signed) in enumerate(modalities):
            if signed:
                absolute = float(
                    np.percentile(
                        np.abs(
                            np.concatenate([left_image.ravel(), right_image.ravel()])
                        ),
                        99.0,
                    )
                )
                absolute = max(absolute, np.finfo(np.float32).eps)
                limits = (-absolute, absolute)
                cmap = "coolwarm"
            else:
                limits = _shared_limits(left_image, right_image)
                cmap = "gray"
            axes[0, column].imshow(
                left_image, cmap=cmap, vmin=limits[0], vmax=limits[1]
            )
            axes[1, column].imshow(
                right_image, cmap=cmap, vmin=limits[0], vmax=limits[1]
            )
            axes[0, column].set_title(title)
            axes[0, column].set_ylabel(left_id, fontsize=7)
            axes[1, column].set_ylabel(right_id, fontsize=7)
            axes[0, column].set_xticks([])
            axes[0, column].set_yticks([])
            axes[1, column].set_xticks([])
            axes[1, column].set_yticks([])
        figure.suptitle(
            f"rank {rank}: combined={float(row.combined_image_score):.4f}, "
            f"UFAST={float(row.ufast_baseline_correlation):.4f}, "
            f"enh={float(row.ufast_enhancement_correlation):.4f}, "
            f"HR={float(row.hr_baseline_correlation):.4f}",
            fontsize=10,
        )
        path = output_dir / f"{prefix}_{rank:03d}.png"
        figure.savefig(path, dpi=130)
        plt.close(figure)
        paths.append(str(path.name))
    return paths


def _git_state(repo_root: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        return subprocess.run(  # noqa: S603 - fixed executable and trusted arguments
            [GIT_EXECUTABLE, *arguments],
            cwd=repo_root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _write_checksums(root: Path) -> None:
    files = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS"
    )
    lines = [f"{_sha256_file(path)}  {path.relative_to(root)}" for path in files]
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _set_protected_permissions(root: Path) -> None:
    for path in root.rglob("*"):
        path.chmod(0o2770 if path.is_dir() else 0o660)
    root.chmod(0o2770)


def _publish(
    *,
    results: list[ExtractionResult],
    pair_scores: pd.DataFrame,
    controls: pd.DataFrame,
    stage: Path,
    output_root: Path,
    input_paths: list[Path],
    arguments: argparse.Namespace,
) -> dict[str, Any]:
    stage.mkdir(parents=True, exist_ok=False)
    qc_root = stage / "qc"
    selected_panel_paths = _write_pair_panels(
        pair_scores,
        results,
        qc_root / "top_selected_pairs",
        maximum=arguments.top_pairs,
        prefix="selected_pair",
    )
    control_panel_paths = _write_pair_panels(
        controls,
        results,
        qc_root / "positive_controls",
        maximum=len(controls),
        prefix="positive_control",
    )

    fingerprint_rows = []
    for result in results:
        fingerprint_rows.append(
            {
                "exam_id": result.exam_id,
                "cohort_component": result.cohort_component,
                "is_positive_control": result.is_positive_control,
                "expected_partner_exam_id": result.expected_partner_exam_id,
                "ufast_source_kind": result.ufast_source_kind,
                "ufast_baseline_shape_zyx": json.dumps(result.ufast_baseline_shape),
                "ufast_late_shape_zyx": json.dumps(result.ufast_late_shape),
                "hr_baseline_shape_zyx": json.dumps(result.hr_baseline_shape),
                "ufast_baseline_sha256": result.ufast_baseline_sha256,
                "ufast_late_sha256": result.ufast_late_sha256,
                "hr_baseline_sha256": result.hr_baseline_sha256,
            }
        )
    pd.DataFrame(fingerprint_rows).to_csv(
        stage / "exam_image_fingerprints.csv", index=False
    )
    pair_scores.to_csv(stage / "all_selected_pair_scores.csv", index=False)
    pair_scores.head(arguments.top_pairs).to_csv(
        stage / "top_selected_pair_scores.csv", index=False
    )
    controls.to_csv(stage / "positive_control_scores.csv", index=False)
    np.savez_compressed(
        stage / "comparison_representations.npz",
        exam_ids=np.asarray([result.exam_id for result in results]),
        ufast_baseline=np.stack(
            [result.ufast_baseline_descriptor for result in results]
        ),
        ufast_enhancement=np.stack(
            [result.ufast_enhancement_descriptor for result in results]
        ),
        hr_baseline=np.stack([result.hr_baseline_descriptor for result in results]),
    )

    selected_pairs = pair_scores
    exact = selected_pairs["exact_exam_pixel_match"].map(_truthy)
    probable = selected_pairs["automatic_probable_duplicate"].map(_truthy)
    review = selected_pairs["automatic_review_candidate"].map(_truthy)
    control_rank_one = controls["expected_partner_rank_among_341"].astype(int).eq(1)
    summary = {
        "schema": SCHEMA,
        "selected_exams": int(
            sum(not result.is_positive_control for result in results)
        ),
        "positive_control_exams": int(
            sum(result.is_positive_control for result in results)
        ),
        "selected_pair_comparisons": int(len(selected_pairs)),
        "selected_ready_ufast_exports": int(
            sum(
                not result.is_positive_control and result.ufast_source_kind == "nifti"
                for result in results
            )
        ),
        "selected_raw_ufast_reconstructions": int(
            sum(
                not result.is_positive_control
                and result.ufast_source_kind == "raw_retro_dicom"
                for result in results
            )
        ),
        "selected_hr_raw_reconstructions": int(
            sum(not result.is_positive_control for result in results)
        ),
        "exact_selected_exam_pixel_matches": int(exact.sum()),
        "automatic_probable_duplicate_pairs": int(probable.sum()),
        "automatic_review_candidate_pairs": int(review.sum()),
        "maximum_selected_combined_image_score": float(
            selected_pairs.iloc[0]["combined_image_score"]
        ),
        "maximum_selected_ufast_baseline_correlation": float(
            selected_pairs["ufast_baseline_correlation"].max()
        ),
        "maximum_selected_hr_baseline_correlation": float(
            selected_pairs["hr_baseline_correlation"].max()
        ),
        "positive_controls_expected_partner_rank_one": int(control_rank_one.sum()),
        "positive_controls_total": int(len(controls)),
        "positive_control_minimum_combined_image_score": float(
            controls["combined_image_score"].min()
        ),
        "preliminary_status": (
            "exact_duplicate_detected"
            if exact.any()
            else "probable_duplicate_requires_review"
            if probable.any()
            else "no_duplicate_signal_review_top_pairs"
        ),
        "selected_qc_panels": selected_panel_paths,
        "positive_control_qc_panels": control_panel_paths,
    }
    (stage / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    provenance = {
        "schema": SCHEMA,
        "command": [sys.executable, *sys.argv],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "repo": _git_state(arguments.repo_root.resolve()),
        "inputs": {
            str(path): {"sha256": _sha256_file(path), "size_bytes": path.stat().st_size}
            for path in input_paths
        },
        "canonical_instance_inventories": [
            str(arguments.legacy_dicom_inventory.resolve()),
            str(arguments.zhen_dicom_inventory.resolve()),
            str(arguments.staging_dicom_inventory.resolve()),
        ],
        "method": {
            "ufast": (
                "first/last released NIfTI phases when available; otherwise first/last "
                "temporally indexed raw DICOM phases"
            ),
            "hr": "first temporally indexed raw DICOM phase for every exam",
            "exact_match": "SHA-256 over shape plus decompressed float32 pixel array",
            "robust_match": (
                "flip-invariant correlations of whole-volume low-resolution UFAST baseline, "
                "UFAST enhancement, and HR baseline descriptors"
            ),
            "combined_score_weights": {
                "ufast_baseline": 0.50,
                "ufast_enhancement": 0.20,
                "hr_baseline": 0.30,
            },
            "descriptor_shape_zyx": list(DESCRIPTOR_SHAPE_ZYX),
            "positive_controls": (
                "four same-patient/same-date/same-description technical re-export exams "
                "excluded before cohort release"
            ),
        },
    }
    (stage / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    readme = f"""# Sarit pCR exam duplicate audit

This protected audit compares image pixels for all {summary["selected_exams"]} source-eligible
exams in the expanded Sarit pCR cohort. It does not use patient, StudyInstanceUID, or
SeriesInstanceUID equality to decide whether exams are duplicates.

## Surfaces checked

- UFAST baseline and late-phase pixels for every selected exam.
- HR baseline pixels for every selected exam.
- Exact decompressed-pixel SHA-256 fingerprints.
- Flip-invariant whole-volume image correlations for exact re-exports that received new
  identifiers or passed through a different export path.
- Four known same-patient technical re-export exclusions as sensitivity controls.

The QC panels use one shared intensity window for both exams in each modality and one shared
symmetric window for both enhancement images. The automatic status is preliminary until the
ranked panels have been visually inspected.

## Preliminary result

- exact selected exam pixel matches: {summary["exact_selected_exam_pixel_matches"]}
- automatic probable duplicate pairs: {summary["automatic_probable_duplicate_pairs"]}
- automatic review candidates: {summary["automatic_review_candidate_pairs"]}
- positive controls whose expected retained partner ranked first: {summary["positive_controls_expected_partner_rank_one"]}/{summary["positive_controls_total"]}
- status: `{summary["preliminary_status"]}`

See `validation_summary.json`, `top_selected_pair_scores.csv`, `positive_control_scores.csv`,
and `qc/`. Raw identifiers are intentionally absent from the audit tables and panels.
"""
    (stage / "README.md").write_text(readme, encoding="utf-8")
    _write_checksums(stage)
    _set_protected_permissions(stage)
    stage.rename(output_root)
    return summary


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-root", type=Path, default=DEFAULT_COHORT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--legacy-dicom-inventory", type=Path, default=DEFAULT_LEGACY_DICOM
    )
    parser.add_argument("--zhen-dicom-inventory", type=Path, default=DEFAULT_ZHEN_DICOM)
    parser.add_argument(
        "--staging-dicom-inventory", type=Path, default=DEFAULT_STAGING_DICOM
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--top-pairs", type=int, default=25)
    parser.add_argument(
        "--smoke-exam-id",
        default="",
        help="Extract one pseudonymous exam and exit without publishing outputs.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the protected image-content duplicate audit."""
    arguments = parse_arguments()
    os.umask(0o007)
    output_root = arguments.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(
            f"refusing to overwrite existing audit root: {output_root}"
        )
    job_token = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    stage = output_root.parent / f".{output_root.name}.staging-{job_token}"
    if stage.exists():
        raise FileExistsError(f"staging directory already exists: {stage}")

    records, input_paths = _build_records(arguments.cohort_root.expanduser().resolve())
    inventory_paths = [
        arguments.legacy_dicom_inventory.expanduser().resolve(),
        arguments.zhen_dicom_inventory.expanduser().resolve(),
        arguments.staging_dicom_inventory.expanduser().resolve(),
    ]
    for path in inventory_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    input_paths.extend(inventory_paths)
    _prepare_inventory_specs(records, inventory_paths)
    if arguments.smoke_exam_id:
        selected = [
            record for record in records if record["exam_id"] == arguments.smoke_exam_id
        ]
        if len(selected) != 1:
            raise ValueError("smoke exam ID does not resolve uniquely")
        print(f"[audit] smoke extraction exam={arguments.smoke_exam_id}", flush=True)
        result = _extract_record(selected[0])
        print(
            "[audit] smoke complete "
            f"ufast_baseline_shape={result.ufast_baseline_shape} "
            f"ufast_late_shape={result.ufast_late_shape} "
            f"hr_baseline_shape={result.hr_baseline_shape}",
            flush=True,
        )
        return
    print(
        "[audit] prepared "
        f"selected={sum(not record['is_positive_control'] for record in records)} "
        f"controls={sum(record['is_positive_control'] for record in records)} "
        f"workers={arguments.workers}",
        flush=True,
    )

    results_by_exam: dict[str, ExtractionResult] = {}
    failures: list[tuple[str, str]] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=arguments.workers
    ) as executor:
        future_to_exam = {
            executor.submit(_extract_record, record): record["exam_id"]
            for record in records
        }
        for completed, future in enumerate(
            concurrent.futures.as_completed(future_to_exam), start=1
        ):
            exam_id = future_to_exam[future]
            try:
                result = future.result()
            except Exception as error:  # noqa: BLE001
                failures.append((exam_id, type(error).__name__))
                print(
                    f"[audit] extraction_failure exam={exam_id} "
                    f"kind={type(error).__name__}",
                    flush=True,
                )
            else:
                results_by_exam[exam_id] = result
            if completed % 10 == 0 or completed == len(records):
                print(
                    f"[audit] extracted={completed}/{len(records)} "
                    f"failures={len(failures)}",
                    flush=True,
                )
    if failures:
        safe_failures = ", ".join(f"{exam_id}:{kind}" for exam_id, kind in failures)
        raise RuntimeError(f"image extraction failures: {safe_failures}")
    results = [results_by_exam[record["exam_id"]] for record in records]

    print("[audit] computing all-pairs image similarities", flush=True)
    pair_scores, controls = _pair_table(results)
    summary = _publish(
        results=results,
        pair_scores=pair_scores,
        controls=controls,
        stage=stage,
        output_root=output_root,
        input_paths=input_paths,
        arguments=arguments,
    )
    print(
        "[audit] complete "
        f"output={output_root} "
        f"exact={summary['exact_selected_exam_pixel_matches']} "
        f"probable={summary['automatic_probable_duplicate_pairs']} "
        f"review={summary['automatic_review_candidate_pairs']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
