"""Strict DICOM loading and physical geometry for Vanguard preprocessing."""

from __future__ import annotations

import hashlib
import io
import os
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

PHYSICAL_TOLERANCE_MM = 1e-3
SECONDS_PER_DAY = 24.0 * 3600.0
DICOM_TM_MINIMUM_CHARACTERS = 6


@dataclass(frozen=True)
class DicomGeometry:
    """Physical geometry shared by every phase of one DICOM series."""

    series_instance_uid: str
    frame_of_reference_uid: str
    shape_zyx: tuple[int, int, int]
    spacing_xyz_mm: tuple[float, float, float]
    origin_lps_mm: tuple[float, float, float]
    direction_lps: tuple[float, ...]
    slice_thickness_mm: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation."""
        payload = asdict(self)
        payload["shape_zyx"] = list(self.shape_zyx)
        payload["spacing_xyz_mm"] = list(self.spacing_xyz_mm)
        payload["origin_lps_mm"] = list(self.origin_lps_mm)
        payload["direction_lps"] = list(self.direction_lps)
        payload["field_of_view_xyz_mm"] = (
            np.asarray(self.shape_zyx[::-1]) * np.asarray(self.spacing_xyz_mm)
        ).tolist()
        return payload

    @classmethod
    def from_dict(cls: type[DicomGeometry], payload: dict[str, Any]) -> DicomGeometry:
        """Reconstruct geometry from saved provenance."""
        return cls(
            series_instance_uid=str(payload["series_instance_uid"]),
            frame_of_reference_uid=str(payload["frame_of_reference_uid"]),
            shape_zyx=tuple(int(value) for value in payload["shape_zyx"]),
            spacing_xyz_mm=tuple(float(value) for value in payload["spacing_xyz_mm"]),
            origin_lps_mm=tuple(float(value) for value in payload["origin_lps_mm"]),
            direction_lps=tuple(float(value) for value in payload["direction_lps"]),
            slice_thickness_mm=float(payload["slice_thickness_mm"]),
        )


@dataclass(frozen=True)
class LoadedDicomSeries:
    """Every phase and timestamp from one exact DICOM series."""

    signal_tzyx: np.ndarray
    times_seconds: np.ndarray
    geometry: DicomGeometry
    archive_path: Path
    source_kind: str
    source_sha256: str
    temporal_positions: tuple[int, ...]


def parse_dicom_clock_seconds(value: object) -> float:
    """Convert a DICOM TM value to seconds after midnight."""
    text = str(value).strip()
    if len(text) < DICOM_TM_MINIMUM_CHARACTERS:
        raise ValueError(f"invalid DICOM acquisition time: {value!r}")
    return float(int(text[:2]) * 3600 + int(text[2:4]) * 60 + float(text[4:]))


def _inventory_columns(path: Path) -> set[str]:
    """Return an inventory's columns without loading every DICOM row."""
    import pandas as pd

    if path.suffix.lower() == ".parquet":
        try:
            from fastparquet import ParquetFile

            return set(ParquetFile(path).columns)
        except ImportError:
            return set(pd.read_parquet(path).columns)
    return set(pd.read_csv(path, nrows=0).columns)


def _read_inventory(
    path: Path, *, study_uid: str, series_uid: str
) -> tuple[Any, str, Path]:
    import pandas as pd

    available = _inventory_columns(path)
    columns = [
        "study_instance_uid",
        "series_instance_uid",
        "read_ok",
        "temporal_position_identifier",
        "instance_number",
    ]
    columns.extend(
        column
        for column in ("archive_path", "archive_member", "source_path")
        if column in available
    )
    if "sop_instance_uid" in available:
        columns.append("sop_instance_uid")
    missing = set(columns[:5]).difference(available)
    if missing:
        raise ValueError(f"inventory is missing required columns: {sorted(missing)}")
    has_archive = {"archive_path", "archive_member"}.issubset(available)
    has_files = "source_path" in available
    if not has_archive and not has_files:
        raise ValueError(
            "inventory requires either archive_path/archive_member or source_path"
        )
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(
            path,
            columns=columns,
            filters=[
                ("study_instance_uid", "==", study_uid),
                ("series_instance_uid", "==", series_uid),
                ("read_ok", "==", True),  # noqa: E712
            ],
        )
    else:
        frame = pd.read_csv(path, usecols=columns)
    readable = frame["read_ok"].fillna(False).isin((True, 1, "true", "True"))
    frame = frame.loc[
        (frame["study_instance_uid"].astype(str) == study_uid)
        & (frame["series_instance_uid"].astype(str) == series_uid)
        & readable
    ].copy()
    if frame.empty:
        raise ValueError(f"no readable inventory rows for series {series_uid}")
    if frame["temporal_position_identifier"].isna().any():
        raise ValueError(f"series {series_uid} has missing temporal positions")
    archive_rows = (
        frame["archive_path"].fillna("").astype(str).str.strip().ne("")
        if has_archive
        else pd.Series(False, index=frame.index)
    )
    file_rows = (
        frame["source_path"].fillna("").astype(str).str.strip().ne("")
        if has_files
        else pd.Series(False, index=frame.index)
    )
    if archive_rows.all() and not file_rows.any():
        archives = sorted(set(frame["archive_path"].astype(str)))
        if len(archives) != 1:
            raise ValueError(f"expected one archive for {series_uid}, got {archives}")
        if frame["archive_member"].isna().any():
            raise ValueError(f"series {series_uid} has missing archive members")
        return frame, "zip", Path(archives[0])
    if file_rows.all() and not archive_rows.any():
        source_paths = [Path(value) for value in frame["source_path"].astype(str)]
        missing_files = [str(value) for value in source_paths if not value.is_file()]
        if missing_files:
            raise FileNotFoundError(
                f"series {series_uid} has missing source files: {missing_files[:3]}"
            )
        common_parent = Path(
            os.path.commonpath([str(value.parent) for value in source_paths])
        )
        return frame, "filesystem", common_parent
    raise ValueError(f"series {series_uid} mixes ZIP and filesystem source rows")


def _phase_geometry(
    datasets: list[Any], *, series_uid: str
) -> tuple[DicomGeometry, list[int]]:
    first = datasets[0]
    orientation = np.asarray(first.ImageOrientationPatient, dtype=np.float64)
    if orientation.shape != (6,):
        raise ValueError(f"unexpected ImageOrientationPatient: {orientation}")
    x_direction = orientation[:3] / np.linalg.norm(orientation[:3])
    y_direction = orientation[3:] / np.linalg.norm(orientation[3:])
    z_direction = np.cross(x_direction, y_direction)
    z_direction /= np.linalg.norm(z_direction)
    positions = np.stack(
        [
            np.asarray(dataset.ImagePositionPatient, dtype=np.float64)
            for dataset in datasets
        ]
    )
    projections = positions @ z_direction
    order = np.argsort(projections).tolist()
    differences = np.diff(projections[order])
    if differences.size == 0 or np.any(differences <= 0):
        raise ValueError(f"series {series_uid} has invalid slice positions")
    slice_spacing = float(np.median(differences))
    if not np.allclose(
        differences, slice_spacing, rtol=0.0, atol=PHYSICAL_TOLERANCE_MM
    ):
        raise ValueError(f"series {series_uid} has irregular slice spacing")
    pixel_spacing = np.asarray(first.PixelSpacing, dtype=np.float64)
    if pixel_spacing.shape != (2,) or np.any(pixel_spacing <= 0):
        raise ValueError(f"invalid PixelSpacing: {pixel_spacing}")
    direction = np.column_stack((x_direction, y_direction, z_direction))
    return (
        DicomGeometry(
            series_instance_uid=series_uid,
            frame_of_reference_uid=str(first.FrameOfReferenceUID),
            shape_zyx=(len(datasets), int(first.Rows), int(first.Columns)),
            spacing_xyz_mm=(
                float(pixel_spacing[1]),
                float(pixel_spacing[0]),
                slice_spacing,
            ),
            origin_lps_mm=tuple(float(value) for value in positions[order[0]]),
            direction_lps=tuple(float(value) for value in direction.reshape(-1)),
            slice_thickness_mm=float(first.SliceThickness),
        ),
        order,
    )


def _assert_same_geometry(reference: DicomGeometry, candidate: DicomGeometry) -> None:
    checks = {
        "shape": reference.shape_zyx == candidate.shape_zyx,
        "frame": reference.frame_of_reference_uid == candidate.frame_of_reference_uid,
        "spacing": np.allclose(
            reference.spacing_xyz_mm,
            candidate.spacing_xyz_mm,
            atol=PHYSICAL_TOLERANCE_MM,
            rtol=0.0,
        ),
        "origin": np.allclose(
            reference.origin_lps_mm,
            candidate.origin_lps_mm,
            atol=PHYSICAL_TOLERANCE_MM,
            rtol=0.0,
        ),
        "direction": np.allclose(
            reference.direction_lps, candidate.direction_lps, atol=1e-6, rtol=0.0
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"geometry differs between temporal positions: {checks}")


def load_dicom_series(
    inventory_path: str | Path,
    *,
    study_uid: str,
    series_uid: str,
) -> LoadedDicomSeries:
    """Load all temporal positions from one exact ZIP- or filesystem-backed series."""
    import pydicom

    inventory = Path(inventory_path).expanduser().resolve()
    rows, source_kind, source_location = _read_inventory(
        inventory, study_uid=study_uid, series_uid=series_uid
    )
    phases: list[np.ndarray] = []
    clocks: list[float] = []
    reference: DicomGeometry | None = None
    source_digest = hashlib.sha256()
    temporal_positions = tuple(
        sorted(
            int(round(float(value)))
            for value in rows["temporal_position_identifier"].unique()
        )
    )
    archive = zipfile.ZipFile(source_location) if source_kind == "zip" else None
    try:
        for temporal_position in temporal_positions:
            phase_rows = rows.loc[
                np.isclose(
                    rows["temporal_position_identifier"].astype(float),
                    temporal_position,
                )
            ].sort_values("instance_number")
            datasets: list[Any] = []
            locations = (
                phase_rows["archive_member"].astype(str)
                if archive is not None
                else phase_rows["source_path"].astype(str)
            )
            for row_index, location in zip(phase_rows.index, locations, strict=True):
                payload = (
                    archive.read(location)
                    if archive is not None
                    else Path(location).read_bytes()
                )
                identity = (
                    location
                    if archive is not None
                    else str(phase_rows.loc[row_index, "sop_instance_uid"])
                    if "sop_instance_uid" in phase_rows
                    else location
                )
                source_digest.update(identity.encode())
                source_digest.update(payload)
                datasets.append(pydicom.dcmread(io.BytesIO(payload), force=True))
            geometry, order = _phase_geometry(datasets, series_uid=series_uid)
            if reference is None:
                reference = geometry
            else:
                _assert_same_geometry(reference, geometry)
            volume = np.stack(
                [np.asarray(datasets[index].pixel_array) for index in order], axis=0
            )
            if volume.shape != geometry.shape_zyx:
                raise ValueError(f"loaded shape {volume.shape} != {geometry.shape_zyx}")
            if not np.all(np.isfinite(volume)) or np.any(volume < 0):
                raise ValueError("DICOM MR signal must be finite and nonnegative")
            phases.append(np.asarray(volume, dtype=np.float32))
            clocks.append(parse_dicom_clock_seconds(datasets[order[0]].AcquisitionTime))
    finally:
        if archive is not None:
            archive.close()
    if reference is None:
        raise ValueError(f"no DICOM phases loaded for {series_uid}")
    adjusted: list[float] = []
    day_offset = 0.0
    previous = clocks[0]
    for clock in clocks:
        value = clock + day_offset
        if value < previous:
            day_offset += SECONDS_PER_DAY
            value = clock + day_offset
        adjusted.append(value - clocks[0])
        previous = value
    times = np.asarray(adjusted, dtype=np.float64)
    if not np.all(np.diff(times) > 0):
        raise ValueError("physical acquisition times are not strictly increasing")
    return LoadedDicomSeries(
        signal_tzyx=np.stack(phases, axis=0),
        times_seconds=times,
        geometry=reference,
        archive_path=source_location,
        source_kind=source_kind,
        source_sha256=source_digest.hexdigest(),
        temporal_positions=temporal_positions,
    )


def geometry_alignment_checks(
    hr: DicomGeometry, ufast: DicomGeometry
) -> dict[str, object]:
    """Describe whether identity mapping is justified by DICOM geometry."""
    hr_fov = np.asarray(hr.shape_zyx[::-1]) * np.asarray(hr.spacing_xyz_mm)
    ufast_fov = np.asarray(ufast.shape_zyx[::-1]) * np.asarray(ufast.spacing_xyz_mm)
    origin_delta = np.asarray(ufast.origin_lps_mm) - np.asarray(hr.origin_lps_mm)
    return {
        "same_frame_of_reference_uid": hr.frame_of_reference_uid
        == ufast.frame_of_reference_uid,
        "direction_equal": bool(
            np.allclose(hr.direction_lps, ufast.direction_lps, atol=1e-6, rtol=0.0)
        ),
        "field_of_view_difference_xyz_mm": (ufast_fov - hr_fov).tolist(),
        "origin_delta_ufast_minus_hr_lps_mm": origin_delta.tolist(),
        "origin_delta_norm_mm": float(np.linalg.norm(origin_delta)),
    }
