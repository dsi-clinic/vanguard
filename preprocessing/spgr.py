"""Load all frames from a checksum-tracked SPGR-safe DCE archive.

The loader deliberately has no phase-selection or temporal-resampling option.
Every acquired frame is returned in manifest order with its recorded time.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import tarfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

RAW_SIGNAL_POLICY = "hfdp_t1_raw_signal_v1"
MINIMUM_DYNAMIC_FRAMES = 2
NIFTI_SPATIAL_DIMENSIONS = 3
DYNAMIC_ARRAY_DIMENSIONS = 4


class PreprocessingContractError(ValueError):
    """Raised when an exam violates the declared preprocessing contract."""


def _strict_bool(value: object, *, field: str) -> bool:
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    message = f"{field} must be explicitly true or false, got {value!r}"
    raise PreprocessingContractError(message)


def _json_list(value: str, *, field: str) -> list[object]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise PreprocessingContractError(f"{field} is not valid JSON") from exc
    if not isinstance(parsed, list):
        raise PreprocessingContractError(f"{field} must contain a JSON list")
    return parsed


def _resolve_path(value: str, *, manifest_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else manifest_dir / path


@dataclass(frozen=True)
class ExamRecord:
    """Fields needed to load one DCE exam from an archive manifest."""

    row_index: int
    exam_id: str
    dataset: str
    phase_archive_path: Path
    phase_archive_members: tuple[str, ...]
    phase_member_sha256: tuple[str, ...]
    phase_member_bytes: tuple[int, ...]
    acquisition_times_seconds: tuple[float, ...]
    n_phases: int
    preprocessing_policy: str
    motion_correction_applied: bool
    baseline_frame_count: int
    repetition_time_ms: float
    flip_angle_degrees: float
    mask_archive_path: Path | None
    mask_archive_member: str | None
    mask_sha256: str | None

    @classmethod
    def from_row(
        cls,
        row: Mapping[str, str],
        *,
        row_index: int,
        manifest_dir: Path,
    ) -> ExamRecord:
        """Parse and validate one CSV row."""
        required = {
            "exam_id",
            "dataset",
            "n_phases",
            "times_seconds_json",
            "phase_archive_path",
            "phase_archive_members_json",
            "phase_member_sha256_json",
            "preprocessing_policy",
            "motion_correction_applied",
            "baseline_frame_count",
            "repetition_time_ms",
            "flip_angle_degrees",
        }
        missing = sorted(
            field for field in required if not str(row.get(field, "")).strip()
        )
        if missing:
            message = f"manifest row is missing required fields: {missing}"
            raise PreprocessingContractError(message)

        members = tuple(
            str(value)
            for value in _json_list(
                row["phase_archive_members_json"], field="phase_archive_members_json"
            )
        )
        hashes = tuple(
            str(value)
            for value in _json_list(
                row["phase_member_sha256_json"], field="phase_member_sha256_json"
            )
        )
        bytes_text = str(row.get("phase_member_bytes_json", "")).strip()
        sizes = (
            tuple(
                int(value)
                for value in _json_list(bytes_text, field="phase_member_bytes_json")
            )
            if bytes_text
            else tuple(0 for _ in members)
        )
        times = tuple(
            float(value)
            for value in _json_list(
                row["times_seconds_json"], field="times_seconds_json"
            )
        )
        n_phases = int(row["n_phases"])
        if not (n_phases == len(members) == len(hashes) == len(sizes) == len(times)):
            message = (
                "n_phases, archive members, checksums, sizes, and acquisition times "
                "must have equal lengths"
            )
            raise PreprocessingContractError(message)
        if n_phases < MINIMUM_DYNAMIC_FRAMES:
            raise PreprocessingContractError(
                "a dynamic exam must contain at least two frames"
            )
        if not np.all(np.isfinite(times)) or not np.all(np.diff(times) > 0):
            message = "acquisition times must be finite and strictly increasing"
            raise PreprocessingContractError(message)

        baseline_frame_count = int(row["baseline_frame_count"])
        if not 1 <= baseline_frame_count < n_phases:
            message = "baseline_frame_count must be in [1, n_phases)"
            raise PreprocessingContractError(message)

        mask_path_text = str(row.get("whole_breast_mask_archive_path", "")).strip()
        mask_member = (
            str(row.get("whole_breast_mask_archive_member", "")).strip() or None
        )
        if bool(mask_path_text) != bool(mask_member):
            message = (
                "whole-breast mask archive path and member must either both be set "
                "or both be empty"
            )
            raise PreprocessingContractError(message)

        return cls(
            row_index=int(row_index),
            exam_id=str(row["exam_id"]),
            dataset=str(row["dataset"]),
            phase_archive_path=_resolve_path(
                row["phase_archive_path"], manifest_dir=manifest_dir
            ),
            phase_archive_members=members,
            phase_member_sha256=hashes,
            phase_member_bytes=sizes,
            acquisition_times_seconds=times,
            n_phases=n_phases,
            preprocessing_policy=str(row["preprocessing_policy"]),
            motion_correction_applied=_strict_bool(
                row["motion_correction_applied"], field="motion_correction_applied"
            ),
            baseline_frame_count=baseline_frame_count,
            repetition_time_ms=float(row["repetition_time_ms"]),
            flip_angle_degrees=float(row["flip_angle_degrees"]),
            mask_archive_path=(
                _resolve_path(mask_path_text, manifest_dir=manifest_dir)
                if mask_path_text
                else None
            ),
            mask_archive_member=mask_member,
            mask_sha256=str(row.get("whole_breast_mask_sha256", "")).strip() or None,
        )


@dataclass(frozen=True)
class LoadedExam:
    """A complete acquired DCE sequence and its matching breast mask."""

    record: ExamRecord
    raw_signal: np.ndarray
    acquisition_times_seconds: np.ndarray
    affine: np.ndarray
    whole_breast_mask: np.ndarray | None


@dataclass(frozen=True)
class RelativeEnhancement:
    """Baseline-relative signal and the support on which it is valid."""

    values: np.ndarray
    baseline_signal: np.ndarray
    valid_support: np.ndarray


def read_manifest(manifest_path: str | Path) -> list[ExamRecord]:
    """Read and validate every row while preserving manifest order."""
    path = Path(manifest_path).expanduser().resolve()
    records: list[ExamRecord] = []
    seen: set[str] = set()
    with path.open(newline="") as stream:
        for row_index, row in enumerate(csv.DictReader(stream)):
            record = ExamRecord.from_row(
                row,
                row_index=row_index,
                manifest_dir=path.parent,
            )
            if record.exam_id in seen:
                raise PreprocessingContractError(f"duplicate exam_id: {record.exam_id}")
            seen.add(record.exam_id)
            records.append(record)
    if not records:
        raise PreprocessingContractError("manifest contains no exams")
    return records


def read_member_payload(
    archive: tarfile.TarFile,
    member_name: str,
    *,
    expected_sha256: str | None,
) -> bytes:
    """Read one regular TAR member and optionally verify its checksum."""
    try:
        extracted = archive.extractfile(member_name)
    except KeyError as exc:
        raise PreprocessingContractError(
            f"archive member is missing: {member_name}"
        ) from exc
    if extracted is None:
        message = f"archive member is not a regular file: {member_name}"
        raise PreprocessingContractError(message)
    payload = extracted.read()
    if expected_sha256 and hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise PreprocessingContractError(
            f"checksum mismatch for archive member: {member_name}"
        )
    return payload


def nifti_from_payload(payload: bytes, *, member_name: str) -> nib.Nifti1Image:
    """Decode a compressed or uncompressed single-file NIfTI payload."""
    if member_name.endswith(".gz"):
        try:
            payload = gzip.decompress(payload)
        except gzip.BadGzipFile as exc:
            raise PreprocessingContractError(
                f"invalid gzip member: {member_name}"
            ) from exc
    try:
        image = nib.Nifti1Image.from_bytes(payload)
    except Exception as exc:
        raise PreprocessingContractError(
            f"invalid NIfTI member: {member_name}"
        ) from exc
    if len(image.shape) != NIFTI_SPATIAL_DIMENSIONS:
        message = f"expected a 3D frame, got {image.shape}: {member_name}"
        raise PreprocessingContractError(message)
    return image


def load_nifti_member(
    archive: tarfile.TarFile,
    member_name: str,
    *,
    expected_sha256: str | None,
) -> tuple[nib.Nifti1Image, bytes]:
    """Read, verify, and decode one NIfTI archive member."""
    payload = read_member_payload(
        archive,
        member_name,
        expected_sha256=expected_sha256,
    )
    return nifti_from_payload(payload, member_name=member_name), payload


def _load_phase_sequence(record: ExamRecord) -> tuple[np.ndarray, np.ndarray]:
    if record.preprocessing_policy != RAW_SIGNAL_POLICY:
        message = f"unexpected preprocessing policy: {record.preprocessing_policy!r}"
        raise PreprocessingContractError(message)

    sequence: np.ndarray | None = None
    reference_affine: np.ndarray | None = None
    reference_shape: tuple[int, ...] | None = None
    with tarfile.open(record.phase_archive_path, mode="r") as archive:
        for phase_index, (member, checksum) in enumerate(
            zip(record.phase_archive_members, record.phase_member_sha256, strict=True)
        ):
            image, _ = load_nifti_member(
                archive,
                member,
                expected_sha256=checksum,
            )
            data = np.asarray(image.dataobj, dtype=np.float32)
            if not np.all(np.isfinite(data)):
                message = f"nonfinite signal in {record.exam_id}, frame {phase_index}"
                raise PreprocessingContractError(message)
            if np.any(data < 0):
                message = f"negative signal in {record.exam_id}, frame {phase_index}"
                raise PreprocessingContractError(message)

            if sequence is None:
                reference_shape = tuple(int(value) for value in data.shape)
                reference_affine = np.asarray(image.affine, dtype=np.float64)
                sequence = np.empty(
                    (record.n_phases, *reference_shape), dtype=np.float32
                )
            elif data.shape != reference_shape or not np.allclose(
                image.affine,
                reference_affine,
                rtol=0.0,
                atol=1e-5,
            ):
                message = (
                    f"within-exam geometry mismatch in {record.exam_id}, "
                    f"frame {phase_index}"
                )
                raise PreprocessingContractError(message)
            sequence[phase_index] = data

    if sequence is None or reference_affine is None:
        raise PreprocessingContractError(f"no frames loaded for {record.exam_id}")
    return sequence, reference_affine


def load_mask(
    record: ExamRecord, *, shape: Sequence[int], affine: np.ndarray
) -> np.ndarray | None:
    """Load and validate the optional whole-breast label map."""
    if record.mask_archive_path is None or record.mask_archive_member is None:
        return None
    with tarfile.open(record.mask_archive_path, mode="r") as archive:
        image, _ = load_nifti_member(
            archive,
            record.mask_archive_member,
            expected_sha256=record.mask_sha256,
        )
    if tuple(image.shape) != tuple(shape) or not np.allclose(
        image.affine,
        affine,
        rtol=0.0,
        atol=1e-5,
    ):
        message = f"whole-breast mask geometry mismatch for {record.exam_id}"
        raise PreprocessingContractError(message)
    data = np.asarray(image.dataobj)
    if not np.all(np.isfinite(data)) or not np.all(data == np.rint(data)):
        message = f"whole-breast mask is not an integer label map: {record.exam_id}"
        raise PreprocessingContractError(message)
    labels = {int(value) for value in np.unique(data)}
    if not labels.issubset({0, 1, 2}):
        message = f"unexpected whole-breast mask labels {labels}: {record.exam_id}"
        raise PreprocessingContractError(message)
    return np.asarray(data, dtype=np.uint8)


def load_exam(record: ExamRecord) -> LoadedExam:
    """Load every frame and the exact-grid whole-breast mask."""
    signal, affine = _load_phase_sequence(record)
    mask = load_mask(record, shape=signal.shape[1:], affine=affine)
    return LoadedExam(
        record=record,
        raw_signal=signal,
        acquisition_times_seconds=np.asarray(
            record.acquisition_times_seconds,
            dtype=np.float64,
        ),
        affine=affine,
        whole_breast_mask=mask,
    )


def baseline_relative_enhancement(
    signal: np.ndarray,
    *,
    baseline_frame_count: int,
    support_mask: np.ndarray | None = None,
) -> RelativeEnhancement:
    """Compute unclipped ``(S(t) - S0) / S0`` for every acquired frame."""
    array = np.asarray(signal)
    if array.ndim != DYNAMIC_ARRAY_DIMENSIONS:
        message = "signal must have shape [time, x, y, z]"
        raise PreprocessingContractError(message)
    if not 1 <= int(baseline_frame_count) < array.shape[0]:
        message = "baseline_frame_count must be in [1, n_phases)"
        raise PreprocessingContractError(message)
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        raise PreprocessingContractError("signal must be finite and nonnegative")

    baseline = (
        array[: int(baseline_frame_count)]
        .mean(
            axis=0,
            dtype=np.float64,
        )
        .astype(np.float32)
    )
    valid = np.isfinite(baseline) & (baseline > 0)
    if support_mask is not None:
        support = np.asarray(support_mask)
        if support.shape != array.shape[1:]:
            message = "support_mask geometry does not match signal"
            raise PreprocessingContractError(message)
        valid &= support > 0

    relative = np.zeros(array.shape, dtype=np.float32)
    baseline_valid = baseline[valid]
    for phase_index in range(array.shape[0]):
        values = np.asarray(array[phase_index], dtype=np.float32)
        relative[phase_index, valid] = (values[valid] - baseline_valid) / baseline_valid
    if not np.all(np.isfinite(relative)):
        message = "relative enhancement contains nonfinite values"
        raise PreprocessingContractError(message)
    return RelativeEnhancement(
        values=relative,
        baseline_signal=baseline,
        valid_support=valid,
    )
