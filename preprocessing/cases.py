"""Explicit case manifest for paired HR and UFAST DCE series."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CaseRecord:
    """Exact DICOM series identifiers needed for one preprocessing run."""

    exam_id: str
    dataset: str
    study_instance_uid: str
    hr_series_instance_uid: str
    ufast_series_instance_uid: str
    ufast_baseline_frame_count: int


def _validate_path_component(value: str, *, field: str) -> str:
    """Reject identifiers that could escape the derived-output layout."""
    if not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError(f"{field} must be one safe path component, got {value!r}")
    return value


def read_case_manifest(path: str | Path) -> list[CaseRecord]:
    """Read a case manifest without guessing series or changing the cohort."""
    required = {
        "exam_id",
        "dataset",
        "study_instance_uid",
        "hr_series_instance_uid",
        "ufast_series_instance_uid",
        "ufast_baseline_frame_count",
    }
    records: list[CaseRecord] = []
    seen: set[str] = set()
    with Path(path).expanduser().open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"case manifest is missing columns: {sorted(missing)}")
        for row in reader:
            record = CaseRecord(
                exam_id=_validate_path_component(
                    str(row["exam_id"]).strip(), field="exam_id"
                ),
                dataset=_validate_path_component(
                    str(row["dataset"]).strip(), field="dataset"
                ),
                study_instance_uid=str(row["study_instance_uid"]).strip(),
                hr_series_instance_uid=str(row["hr_series_instance_uid"]).strip(),
                ufast_series_instance_uid=str(row["ufast_series_instance_uid"]).strip(),
                ufast_baseline_frame_count=int(row["ufast_baseline_frame_count"]),
            )
            if not all(
                (
                    record.exam_id,
                    record.dataset,
                    record.study_instance_uid,
                    record.hr_series_instance_uid,
                    record.ufast_series_instance_uid,
                )
            ):
                raise ValueError("case manifest identifiers cannot be empty")
            if record.exam_id in seen:
                raise ValueError(f"duplicate exam_id: {record.exam_id}")
            seen.add(record.exam_id)
            records.append(record)
    if not records:
        raise ValueError("case manifest contains no cases")
    return records


def select_case(path: str | Path, exam_id: str) -> CaseRecord:
    """Select exactly one case by its stable exam identifier."""
    matches = [
        record for record in read_case_manifest(path) if record.exam_id == exam_id
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {exam_id!r} row, got {len(matches)}")
    return matches[0]
