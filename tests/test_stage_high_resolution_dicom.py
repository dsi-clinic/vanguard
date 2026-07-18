"""Tests for restricted, byte-preserving HR DICOM staging."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from preprocessing.stage_high_resolution_dicom import finalize, stage_exam

EXPECTED_FILES = 4


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, bytes]]:
    payloads = {
        "source/hr-a.dcm": b"first hr dicom",
        "source/hr-b.dcm": b"second hr dicom",
        "source/ufast-a.dcm": b"first ufast dicom",
        "source/ufast-b.dcm": b"second ufast dicom",
    }
    source_zip = tmp_path / "source.zip"
    with zipfile.ZipFile(source_zip, "w") as archive:
        for member, payload in payloads.items():
            archive.writestr(member, payload)
    inventory = pd.DataFrame(
        {
            "study_instance_uid": ["study"] * EXPECTED_FILES,
            "series_instance_uid": [
                "hr-series",
                "hr-series",
                "ufast-series",
                "ufast-series",
            ],
            "archive_path": [str(source_zip)] * EXPECTED_FILES,
            "archive_member": list(payloads),
            "read_ok": [True] * EXPECTED_FILES,
            "temporal_position_identifier": [1, 2, 1, 2],
            "instance_number": [1, 2, 1, 2],
            "sop_instance_uid": ["sop-1", "sop-2", "sop-3", "sop-4"],
            "file_size_bytes": [len(value) for value in payloads.values()],
        }
    )
    inventory_path = tmp_path / "source_inventory.parquet"
    inventory.to_parquet(inventory_path, index=False)
    selection = pd.DataFrame(
        {
            "exam_id": ["exam", "exam"],
            "dataset": ["cohort", "cohort"],
            "study_instance_uid": ["study", "study"],
            "series_instance_uid": ["hr-series", "ufast-series"],
            "series_role": ["hr", "ufast"],
            "selection_status": ["paired_complete", "paired_complete"],
            "paired_ufast_series_instance_uid": ["ufast-series", "ufast-series"],
            "ufast_baseline_frame_count": [1, 1],
            "source_inventory": [str(inventory_path), str(inventory_path)],
            "expected_n_instances": [2, 2],
        }
    )
    selection_path = tmp_path / "selection.csv"
    selection.to_csv(selection_path, index=False)
    return selection_path, source_zip, payloads


def test_stage_exam_preserves_payloads_and_omits_source_names(tmp_path: Path) -> None:
    """Staged files retain bytes without exposing source member names."""
    selection, _, payloads = _write_fixture(tmp_path)
    destination = tmp_path / "shared"
    stage_exam(selection, destination, 0)

    archive_path = destination / "archives" / "cohort" / "exam.zip"
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.namelist() == [
            "series/hr-series/000000.dcm",
            "series/hr-series/000001.dcm",
            "series/ufast-series/000000.dcm",
            "series/ufast-series/000001.dcm",
        ]
        assert [archive.read(member) for member in archive.namelist()] == list(
            payloads.values()
        )
    shard = pd.read_parquet(
        destination / "inventory_shards" / "cohort" / "exam.parquet"
    )
    assert "patient_id" not in shard.columns
    assert set(shard["archive_path"]) == {str(archive_path.resolve())}

    metadata = json.loads(
        (destination / "provenance_shards" / "cohort" / "exam.json").read_text()
    )
    assert (
        metadata["archive_sha256"]
        == hashlib.sha256(archive_path.read_bytes()).hexdigest()
    )


def test_stage_exam_rejects_reuse_after_selection_changes(tmp_path: Path) -> None:
    """A completed archive can't be relabeled under a different selection."""
    selection, _, _ = _write_fixture(tmp_path)
    destination = tmp_path / "shared"
    stage_exam(selection, destination, 0)
    changed = pd.read_csv(selection, dtype=str)
    changed["selection_status"] = "new_review_status"
    changed.to_csv(selection, index=False)

    with pytest.raises(RuntimeError, match="refusing to reuse stale"):
        stage_exam(selection, destination, 0)


def test_stage_exam_rejects_partial_existing_outputs(tmp_path: Path) -> None:
    """Restarting never overwrites one fragment of a staged package."""
    selection, _, _ = _write_fixture(tmp_path)
    destination = tmp_path / "shared"
    archive = destination / "archives" / "cohort" / "exam.zip"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"partial")

    with pytest.raises(RuntimeError, match="incomplete staged DICOM package"):
        stage_exam(selection, destination, 0)
    assert archive.read_bytes() == b"partial"


def test_finalize_links_hr_to_existing_ufast_manifest(tmp_path: Path) -> None:
    """Finalization covers the cohort and leaves the original manifest intact."""
    selection, _, _ = _write_fixture(tmp_path)
    destination = tmp_path / "shared"
    stage_exam(selection, destination, 0)
    ufast = tmp_path / "dce2d_internal_ultrafast_manifest.csv"
    original = "exam_id,dataset,n_phases\nexam,cohort,13\n"
    ufast.write_text(original)

    finalize(selection, destination, ufast)

    assert ufast.read_text() == original
    enriched = pd.read_csv(
        tmp_path / "dce2d_internal_ultrafast_with_high_resolution_manifest.csv"
    )
    assert enriched.loc[0, "hr_selection_status"] == "paired_complete"
    assert enriched.loc[0, "hr_series_instance_uids"] == "hr-series"
    assert enriched.loc[0, "ufast_series_instance_uid"] == "ufast-series"
    assert bool(enriched.loc[0, "ufast_source_staged"])
    assert Path(enriched.loc[0, "hr_source_archive_path"]).exists()
    runnable = pd.read_csv(destination / "paired_preprocessing_case_manifest.csv")
    assert runnable.loc[0, "hr_series_instance_uid"] == "hr-series"
    assert runnable.loc[0, "ufast_series_instance_uid"] == "ufast-series"
    combined = pd.read_parquet(destination / "dicom_file_manifest.parquet")
    assert len(combined) == EXPECTED_FILES


def test_finalize_excludes_shared_hr_ufast_acquisition(tmp_path: Path) -> None:
    """A low-resolution shared acquisition cannot stand in for distinct HR data."""
    selection, _, _ = _write_fixture(tmp_path)
    selected = pd.read_csv(selection, dtype=str)
    selected = selected.loc[selected["series_instance_uid"].eq("ufast-series")].copy()
    selected["series_role"] = "shared_hr_ufast"
    selected["selection_status"] = "shared_hr_ufast_no_distinct_hr"
    selected["expected_n_instances"] = 2
    selected.to_csv(selection, index=False)

    destination = tmp_path / "shared"
    stage_exam(selection, destination, 0)
    ufast = tmp_path / "dce2d_internal_ultrafast_manifest.csv"
    ufast.write_text("exam_id,dataset,n_phases\nexam,cohort,2\n")
    finalize(selection, destination, ufast)

    runnable = pd.read_csv(destination / "paired_preprocessing_case_manifest.csv")
    assert runnable.empty
    exclusions = pd.read_csv(destination / "paired_preprocessing_exclusions.csv")
    assert exclusions.loc[0, "exam_id"] == "exam"
    assert "no distinct high-resolution series" in exclusions.loc[0, "reason"]
