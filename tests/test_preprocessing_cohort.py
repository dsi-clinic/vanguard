"""Tests for stable cohort-array selection."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from preprocessing.cases import CaseRecord
from preprocessing.run_cohort import _assert_reuse_contract, select_array_case


def test_array_selection_is_stable_by_exam_id(tmp_path: Path) -> None:
    """Slurm array indices must not depend on CSV row order."""
    path = tmp_path / "cases.csv"
    fields = [
        "exam_id",
        "dataset",
        "study_instance_uid",
        "hr_series_instance_uid",
        "ufast_series_instance_uid",
        "ufast_baseline_frame_count",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for exam_id in ("case-b", "case-a"):
            writer.writerow(
                {
                    "exam_id": exam_id,
                    "dataset": "cohort",
                    "study_instance_uid": f"study-{exam_id}",
                    "hr_series_instance_uid": f"hr-{exam_id}",
                    "ufast_series_instance_uid": f"ufast-{exam_id}",
                    "ufast_baseline_frame_count": 1,
                }
            )
    assert select_array_case(path, 0).exam_id == "case-a"
    assert select_array_case(path, 1).exam_id == "case-b"


def test_existing_case_rejects_an_old_preprocessing_policy(tmp_path: Path) -> None:
    """Resubmission must not silently call an older run complete."""
    record = CaseRecord("case", "cohort", "study", "hr", "ufast", 1)
    inventory = tmp_path / "inventory.parquet"
    manifest = tmp_path / "cases.csv"
    inventory.write_bytes(b"inventory")
    manifest.write_text("manifest")
    case_root = tmp_path / "outputs" / "work" / record.exam_id
    case_root.mkdir(parents=True)
    provenance = {
        "policy": {"name": "vanguard_spgr_raw_signal_v3"},
        "case": record.__dict__,
        "inventory_path": str(inventory.resolve()),
        "inventory_sha256": hashlib.sha256(inventory.read_bytes()).hexdigest(),
        "case_manifest_path": str(manifest.resolve()),
        "case_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    (case_root / "preprocessing_provenance.json").write_text(json.dumps(provenance))

    with pytest.raises(RuntimeError, match="changed preprocessing policy"):
        _assert_reuse_contract(
            case_root=case_root,
            record=record,
            inventory=inventory,
            case_manifest=manifest,
            breast_model=tmp_path / "breast.pth",
            vessel_model=tmp_path / "vessel.pth",
        )
