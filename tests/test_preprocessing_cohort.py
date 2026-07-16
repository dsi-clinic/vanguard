"""Tests for stable cohort-array selection."""

from __future__ import annotations

import csv
from pathlib import Path

from preprocessing.run_cohort import select_array_case


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
