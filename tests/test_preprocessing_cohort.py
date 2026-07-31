"""Tests for stable cohort-array selection and two-route completion checks."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from preprocessing.cases import CaseRecord
from preprocessing.merge import LABEL_CONFIRMED_BOTH, LABEL_HR_ONLY, merge_skeletons
from preprocessing.model import model_subject_id
from preprocessing.pipeline import POLICY_NAME
from preprocessing.run_cohort import (
    _assert_reuse_contract,
    _stage_complete,
    select_array_case,
)


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
        "policy": {"name": "vanguard_spgr_raw_signal_v5"},
        "case": record.__dict__,
        "inventory_path": str(inventory.resolve()),
        "inventory_sha256": hashlib.sha256(inventory.read_bytes()).hexdigest(),
        "case_manifest_path": str(manifest.resolve()),
        "case_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    (case_root / "preprocessing_provenance.json").write_text(json.dumps(provenance))

    assert POLICY_NAME == "vanguard_spgr_raw_signal_v6"
    with pytest.raises(RuntimeError, match="changed preprocessing policy"):
        _assert_reuse_contract(
            case_root=case_root,
            record=record,
            inventory=inventory,
            case_manifest=manifest,
            breast_model=tmp_path / "breast.pth",
            vessel_model=tmp_path / "vessel.pth",
        )


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def test_infer_stage_requires_two_route_phase_counts_and_artifacts(
    tmp_path: Path,
) -> None:
    """HR-only inference provenance must not look complete under the merge contract."""
    exam_id = "exam_a"
    case_root = tmp_path / "work" / exam_id
    case_root.mkdir(parents=True)
    (case_root / "preprocessing_provenance.json").write_text(
        json.dumps({"inference": {"phases": 2}})
    )
    assert not _stage_complete("infer", case_root)

    for index in range(2):
        _touch(case_root / "hr_breast_predictions" / f"{model_subject_id(index)}.npy")
        _touch(case_root / "hr_vessel_predictions" / f"{model_subject_id(index)}.npz")
    (case_root / "preprocessing_provenance.json").write_text(
        json.dumps({"inference": {"hr_phases": 2, "ufast_phases": 3}})
    )
    with pytest.raises(RuntimeError, match="artifacts are missing"):
        _stage_complete("infer", case_root)

    for index in range(3):
        _touch(case_root / "ufast_breast_predictions" / f"{exam_id}_{index:04d}.npy")
        _touch(case_root / "ufast_vessel_predictions" / f"{exam_id}_{index:04d}.npz")
    assert _stage_complete("infer", case_root)


def test_tc4d_and_map_and_qc_require_merged_route_artifacts(tmp_path: Path) -> None:
    """Completion checks must see UFAST TC4D, merge sidecars, and temporal MIP."""
    exam_id = "exam_a"
    case_root = tmp_path / "work" / exam_id
    case_root.mkdir(parents=True)
    centerline_dir = tmp_path / "centerlines" / "cohort" / exam_id
    provenance = {
        "case": {"dataset": "cohort"},
        "tc4d": {},
        "mapping": {},
    }
    (case_root / "preprocessing_provenance.json").write_text(json.dumps(provenance))
    for name in (
        "exam_skeleton_zyx.npy",
        "exam_support_zyx.npy",
        "center_manifold_4d_yxz.npy",
    ):
        _touch(case_root / "hr_tc4d" / name)
    assert not _stage_complete("tc4d", case_root)

    provenance["ufast_tc4d"] = {}
    (case_root / "preprocessing_provenance.json").write_text(json.dumps(provenance))
    for name in (
        "exam_skeleton_zyx.npy",
        "exam_support_zyx.npy",
        "center_manifold_4d_yxz.npy",
    ):
        _touch(case_root / "ufast_tc4d" / name)
    assert _stage_complete("tc4d", case_root)

    for name in (
        f"{exam_id}_skeleton_4d_exam_mask.npy",
        f"{exam_id}_skeleton_4d_exam_support_mask.npy",
        "run_summary.json",
    ):
        _touch(centerline_dir / name)
    assert not _stage_complete("map", case_root)

    provenance["merge"] = {}
    (case_root / "preprocessing_provenance.json").write_text(json.dumps(provenance))
    with pytest.raises(RuntimeError, match="artifacts are missing"):
        _stage_complete("map", case_root)
    _touch(centerline_dir / f"{exam_id}_skeleton_4d_exam_mask_hr_only.npy")
    _touch(centerline_dir / f"{exam_id}_merge_provenance_label_zyx.npy")
    _touch(centerline_dir / "merge_provenance.json")
    assert _stage_complete("map", case_root)

    _touch(centerline_dir / "mapping_qc.png")
    assert not _stage_complete("qc", case_root)
    _touch(centerline_dir / "temporal_mip.png")
    assert _stage_complete("qc", case_root)


def test_confirmed_both_labels_hr_voxels_near_ufast_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Proximity agreement is labeled on surviving HR coordinates, not UFAST ones."""
    monkeypatch.setattr(
        "preprocessing.merge._finalize_exam_mask_topology",
        lambda mask, support_count_zyx: mask,
    )
    shape = (9, 9, 9)
    hr = np.zeros(shape, dtype=bool)
    ufast = np.zeros(shape, dtype=bool)
    # Two voxels apart along z: within the default radius-2 proximity window,
    # but not the same voxel, so merged_skeleton & confirmed_ufast_mask is empty.
    hr[4, 4, 4] = True
    ufast[4, 4, 6] = True
    support = np.zeros(shape, dtype=bool)
    support[4, 4, 4] = True
    support[4, 4, 6] = True

    _, _, provenance, labels = merge_skeletons(
        preprocessing_root=tmp_path,
        exam_id="exam_a",
        hr_skeleton=hr,
        hr_support=support,
        ufast_skeleton=ufast,
        ufast_support=support,
        enable_kinetics_filter=False,
    )

    assert labels[4, 4, 4] == LABEL_CONFIRMED_BOTH
    assert labels[4, 4, 6] == 0  # proximity-confirmed UFAST voxel is not kept
    assert provenance["label_voxel_counts"]["confirmed_both"] == 1
    assert provenance["label_voxel_counts"]["hr_only"] == 0

    # An HR voxel far from every UFAST voxel stays hr_only.
    hr_far = hr.copy()
    hr_far[0, 0, 0] = True
    support_far = support.copy()
    support_far[0, 0, 0] = True
    _, _, provenance_far, labels_far = merge_skeletons(
        preprocessing_root=tmp_path,
        exam_id="exam_a",
        hr_skeleton=hr_far,
        hr_support=support_far,
        ufast_skeleton=ufast,
        ufast_support=support,
        enable_kinetics_filter=False,
    )
    assert labels_far[0, 0, 0] == LABEL_HR_ONLY
    assert provenance_far["label_voxel_counts"]["hr_only"] == 1
    assert provenance_far["label_voxel_counts"]["confirmed_both"] == 1
