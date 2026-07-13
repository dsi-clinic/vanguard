"""Contract tests for SPGR-safe loading and motion-corrected archives."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from scipy import ndimage

from preprocessing.merge_motion_shards import merge_motion_shards
from preprocessing.motion import (
    MotionSettings,
    correct_phase,
    motion_correct_exam_to_shard,
)
from preprocessing.spgr import (
    PreprocessingContractError,
    baseline_relative_enhancement,
    load_exam,
    read_manifest,
)

TEST_VOLUME_SIZE = 20
TRANSLATION_TOLERANCE_VOXELS = 0.3
TIGHT_TRANSLATION_BOUND_MM = 0.5


def _gzip_nifti(data: np.ndarray, affine: np.ndarray) -> bytes:
    image = nib.Nifti1Image(np.asarray(data), affine)
    return gzip.compress(image.to_bytes(), mtime=0)


def _add_tar_payload(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


def _structured_volume() -> np.ndarray:
    coordinates = np.indices((TEST_VOLUME_SIZE,) * 3, dtype=np.float32)
    first = np.exp(
        -(
            (coordinates[0] - 7.0) ** 2
            + 1.4 * (coordinates[1] - 11.0) ** 2
            + 0.7 * (coordinates[2] - 9.0) ** 2
        )
        / 18.0
    )
    second = 0.6 * np.exp(
        -(
            (coordinates[0] - 14.0) ** 2
            + (coordinates[1] - 6.0) ** 2
            + (coordinates[2] - 13.0) ** 2
        )
        / 8.0
    )
    return np.asarray(10.0 + 90.0 * (first + second), dtype=np.float32)


def _write_fixture(tmp_path: Path) -> Path:
    affine = np.eye(4, dtype=np.float64)
    fixed = _structured_volume()
    phases = [
        fixed,
        ndimage.shift(
            fixed,
            shift=(2.0, -1.0, 0.0),
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        ),
        fixed * np.float32(0.9),
    ]
    phase_archive = tmp_path / "source_phases.tar"
    members: list[str] = []
    checksums: list[str] = []
    sizes: list[int] = []
    with tarfile.open(phase_archive, mode="w") as archive:
        for phase_index, phase in enumerate(phases):
            member = f"source/exam/phase_{phase_index:04d}.nii.gz"
            payload = _gzip_nifti(phase, affine)
            _add_tar_payload(archive, member, payload)
            members.append(member)
            checksums.append(hashlib.sha256(payload).hexdigest())
            sizes.append(len(payload))

    mask = np.ones_like(fixed, dtype=np.uint8)
    mask_archive = tmp_path / "masks.tar"
    mask_member = "masks/exam.nii.gz"
    mask_payload = _gzip_nifti(mask, affine)
    with tarfile.open(mask_archive, mode="w") as archive:
        _add_tar_payload(archive, mask_member, mask_payload)

    manifest = tmp_path / "manifest.csv"
    row = {
        "exam_id": "exam",
        "dataset": "fixture",
        "patient_id": "patient",
        "n_phases": str(len(phases)),
        "times_seconds_json": "[0.0,2.0,5.0]",
        "phase_archive_path": str(phase_archive),
        "phase_archive_members_json": json.dumps(members),
        "phase_member_bytes_json": json.dumps(sizes),
        "phase_member_sha256_json": json.dumps(checksums),
        "phase_archive_sha256": hashlib.sha256(phase_archive.read_bytes()).hexdigest(),
        "preprocessing_policy": "hfdp_t1_raw_signal_v1",
        "motion_correction_applied": "false",
        "motion_correction_method": "none",
        "temporal_resampling_applied": "false",
        "baseline_frame_count": "2",
        "repetition_time_ms": "4.0",
        "flip_angle_degrees": "10.0",
        "whole_breast_mask_archive_path": str(mask_archive),
        "whole_breast_mask_archive_member": mask_member,
        "whole_breast_mask_sha256": hashlib.sha256(mask_payload).hexdigest(),
    }
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return manifest


def test_loader_preserves_every_frame_time_and_negative_relative_change(
    tmp_path: Path,
) -> None:
    """Loading and relative signal must preserve the complete temporal contract."""
    record = read_manifest(_write_fixture(tmp_path))[0]
    loaded = load_exam(record)

    assert loaded.raw_signal.shape == (
        3,
        TEST_VOLUME_SIZE,
        TEST_VOLUME_SIZE,
        TEST_VOLUME_SIZE,
    )
    np.testing.assert_array_equal(loaded.acquisition_times_seconds, [0.0, 2.0, 5.0])
    result = baseline_relative_enhancement(
        loaded.raw_signal,
        baseline_frame_count=record.baseline_frame_count,
        support_mask=loaded.whole_breast_mask,
    )
    assert np.any(result.values[2] < 0)


def test_translation_is_accepted_only_when_same_support_correlation_improves() -> None:
    """The saved transform must improve correlation or fall back to identity."""
    fixed = _structured_volume()
    moving = ndimage.shift(
        fixed,
        shift=(2.0, -1.0, 0.0),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    settings = MotionSettings(downsample_xyz=(1, 1, 1), upsample_factor=10)
    corrected, _, metrics = correct_phase(
        fixed,
        moving,
        support=np.ones_like(fixed, dtype=bool),
        spacing_xyz_mm=np.ones(3),
        settings=settings,
    )
    assert metrics["transform_accepted"] is True
    assert float(metrics["corr_delta"]) >= 0.0
    assert (
        np.linalg.norm(np.asarray(metrics["translation_voxels"]) - [-2.0, 1.0, 0.0])
        < TRANSLATION_TOLERANCE_VOXELS
    )
    assert np.mean(np.abs(corrected - fixed)) < np.mean(np.abs(moving - fixed))

    reject_settings = MotionSettings(
        downsample_xyz=(1, 1, 1),
        upsample_factor=10,
        minimum_correlation_delta=2.0,
    )
    rejected, _, rejected_metrics = correct_phase(
        fixed,
        moving,
        support=np.ones_like(fixed, dtype=bool),
        spacing_xyz_mm=np.ones(3),
        settings=reject_settings,
    )
    assert rejected_metrics["transform_accepted"] is False
    np.testing.assert_array_equal(rejected, moving)

    bound_settings = MotionSettings(
        downsample_xyz=(1, 1, 1),
        upsample_factor=10,
        max_translation_mm=TIGHT_TRANSLATION_BOUND_MM,
    )
    bounded, _, bounded_metrics = correct_phase(
        fixed,
        moving,
        support=np.ones_like(fixed, dtype=bool),
        spacing_xyz_mm=np.ones(3),
        settings=bound_settings,
    )
    assert bounded_metrics["transform_accepted"] is False
    assert (
        bounded_metrics["transform_rejection_reason"]
        == "proposed_translation_exceeds_maximum"
    )
    assert (
        float(bounded_metrics["proposed_translation_norm_mm"])
        > TIGHT_TRANSLATION_BOUND_MM
    )
    np.testing.assert_array_equal(bounded, moving)


def test_exam_shard_and_merge_publish_a_loadable_motion_manifest(
    tmp_path: Path,
) -> None:
    """The exam shard, cohort merge, checksums, and loader must round-trip."""
    manifest = _write_fixture(tmp_path)
    record = read_manifest(manifest)[0]
    output_root = tmp_path / "motion"

    metadata = motion_correct_exam_to_shard(record, output_root=output_root)
    assert metadata["status"] == "complete"
    assert int(metadata["n_phases"]) == record.n_phases
    reused = motion_correct_exam_to_shard(record, output_root=output_root)
    assert reused["shard_archive_sha256"] == metadata["shard_archive_sha256"]
    with pytest.raises(PreprocessingContractError, match="settings"):
        motion_correct_exam_to_shard(
            record,
            output_root=output_root,
            settings=MotionSettings(max_translation_mm=10.0),
        )

    summary = merge_motion_shards(
        manifest_path=manifest,
        output_root=output_root,
        maximum_qc_panels=1,
    )
    assert summary["status"] == "complete"
    assert int(summary["exams"]) == 1
    assert int(summary["phases"]) == record.n_phases

    corrected_record = read_manifest(output_root / "manifest.csv")[0]
    assert corrected_record.motion_correction_applied is True
    corrected = load_exam(corrected_record)
    assert corrected.raw_signal.shape == (3,) + (TEST_VOLUME_SIZE,) * 3
    np.testing.assert_array_equal(corrected.acquisition_times_seconds, [0.0, 2.0, 5.0])
