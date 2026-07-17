"""Unit tests for the Vanguard-owned paired HR/UFAST preprocessing contract."""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import ndimage

from gnn.raw_dce import discover_raw_dce_paths, load_raw_dce_times
from preprocessing.cases import read_case_manifest
from preprocessing.dicom import DicomGeometry, geometry_alignment_checks
from preprocessing.model import (
    frozen_model_intensity_preprocess,
    prepare_hr_phase_for_model,
)
from preprocessing.motion import MotionSettings, correct_phase
from preprocessing.pipeline import _identity_alignment_qc, _validate_pair
from preprocessing.spatial import (
    isotropic_geometry,
    rasterize_skeleton_identity,
    resample_to_geometry,
)

TEST_VOLUME_SIZE = 20
EXPECTED_BASELINE_COUNT = 5
ALIGNMENT_BASELINE_COUNT = 3


def _geometry(
    *,
    shape_zyx: tuple[int, int, int] = (100, 384, 384),
    spacing_xyz: tuple[float, float, float] = (0.78125, 0.78125, 1.750002),
    origin_lps: tuple[float, float, float] = (1.0, 2.0, 3.0),
    series_uid: str = "series",
) -> DicomGeometry:
    return DicomGeometry(
        series_instance_uid=series_uid,
        frame_of_reference_uid="frame",
        shape_zyx=shape_zyx,
        spacing_xyz_mm=spacing_xyz,
        origin_lps_mm=origin_lps,
        direction_lps=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        slice_thickness_mm=spacing_xyz[2],
    )


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
    return np.asarray(10.0 + 90.0 * first, dtype=np.float32)


def test_model_adapter_uses_native_hr_axes_and_original_intensity_contract() -> None:
    """Only HR model inputs are clipped/z-scored and axes become YXZ."""
    phase_zyx = np.arange(12 * 10 * 8, dtype=np.float32).reshape(12, 10, 8)
    result = prepare_hr_phase_for_model(phase_zyx)
    assert result.shape == (10, 8, 12)
    assert np.isclose(float(result.mean()), 0.0, atol=1e-6)
    assert np.isclose(float(result.std()), 1.0, atol=1e-6)

    direct = frozen_model_intensity_preprocess(np.transpose(phase_zyx, (1, 2, 0)))
    np.testing.assert_array_equal(result, direct)


def test_motion_keeps_raw_signal_and_rejects_implausible_proposals() -> None:
    """Motion output remains raw signal and the QC gate can retain identity."""
    fixed = _structured_volume()
    moving = ndimage.shift(
        fixed,
        shift=(2.0, -1.0, 0.0),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    corrected, _, accepted = correct_phase(
        fixed,
        moving,
        support=np.ones_like(fixed, dtype=bool),
        spacing_xyz_mm=np.ones(3),
        settings=MotionSettings(downsample_xyz=(1, 1, 1), upsample_factor=10),
    )
    assert accepted["transform_accepted"] is True
    assert float(accepted["corr_delta"]) >= 0.0
    assert np.all(corrected >= 0)
    assert np.mean(np.abs(corrected - fixed)) < np.mean(np.abs(moving - fixed))

    retained, _, rejected = correct_phase(
        fixed,
        moving,
        support=np.ones_like(fixed, dtype=bool),
        spacing_xyz_mm=np.ones(3),
        settings=MotionSettings(
            downsample_xyz=(1, 1, 1),
            upsample_factor=10,
            minimum_correlation_delta=2.0,
        ),
    )
    assert rejected["transform_accepted"] is False
    np.testing.assert_array_equal(retained, moving)


def test_corrected_pixel_spacing_produces_expected_one_mm_grid() -> None:
    """Bracketed DICOM PixelSpacing must yield 300x300x175, not a 1-mm fallback."""
    target = isotropic_geometry(_geometry())
    assert target.shape_zyx == (175, 300, 300)
    assert target.spacing_xyz_mm == (1.0, 1.0, 1.0)


def test_identity_mapping_uses_physical_coordinates_not_array_indices() -> None:
    """HR skeleton points map into UFAST through the shared DICOM frame."""
    hr = _geometry(
        shape_zyx=(10, 12, 14),
        spacing_xyz=(0.5, 0.5, 1.0),
        series_uid="hr",
    )
    ufast = _geometry(
        shape_zyx=(10, 6, 7),
        spacing_xyz=(1.0, 1.0, 1.0),
        series_uid="ufast",
    )
    skeleton = np.zeros(hr.shape_zyx, dtype=np.uint8)
    skeleton[4, 4, 6] = 1
    mapped, metrics = rasterize_skeleton_identity(skeleton, hr, ufast)
    assert mapped[4, 2, 3] == 1
    assert metrics["fraction_inside_target"] == 1.0
    checks = geometry_alignment_checks(hr, ufast)
    assert checks["same_frame_of_reference_uid"] is True
    assert checks["direction_equal"] is True


def test_motion_shift_is_composed_into_one_spatial_resample() -> None:
    """A positive target-index shift moves content without a second interpolation."""
    geometry = _geometry(shape_zyx=(7, 7, 7), spacing_xyz=(1.0, 1.0, 1.0))
    source = np.zeros(geometry.shape_zyx, dtype=np.float32)
    source[3, 3, 3] = 1.0
    shifted = resample_to_geometry(
        source,
        geometry,
        geometry,
        output_shift_xyz_voxels=np.asarray([1.0, 0.0, 0.0]),
    )
    assert shifted[3, 3, 4] == 1.0
    assert int(np.count_nonzero(shifted)) == 1


def test_interseries_alignment_qc_uses_all_baselines_and_flags_translation() -> None:
    """Frame identity is supplemented by an image-content translation diagnostic."""
    geometry = _geometry(
        shape_zyx=(TEST_VOLUME_SIZE,) * 3,
        spacing_xyz=(1.0, 1.0, 1.0),
    )
    baseline = _structured_volume()
    ufast = SimpleNamespace(
        geometry=geometry,
        signal_tzyx=np.stack([baseline, baseline, baseline, baseline * 1.2]),
    )
    aligned = SimpleNamespace(geometry=geometry, signal_tzyx=baseline[None])
    aligned_qc = _identity_alignment_qc(
        aligned, ufast, baseline_frame_count=ALIGNMENT_BASELINE_COUNT
    )
    assert aligned_qc["baseline_frame_count"] == ALIGNMENT_BASELINE_COUNT
    assert aligned_qc["status"] == "pass"

    displaced = ndimage.shift(
        baseline,
        shift=(0.0, 0.0, 4.0),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    displaced_hr = SimpleNamespace(geometry=geometry, signal_tzyx=displaced[None])
    displaced_qc = _identity_alignment_qc(
        displaced_hr, ufast, baseline_frame_count=ALIGNMENT_BASELINE_COUNT
    )
    assert displaced_qc["status"] == "review_required"
    assert float(displaced_qc["proposed_translation_norm_mm"]) > float(
        displaced_qc["review_translation_threshold_mm"]
    )


def test_pair_validation_allows_different_array_directions_in_shared_frame() -> None:
    """Physical mapping needs a shared frame, not identical voxel-axis directions."""
    hr = _geometry(series_uid="hr")
    ufast = DicomGeometry(
        series_instance_uid="ufast",
        frame_of_reference_uid=hr.frame_of_reference_uid,
        shape_zyx=hr.shape_zyx,
        spacing_xyz_mm=hr.spacing_xyz_mm,
        origin_lps_mm=hr.origin_lps_mm,
        direction_lps=(0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        slice_thickness_mm=hr.slice_thickness_mm,
    )
    record = SimpleNamespace(ufast_baseline_frame_count=1)
    checks = _validate_pair(
        record,
        SimpleNamespace(geometry=hr),
        SimpleNamespace(geometry=ufast, signal_tzyx=np.zeros((2, 1, 1, 1))),
    )
    assert checks["same_frame_of_reference_uid"] is True
    assert checks["direction_equal"] is False


def test_case_manifest_requires_explicit_series_uids_and_baseline_count(
    tmp_path: Path,
) -> None:
    """Cohort membership and series selection are explicit Vanguard inputs."""
    path = tmp_path / "cases.csv"
    row = {
        "exam_id": "case",
        "dataset": "uch_nac",
        "study_instance_uid": "study",
        "hr_series_instance_uid": "hr",
        "ufast_series_instance_uid": "ufast",
        "ufast_baseline_frame_count": str(EXPECTED_BASELINE_COUNT),
    }
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    record = read_case_manifest(path)[0]
    assert record.ufast_baseline_frame_count == EXPECTED_BASELINE_COUNT


def test_owned_output_layout_exposes_physical_ufast_times(tmp_path: Path) -> None:
    """The downstream GNN uses physical seconds from the owned pipeline sidecar."""
    case_id = "uchicago_case"
    case_root = tmp_path / case_id
    images = case_root
    images.mkdir(parents=True)
    times = np.asarray([0.0, 9.5, 21.0], dtype=np.float64)
    np.save(case_root / "ufast_times_seconds.npy", times)
    for index in range(times.size):
        (images / f"{case_id}_{index:04d}.nii.gz").touch()

    paths = discover_raw_dce_paths(tmp_path, case_id, [0, 1, 2])
    assert paths == [images / f"{case_id}_{index:04d}.nii.gz" for index in range(3)]
    np.testing.assert_array_equal(
        load_raw_dce_times(tmp_path, case_id, [0, 1, 2]), times
    )
