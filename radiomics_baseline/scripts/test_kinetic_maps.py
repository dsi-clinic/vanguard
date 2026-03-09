#!/usr/bin/env python3
"""Unit tests for kinetic parameter map generation using synthetic data.

Run with::

    python -m pytest radiomics_baseline/scripts/test_kinetic_maps.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk
from generate_kinetic_maps import (
    KINETIC_MAP_NAMES,
    SUBTRACTION_MAP_NAMES,
    compute_enhancement_series,
    compute_kinetic_maps,
    compute_subtraction_maps,
    discover_phases,
    find_tumor_peak_phase,
    generate_maps_for_pid,
    sanitize_map,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_volume(
    shape: tuple[int, ...], value: float, spacing: tuple = (1.0, 1.0, 1.0)
) -> sitk.Image:
    """Create a constant SimpleITK Image."""
    arr = np.full(shape, value, dtype=np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    return img


def _write_phase(
    patient_dir: Path,
    pid: str,
    phase_idx: int,
    value: float,
    shape: tuple,
    spacing: tuple = (1.0, 1.0, 1.0),
) -> None:
    """Write a constant-valued phase NIfTI to disk."""
    img = _make_volume(shape, value, spacing)
    sitk.WriteImage(img, str(patient_dir / f"{pid}_{phase_idx:04d}.nii.gz"))


def _write_mask(
    masks_dir: Path, pid: str, mask_arr: np.ndarray, spacing: tuple = (1.0, 1.0, 1.0)
) -> None:
    """Write a binary mask NIfTI to disk."""
    img = sitk.GetImageFromArray(mask_arr.astype(np.uint8))
    img.SetSpacing(spacing)
    sitk.WriteImage(img, str(masks_dir / f"{pid}.nii.gz"))


# ---------------------------------------------------------------------------
# Tests: compute_enhancement_series
# ---------------------------------------------------------------------------


class TestComputeEnhancementSeries:
    def test_basic_subtraction(self):
        pre = np.ones((4, 4, 4)) * 100
        post1 = np.ones((4, 4, 4)) * 150
        post2 = np.ones((4, 4, 4)) * 200
        result = compute_enhancement_series([post1, post2], pre)
        assert len(result) == 2
        np.testing.assert_array_equal(result[0], 50)
        np.testing.assert_array_equal(result[1], 100)

    def test_negative_enhancement(self):
        pre = np.ones((4, 4, 4)) * 200
        post1 = np.ones((4, 4, 4)) * 150
        result = compute_enhancement_series([post1], pre)
        np.testing.assert_array_equal(result[0], -50)

    def test_single_phase(self):
        pre = np.zeros((2, 2, 2))
        post = np.ones((2, 2, 2)) * 42
        result = compute_enhancement_series([post], pre)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], 42)


# ---------------------------------------------------------------------------
# Tests: find_tumor_peak_phase
# ---------------------------------------------------------------------------


class TestFindTumorPeakPhase:
    def test_peak_at_second_phase(self):
        mask = np.zeros((4, 4, 4), dtype=np.uint8)
        mask[1:3, 1:3, 1:3] = 1

        E1 = np.ones((4, 4, 4)) * 50
        E2 = np.ones((4, 4, 4)) * 100  # peak
        E3 = np.ones((4, 4, 4)) * 80

        peak = find_tumor_peak_phase([E1, E2, E3], [1, 2, 3], mask)
        assert peak == 1  # list index 1 → phase index 2

    def test_single_phase(self):
        mask = np.ones((2, 2, 2), dtype=np.uint8)
        E1 = np.ones((2, 2, 2)) * 50
        peak = find_tumor_peak_phase([E1], [1], mask)
        assert peak == 0

    def test_peak_at_first_phase(self):
        mask = np.ones((3, 3, 3), dtype=np.uint8)
        E1 = np.ones((3, 3, 3)) * 200  # peak
        E2 = np.ones((3, 3, 3)) * 100
        E3 = np.ones((3, 3, 3)) * 50
        peak = find_tumor_peak_phase([E1, E2, E3], [1, 2, 3], mask)
        assert peak == 0

    def test_peak_at_last_phase(self):
        mask = np.ones((3, 3, 3), dtype=np.uint8)
        E1 = np.ones((3, 3, 3)) * 50
        E2 = np.ones((3, 3, 3)) * 100
        E3 = np.ones((3, 3, 3)) * 200  # peak = last
        peak = find_tumor_peak_phase([E1, E2, E3], [1, 2, 3], mask)
        assert peak == 2


# ---------------------------------------------------------------------------
# Tests: compute_kinetic_maps
# ---------------------------------------------------------------------------


class TestComputeKineticMaps:
    def test_two_phases_peak_early(self):
        """Peak at first post-contrast; washout follows."""
        shape = (4, 4, 4)
        mask = np.ones(shape, dtype=np.uint8)
        pre = np.zeros(shape)
        post1 = np.ones(shape) * 100  # peak
        post2 = np.ones(shape) * 80  # washout

        maps = compute_kinetic_maps(pre, [post1, post2], [1, 2], mask)
        assert set(KINETIC_MAP_NAMES).issubset(maps.keys())

        # E_early = I_1 - I_0 = 100
        np.testing.assert_array_almost_equal(maps["E_early"], 100)
        # E_peak = 100 (peak at phase 1)
        np.testing.assert_array_almost_equal(maps["E_peak"], 100)
        # slope_in = E_early / 1 = 100
        np.testing.assert_array_almost_equal(maps["slope_in"], 100)
        # slope_out = (I_2 - I_1) / (2 - 1) = -20
        np.testing.assert_array_almost_equal(maps["slope_out"], -20)
        # AUC = trap(0→1: 0+100)/2*1 + trap(1→2: 100+80)/2*1 = 50 + 90 = 140
        np.testing.assert_array_almost_equal(maps["AUC"], 140)

    def test_slope_out_zero_when_peak_is_last(self):
        shape = (4, 4, 4)
        mask = np.ones(shape, dtype=np.uint8)
        pre = np.zeros(shape)
        post1 = np.ones(shape) * 50
        post2 = np.ones(shape) * 100  # peak = last

        maps = compute_kinetic_maps(pre, [post1, post2], [1, 2], mask)
        np.testing.assert_array_equal(maps["slope_out"], 0)

    def test_single_post_contrast(self):
        shape = (4, 4, 4)
        mask = np.ones(shape, dtype=np.uint8)
        pre = np.zeros(shape)
        post1 = np.ones(shape) * 100

        maps = compute_kinetic_maps(pre, [post1], [1], mask)
        # slope_out = 0 (peak is last and only phase)
        np.testing.assert_array_equal(maps["slope_out"], 0)
        # AUC = trapezoid from t=0 (E=0) to t=1 (E=100) = 50
        np.testing.assert_array_almost_equal(maps["AUC"], 50)

    def test_three_phases_monotonic_increase(self):
        """Monotonically increasing: peak at last phase, slope_out=0."""
        shape = (3, 3, 3)
        mask = np.ones(shape, dtype=np.uint8)
        pre = np.zeros(shape)
        phases = [np.ones(shape) * v for v in [50, 100, 150]]

        maps = compute_kinetic_maps(pre, phases, [1, 2, 3], mask)
        np.testing.assert_array_almost_equal(maps["E_peak"], 150)
        np.testing.assert_array_equal(maps["slope_out"], 0)

    def test_tpeak_voxel_generated_with_4_phases(self):
        shape = (4, 4, 4)
        mask = np.ones(shape, dtype=np.uint8)
        pre = np.zeros(shape)
        phases = [np.ones(shape) * v for v in [50, 100, 80, 60]]

        maps = compute_kinetic_maps(
            pre, phases, [1, 2, 3, 4], mask, generate_tpeak_voxel=True
        )
        assert "t_peak_voxel" in maps
        # All voxels have same enhancement curve → peak at list index 1 → phase index 2
        np.testing.assert_array_equal(maps["t_peak_voxel"], 2)

    def test_tpeak_voxel_not_generated_with_3_phases(self):
        shape = (4, 4, 4)
        mask = np.ones(shape, dtype=np.uint8)
        pre = np.zeros(shape)
        phases = [np.ones(shape) * v for v in [50, 100, 80]]

        maps = compute_kinetic_maps(
            pre, phases, [1, 2, 3], mask, generate_tpeak_voxel=True
        )
        assert "t_peak_voxel" not in maps

    def test_auc_trapezoidal(self):
        """Verify AUC against manual trapezoidal calculation."""
        shape = (2, 2, 2)
        mask = np.ones(shape, dtype=np.uint8)
        pre = np.zeros(shape)
        # E values at t=1,2,3: 60, 120, 90
        phases = [np.ones(shape) * v for v in [60, 120, 90]]

        maps = compute_kinetic_maps(pre, phases, [1, 2, 3], mask)
        # AUC = 0.5*(0+60)*1 + 0.5*(60+120)*1 + 0.5*(120+90)*1
        #     = 30 + 90 + 105 = 225
        np.testing.assert_array_almost_equal(maps["AUC"], 225)


# ---------------------------------------------------------------------------
# Tests: compute_subtraction_maps
# ---------------------------------------------------------------------------


class TestComputeSubtractionMaps:
    def test_basic_wash_in_wash_out(self):
        """wash_in = peak - early; wash_out = last - peak."""
        shape = (4, 4, 4)
        post1 = np.ones(shape) * 100  # early
        post2 = np.ones(shape) * 200  # peak
        post3 = np.ones(shape) * 150  # last (washout)

        maps = compute_subtraction_maps([post1, post2, post3], peak_list_idx=1)
        assert set(SUBTRACTION_MAP_NAMES) == set(maps.keys())
        np.testing.assert_array_almost_equal(maps["wash_in"], 100)  # 200 - 100
        np.testing.assert_array_almost_equal(maps["wash_out"], -50)  # 150 - 200

    def test_peak_at_first_phase_wash_in_zero(self):
        """When peak = first post-contrast, wash_in should be zero everywhere."""
        shape = (3, 3, 3)
        post1 = np.ones(shape) * 200  # peak = early
        post2 = np.ones(shape) * 150
        post3 = np.ones(shape) * 100

        maps = compute_subtraction_maps([post1, post2, post3], peak_list_idx=0)
        np.testing.assert_array_equal(maps["wash_in"], 0)

    def test_persistent_pattern_positive_wash_out(self):
        """Monotonically increasing: wash_out > 0 (persistent, BI-RADS type I)."""
        shape = (2, 2, 2)
        post1 = np.ones(shape) * 50
        post2 = np.ones(shape) * 100
        post3 = np.ones(shape) * 150  # peak = last

        maps = compute_subtraction_maps([post1, post2, post3], peak_list_idx=2)
        np.testing.assert_array_almost_equal(maps["wash_out"], 0)  # last == peak

    def test_output_dtype_float32(self):
        shape = (4, 4, 4)
        post1 = np.ones(shape, dtype=np.float64) * 100
        post2 = np.ones(shape, dtype=np.float64) * 200
        maps = compute_subtraction_maps([post1, post2], peak_list_idx=1)
        assert maps["wash_in"].dtype == np.float32
        assert maps["wash_out"].dtype == np.float32

    def test_two_phases_only(self):
        """With only 2 post-contrast phases: peak must be one of them."""
        shape = (2, 2, 2)
        post1 = np.ones(shape) * 100
        post2 = np.ones(shape) * 180  # peak = last

        maps = compute_subtraction_maps([post1, post2], peak_list_idx=1)
        np.testing.assert_array_almost_equal(maps["wash_in"], 80)  # 180 - 100
        np.testing.assert_array_almost_equal(maps["wash_out"], 0)  # 180 - 180


# ---------------------------------------------------------------------------
# Tests: sanitize_map
# ---------------------------------------------------------------------------


class TestSanitizeMap:
    def test_nan_replaced(self):
        arr = np.array([1.0, np.nan, 3.0])
        mask = np.ones(3, dtype=np.uint8)
        result = sanitize_map(arr, mask)
        np.testing.assert_array_equal(result, [1.0, 0.0, 3.0])

    def test_inf_replaced(self):
        arr = np.array([1.0, np.inf, -np.inf])
        mask = np.ones(3, dtype=np.uint8)
        result = sanitize_map(arr, mask)
        np.testing.assert_array_equal(result, [1.0, 0.0, 0.0])

    def test_outside_mask_zeroed(self):
        arr = np.array([100.0, 200.0, 300.0])
        mask = np.array([1, 0, 1], dtype=np.uint8)
        result = sanitize_map(arr, mask)
        np.testing.assert_array_equal(result, [100.0, 0.0, 300.0])

    def test_combined(self):
        arr = np.array([np.nan, 5.0, np.inf, 10.0])
        mask = np.array([1, 0, 1, 1], dtype=np.uint8)
        result = sanitize_map(arr, mask)
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0, 10.0])


# ---------------------------------------------------------------------------
# Tests: discover_phases
# ---------------------------------------------------------------------------


class TestDiscoverPhases:
    def test_finds_all_phases(self, tmp_path):
        pid = "TEST_001"
        patient_dir = tmp_path / pid
        patient_dir.mkdir()

        shape = (4, 4, 4)
        for idx in [0, 1, 2, 3]:
            _write_phase(patient_dir, pid, idx, float(idx * 100), shape)

        phases = discover_phases(str(tmp_path), pid)
        assert len(phases) == 4
        assert [i for i, _ in phases] == [0, 1, 2, 3]

    def test_missing_patient_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            discover_phases(str(tmp_path), "NONEXISTENT")

    def test_no_phase_files(self, tmp_path):
        pid = "EMPTY"
        (tmp_path / pid).mkdir()
        with pytest.raises(FileNotFoundError):
            discover_phases(str(tmp_path), pid)

    def test_ignores_kinetic_files(self, tmp_path):
        """Kinetic map files should not be detected as phases."""
        pid = "TEST_002"
        patient_dir = tmp_path / pid
        patient_dir.mkdir()

        shape = (4, 4, 4)
        _write_phase(patient_dir, pid, 0, 0.0, shape)
        _write_phase(patient_dir, pid, 1, 100.0, shape)

        # Write a kinetic map file that should NOT be picked up
        img = _make_volume(shape, 50.0)
        sitk.WriteImage(img, str(patient_dir / f"{pid}_kinetic_E_early.nii.gz"))

        phases = discover_phases(str(tmp_path), pid)
        assert len(phases) == 2  # only _0000 and _0001


# ---------------------------------------------------------------------------
# Tests: end-to-end generate_maps_for_pid
# ---------------------------------------------------------------------------


class TestGenerateMapsForPid:
    def test_synthetic_3_post_contrast(self, tmp_path):
        pid = "TEST_001"
        images_dir = tmp_path / "images"
        patient_dir = images_dir / pid
        patient_dir.mkdir(parents=True)
        masks_dir = tmp_path / "masks"
        masks_dir.mkdir()

        shape = (8, 16, 16)

        # Pre-contrast (100), 3 post-contrast (200, 300, 250)
        _write_phase(patient_dir, pid, 0, 100.0, shape)
        _write_phase(patient_dir, pid, 1, 200.0, shape)
        _write_phase(patient_dir, pid, 2, 300.0, shape)
        _write_phase(patient_dir, pid, 3, 250.0, shape)

        # Mask
        mask_arr = np.zeros(shape, dtype=np.uint8)
        mask_arr[2:6, 4:12, 4:12] = 1
        _write_mask(masks_dir, pid, mask_arr)

        result = generate_maps_for_pid(
            pid=pid,
            images_dir=str(images_dir),
            masks_dir=str(masks_dir),
            mask_pattern="{pid}.nii.gz",
            overwrite=True,
        )

        assert result["status"] == "success"
        assert result["maps_generated"] == 5

        # Verify all kinetic maps exist
        for name in KINETIC_MAP_NAMES:
            path = patient_dir / f"{pid}_kinetic_{name}.nii.gz"
            assert path.exists(), f"Missing: {path}"

        # Verify map values
        E_early = sitk.GetArrayFromImage(
            sitk.ReadImage(str(patient_dir / f"{pid}_kinetic_E_early.nii.gz"))
        )
        # Inside mask: 200 - 100 = 100, outside: 0
        assert E_early[3, 8, 8] == pytest.approx(100.0)
        assert E_early[0, 0, 0] == pytest.approx(0.0)  # outside mask

    def test_skip_when_exists(self, tmp_path):
        pid = "TEST_002"
        images_dir = tmp_path / "images"
        patient_dir = images_dir / pid
        patient_dir.mkdir(parents=True)
        masks_dir = tmp_path / "masks"
        masks_dir.mkdir()

        shape = (4, 4, 4)
        _write_phase(patient_dir, pid, 0, 0.0, shape)
        _write_phase(patient_dir, pid, 1, 100.0, shape)

        mask_arr = np.ones(shape, dtype=np.uint8)
        _write_mask(masks_dir, pid, mask_arr)

        # First run: generate
        r1 = generate_maps_for_pid(
            pid=pid,
            images_dir=str(images_dir),
            masks_dir=str(masks_dir),
            mask_pattern="{pid}.nii.gz",
            overwrite=True,
        )
        assert r1["status"] == "success"

        # Second run: skip
        r2 = generate_maps_for_pid(
            pid=pid,
            images_dir=str(images_dir),
            masks_dir=str(masks_dir),
            mask_pattern="{pid}.nii.gz",
            overwrite=False,
        )
        assert r2["status"] == "skipped"

    def test_error_no_precontrast(self, tmp_path):
        pid = "TEST_003"
        images_dir = tmp_path / "images"
        patient_dir = images_dir / pid
        patient_dir.mkdir(parents=True)
        masks_dir = tmp_path / "masks"
        masks_dir.mkdir()

        shape = (4, 4, 4)
        # Only post-contrast, no _0000
        _write_phase(patient_dir, pid, 1, 100.0, shape)

        mask_arr = np.ones(shape, dtype=np.uint8)
        _write_mask(masks_dir, pid, mask_arr)

        result = generate_maps_for_pid(
            pid=pid,
            images_dir=str(images_dir),
            masks_dir=str(masks_dir),
            mask_pattern="{pid}.nii.gz",
            overwrite=True,
        )
        assert result["status"] == "error"

    def test_error_missing_patient_dir(self, tmp_path):
        """Patient directory doesn't exist — should return error, not crash."""
        pid = "NONEXISTENT_001"
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        masks_dir = tmp_path / "masks"
        masks_dir.mkdir()

        result = generate_maps_for_pid(
            pid=pid,
            images_dir=str(images_dir),
            masks_dir=str(masks_dir),
            mask_pattern="{pid}.nii.gz",
            overwrite=True,
        )
        assert result["status"] == "error"
        assert result["maps_generated"] == 0

    def test_error_missing_mask(self, tmp_path):
        """Phase images exist but mask is missing — should return error, not crash."""
        pid = "TEST_NOMASK"
        images_dir = tmp_path / "images"
        patient_dir = images_dir / pid
        patient_dir.mkdir(parents=True)
        masks_dir = tmp_path / "masks"
        masks_dir.mkdir()

        shape = (4, 4, 4)
        _write_phase(patient_dir, pid, 0, 50.0, shape)
        _write_phase(patient_dir, pid, 1, 100.0, shape)
        # No mask written

        result = generate_maps_for_pid(
            pid=pid,
            images_dir=str(images_dir),
            masks_dir=str(masks_dir),
            mask_pattern="{pid}.nii.gz",
            overwrite=True,
        )
        assert result["status"] == "error"
        assert result["maps_generated"] == 0
        assert "Mask not found" in result["error"]

    def test_subtraction_maps_generated(self, tmp_path):
        """--generate-subtraction produces wash_in and wash_out NIfTI files."""
        pid = "TEST_SUB"
        images_dir = tmp_path / "images"
        patient_dir = images_dir / pid
        patient_dir.mkdir(parents=True)
        masks_dir = tmp_path / "masks"
        masks_dir.mkdir()

        shape = (8, 16, 16)
        # Pre-contrast=100, early=200, peak=300, last=250 (washout)
        _write_phase(patient_dir, pid, 0, 100.0, shape)
        _write_phase(patient_dir, pid, 1, 200.0, shape)
        _write_phase(patient_dir, pid, 2, 300.0, shape)
        _write_phase(patient_dir, pid, 3, 250.0, shape)

        mask_arr = np.zeros(shape, dtype=np.uint8)
        mask_arr[2:6, 4:12, 4:12] = 1
        _write_mask(masks_dir, pid, mask_arr)

        result = generate_maps_for_pid(
            pid=pid,
            images_dir=str(images_dir),
            masks_dir=str(masks_dir),
            mask_pattern="{pid}.nii.gz",
            generate_subtraction=True,
            overwrite=True,
        )

        assert result["status"] == "success"
        assert result["maps_generated"] == 7  # 5 kinetic + 2 subtraction

        for name in SUBTRACTION_MAP_NAMES:
            path = patient_dir / f"{pid}_subtraction_{name}.nii.gz"
            assert path.exists(), f"Missing: {path}"

        # wash_in = I_peak - I_early = 300 - 200 = 100 (inside mask)
        wash_in = sitk.GetArrayFromImage(
            sitk.ReadImage(str(patient_dir / f"{pid}_subtraction_wash_in.nii.gz"))
        )
        assert wash_in[3, 8, 8] == pytest.approx(100.0)
        assert wash_in[0, 0, 0] == pytest.approx(0.0)  # outside mask

        # wash_out = I_last - I_peak = 250 - 300 = -50 (inside mask)
        wash_out = sitk.GetArrayFromImage(
            sitk.ReadImage(str(patient_dir / f"{pid}_subtraction_wash_out.nii.gz"))
        )
        assert wash_out[3, 8, 8] == pytest.approx(-50.0)

    def test_subtraction_skipped_with_single_post_contrast(self, tmp_path):
        """Subtraction maps require >=2 post-contrast phases.

        Warns and skips gracefully.
        """
        pid = "TEST_SUB_SKIP"
        images_dir = tmp_path / "images"
        patient_dir = images_dir / pid
        patient_dir.mkdir(parents=True)
        masks_dir = tmp_path / "masks"
        masks_dir.mkdir()

        shape = (4, 4, 4)
        _write_phase(patient_dir, pid, 0, 0.0, shape)
        _write_phase(patient_dir, pid, 1, 100.0, shape)

        mask_arr = np.ones(shape, dtype=np.uint8)
        _write_mask(masks_dir, pid, mask_arr)

        import warnings as _warnings

        with _warnings.catch_warnings(record=True) as w:
            _warnings.simplefilter("always")
            result = generate_maps_for_pid(
                pid=pid,
                images_dir=str(images_dir),
                masks_dir=str(masks_dir),
                mask_pattern="{pid}.nii.gz",
                generate_subtraction=True,
                overwrite=True,
            )

        assert result["status"] == "success"
        assert result["maps_generated"] == 5  # kinetic only; subtraction skipped
        assert any("Subtraction" in str(warning.message) for warning in w)

    def test_with_tpeak_voxel(self, tmp_path):
        pid = "TEST_004"
        images_dir = tmp_path / "images"
        patient_dir = images_dir / pid
        patient_dir.mkdir(parents=True)
        masks_dir = tmp_path / "masks"
        masks_dir.mkdir()

        shape = (4, 4, 4)
        _write_phase(patient_dir, pid, 0, 0.0, shape)
        for idx, val in [(1, 50), (2, 100), (3, 80), (4, 60)]:
            _write_phase(patient_dir, pid, idx, float(val), shape)

        mask_arr = np.ones(shape, dtype=np.uint8)
        _write_mask(masks_dir, pid, mask_arr)

        result = generate_maps_for_pid(
            pid=pid,
            images_dir=str(images_dir),
            masks_dir=str(masks_dir),
            mask_pattern="{pid}.nii.gz",
            generate_tpeak_voxel=True,
            overwrite=True,
        )
        assert result["status"] == "success"
        assert result["maps_generated"] == 6  # 5 core + t_peak_voxel

        tpeak_path = patient_dir / f"{pid}_kinetic_t_peak_voxel.nii.gz"
        assert tpeak_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
