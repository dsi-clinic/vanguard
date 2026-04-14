"""Tests for Deep Sets multi-regime point features (issue #120)."""

from __future__ import annotations

import unittest

import numpy as np

from build_deepsets_dataset import (
    DEEPSETS_FEATURE_BASELINE,
    DEEPSETS_FEATURE_GEOMETRY_TOPOLOGY,
    DEEPSETS_FEATURE_GEOMETRY_TOPOLOGY_DYNAMIC,
    _build_case_set,
    deepsets_point_feature_names,
)


def _empty_volume(shape: tuple[int, int, int]) -> np.ndarray:
    return np.zeros(shape, dtype=bool)


class TestDeepsetsPointFeatures(unittest.TestCase):
    """Unit tests for ``deepsets_point_feature_names`` and ``_build_case_set``."""

    def test_deepsets_point_feature_names_lengths(self) -> None:
        self.assertEqual(
            deepsets_point_feature_names(DEEPSETS_FEATURE_BASELINE), ["curvature_rad"]
        )
        self.assertEqual(
            len(deepsets_point_feature_names(DEEPSETS_FEATURE_GEOMETRY_TOPOLOGY)), 16
        )
        self.assertEqual(
            len(
                deepsets_point_feature_names(DEEPSETS_FEATURE_GEOMETRY_TOPOLOGY_DYNAMIC)
            ),
            27,
        )

    def test_deepsets_point_feature_names_unknown_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            deepsets_point_feature_names("not_a_regime")
        self.assertIn("Unknown deepsets_point_feature_set", str(ctx.exception))

    def test_build_case_set_baseline_matches_feature_order(self) -> None:
        shape = (5, 5, 5)
        skel = _empty_volume(shape)
        skel[2, 2, 2] = True
        tumor = _empty_volume(shape)
        tumor[0, 0, 0] = True
        spacing = (1.0, 1.0, 1.0)
        out = _build_case_set(
            case_id="toy",
            label=1,
            skeleton_mask_zyx=skel,
            tumor_mask_zyx=tumor,
            spacing_mm_zyx=spacing,
            local_radius_mm=100.0,
            tumor_equiv_radius_mm=1.0,
            point_feature_set=DEEPSETS_FEATURE_BASELINE,
            support_edt_mm_zyx=None,
            support_radius_available_scalar=0.0,
            signal_4d=None,
        )
        self.assertIsNotNone(out)
        assert out is not None
        self.assertEqual(out["feature_names"], ["curvature_rad"])
        self.assertEqual(tuple(out["x"].shape), (1, 1))

    def test_build_case_set_geometry_topology_inside_tumor(self) -> None:
        shape = (5, 5, 5)
        skel = _empty_volume(shape)
        skel[2, 2, 2] = True
        tumor = _empty_volume(shape)
        tumor[2, 2, 2] = True
        spacing = (1.0, 1.0, 1.0)
        out = _build_case_set(
            case_id="toy",
            label=0,
            skeleton_mask_zyx=skel,
            tumor_mask_zyx=tumor,
            spacing_mm_zyx=spacing,
            local_radius_mm=100.0,
            tumor_equiv_radius_mm=1.0,
            point_feature_set=DEEPSETS_FEATURE_GEOMETRY_TOPOLOGY,
            support_edt_mm_zyx=None,
            support_radius_available_scalar=0.0,
            signal_4d=None,
        )
        self.assertIsNotNone(out)
        assert out is not None
        names = deepsets_point_feature_names(DEEPSETS_FEATURE_GEOMETRY_TOPOLOGY)
        self.assertEqual(out["feature_names"], names)
        row = out["x"][0].numpy()
        idx = {n: i for i, n in enumerate(names)}
        self.assertAlmostEqual(float(row[idx["inside_tumor"]]), 1.0)
        self.assertAlmostEqual(float(row[idx["shell_0_2mm"]]), 0.0)
        self.assertAlmostEqual(float(row[idx["degree"]]), 0.0)

    def test_build_case_set_dynamic_toy_curve(self) -> None:
        shape = (3, 3, 3)
        skel = _empty_volume(shape)
        skel[1, 1, 1] = True
        tumor = _empty_volume(shape)
        tumor[0, 0, 0] = True
        spacing = (1.0, 1.0, 1.0)
        t, z, y, x = 4, 3, 3, 3
        signal_4d = np.zeros((t, z, y, x), dtype=np.float32)
        for ti in range(t):
            signal_4d[ti, 1, 1, 1] = float([0.05, 0.12, 0.30, 0.22][ti])
        out = _build_case_set(
            case_id="toy",
            label=1,
            skeleton_mask_zyx=skel,
            tumor_mask_zyx=tumor,
            spacing_mm_zyx=spacing,
            local_radius_mm=100.0,
            tumor_equiv_radius_mm=1.0,
            point_feature_set=DEEPSETS_FEATURE_GEOMETRY_TOPOLOGY_DYNAMIC,
            support_edt_mm_zyx=None,
            support_radius_available_scalar=0.0,
            signal_4d=signal_4d,
        )
        self.assertIsNotNone(out)
        assert out is not None
        names = deepsets_point_feature_names(DEEPSETS_FEATURE_GEOMETRY_TOPOLOGY_DYNAMIC)
        row = out["x"][0].numpy()
        idx = {n: i for i, n in enumerate(names)}
        self.assertAlmostEqual(float(row[idx["kinetic_signal_ok"]]), 1.0)
        self.assertAlmostEqual(float(row[idx["peak_enhancement"]]), 0.25, places=5)
        self.assertGreater(float(row[idx["positive_enhancement_auc"]]), 0.0)


if __name__ == "__main__":
    unittest.main()
