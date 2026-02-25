"""Smoke tests for 4D skeleton extraction."""

import numpy as np
import pytest

from graph_extraction.skeleton4d import skeletonize4d


def test_skeleton4d_basic_smoke() -> None:
    """Test that skeleton4d runs without crashing on small synthetic data."""
    # Create small 4D volume: (t=3, z=10, y=10, x=10)
    priority = np.random.rand(3, 10, 10, 10).astype(np.float32)

    result = skeletonize4d(priority, threshold_low=0.5)

    assert result.shape == priority.shape
    assert result.dtype == bool


def test_skeleton4d_empty_input() -> None:
    """Test that skeleton4d handles empty input (all below threshold)."""
    priority = np.zeros((3, 10, 10, 10), dtype=np.float32)

    result = skeletonize4d(priority, threshold_low=0.5)

    assert result.shape == priority.shape
    assert not np.any(result)  # Should be all False


def test_skeleton4d_with_anchors() -> None:
    """Test that skeleton4d runs with anchor constraints."""
    priority = np.random.rand(3, 10, 10, 10).astype(np.float32)

    result = skeletonize4d(
        priority,
        threshold_low=0.3,
        threshold_high=0.8,
        min_anchor_fraction=0.01,
        min_anchor_voxels=5,
    )

    assert result.shape == priority.shape
    assert result.dtype == bool


def test_skeleton4d_temporal_connectivity() -> None:
    """Test that skeleton4d respects temporal connectivity parameters."""
    priority = np.random.rand(5, 8, 8, 8).astype(np.float32)

    result = skeletonize4d(
        priority,
        threshold_low=0.4,
        max_temporal_radius=2,
        min_voxels_per_timepoint=10,
    )

    assert result.shape == priority.shape
    assert result.dtype == bool


def test_skeleton4d_invalid_ndim() -> None:
    """Test that skeleton4d raises error on wrong dimensionality."""
    priority = np.random.rand(10, 10, 10).astype(np.float32)  # 3D, not 4D

    with pytest.raises(ValueError, match="must be 4D"):
        skeletonize4d(priority, threshold_low=0.5)


def test_skeleton4d_invalid_params() -> None:
    """Test that skeleton4d validates parameters."""
    priority = np.random.rand(3, 10, 10, 10).astype(np.float32)

    with pytest.raises(ValueError):
        skeletonize4d(priority, threshold_low=0.5, min_voxels_per_timepoint=-1)

    with pytest.raises(ValueError):
        skeletonize4d(priority, threshold_low=0.5, max_candidates=0)
