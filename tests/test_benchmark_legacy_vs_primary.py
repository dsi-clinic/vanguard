"""Unit tests for legacy-vs-primary benchmark helpers."""

from __future__ import annotations

from pathlib import Path

from batch_processing.benchmark_legacy_vs_primary import (
    extract_metrics,
    resolve_visualization_path,
)


def test_resolve_visualization_path_accepts_legacy_vs_primary_key(
    tmp_path: Path,
) -> None:
    """Legacy-vs-primary summaries should resolve the rotating MP4 output key."""
    mp4_path = tmp_path / "rotation_compare_legacy_vs_primary_3d.mp4"
    summary = {
        "outputs": {
            "rotation_compare_legacy_vs_primary_mp4": str(mp4_path),
        }
    }

    resolved = resolve_visualization_path(summary, tmp_path)

    assert resolved == str(mp4_path)


def test_resolve_visualization_path_resolves_relative_paths(tmp_path: Path) -> None:
    """Relative output paths should be anchored to the study artifacts directory."""
    relative_path = "rotation_compare_4d_and_all_3d.mp4"
    summary = {"outputs": {"rotation_compare_4d_and_all_3d_mp4": relative_path}}

    resolved = resolve_visualization_path(summary, tmp_path)

    assert resolved == str(tmp_path / relative_path)


def test_extract_metrics_supports_legacy_vs_primary_summary_shape() -> None:
    """Metric extraction should support keys emitted by legacy-vs-primary compare."""
    expected_legacy_voxels = 11
    expected_primary_voxels = 17
    expected_legacy_components = 2
    expected_primary_components = 3
    expected_overlap_voxels = 9
    expected_overlap_dice = 0.75
    expected_overlap_jaccard = 0.6
    expected_elapsed_seconds = 12.5

    summary = {
        "legacy": {"voxels": expected_legacy_voxels, "components_26": expected_legacy_components},
        "primary": {"voxels": expected_primary_voxels, "components_26": expected_primary_components},
        "overlap": {
            "intersection_voxels": expected_overlap_voxels,
            "dice": expected_overlap_dice,
            "jaccard": expected_overlap_jaccard,
        },
        "timing_seconds": {"total": expected_elapsed_seconds},
    }

    metrics = extract_metrics(summary)

    assert metrics["legacy_voxels"] == expected_legacy_voxels
    assert metrics["primary_voxels"] == expected_primary_voxels
    assert metrics["legacy_components_26"] == expected_legacy_components
    assert metrics["primary_components_26"] == expected_primary_components
    assert metrics["overlap_voxels"] == expected_overlap_voxels
    assert metrics["overlap_dice"] == expected_overlap_dice
    assert metrics["overlap_jaccard"] == expected_overlap_jaccard
    assert metrics["compare_elapsed_seconds"] == expected_elapsed_seconds
