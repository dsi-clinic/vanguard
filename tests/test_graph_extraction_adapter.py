"""Tests for the Step 4 (graph-extraction) adapter seam.

Graph extraction discovers a *different* artifact than the raw-DCE-phase
contract ``DatasetAdapter.load_timepoints()`` already covers: per-timepoint
vessel-segmentation ``.npz`` files produced by the (upstream) segmentation
stage. ``DatasetAdapter.load_segmented_timepoints()`` is the new seam for that,
and ``graph_extraction.pipeline.run_study_pipeline`` takes an optional
``adapter`` that must be a no-op when absent (matching the Step 2/3 fallback
pattern).

Deliberately scoped to the discovery seam only, not a full pipeline run: the
rest of ``run_study_pipeline`` is heavy numerical compute (skeletonization,
TC4D) with no existing test scaffold to build a byte-identical fixture against
yet -- that full-pipeline parity gate is a follow-up, mirroring how each prior
step added its own validation gate incrementally.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cohorts import DatasetAdapter, MamaMiaDataset


def _write_segmentation_npz(
    path: Path, shape: tuple[int, int, int] = (2, 2, 2)
) -> None:
    """Write a minimal valid vessel-segmentation .npz file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, vessel=np.zeros(shape, dtype=np.float32))


def _make_segmentation_dir(root: Path, case_id: str, timepoints: list[int]) -> Path:
    """Create <root>/<case_id>/images/<case_id>_<tp>_vessel_segmentation.npz files."""
    images_dir = root / case_id / "images"
    for tp in timepoints:
        _write_segmentation_npz(
            images_dir / f"{case_id}_{tp:04d}_vessel_segmentation.npz"
        )
    return root


def test_base_load_segmented_timepoints_matches_discover_study_timepoints(
    tmp_path: Path,
) -> None:
    """The base adapter delegates to the exact function graph_extraction already uses."""
    from graph_extraction.core4d import discover_study_timepoints

    case_id = "DUKE_001"
    _make_segmentation_dir(tmp_path, case_id, [0, 1, 2])
    adapter = DatasetAdapter(root=tmp_path)

    via_adapter = adapter.load_segmented_timepoints(input_dir=tmp_path, case_id=case_id)
    via_function = discover_study_timepoints(input_dir=tmp_path, case_id=case_id)

    assert via_adapter == via_function
    assert via_adapter[1] == [0, 1, 2]


def test_mamamia_load_segmented_timepoints_needs_no_override(tmp_path: Path) -> None:
    """MamaMiaDataset uses the inherited base behavior unchanged (no override)."""
    case_id = "ISPY2_045"
    _make_segmentation_dir(tmp_path, case_id, [0, 3])
    adapter = MamaMiaDataset(cohort="ispy2", root=tmp_path)

    files, timepoints = adapter.load_segmented_timepoints(
        input_dir=tmp_path, case_id=case_id
    )

    assert timepoints == [0, 3]
    assert all(f.name.startswith(case_id) for f in files)


def test_load_segmented_timepoints_raises_when_none_found(tmp_path: Path) -> None:
    """Missing segmentation files raise the same error as the underlying function."""
    adapter = DatasetAdapter(root=tmp_path)
    (tmp_path / "DUKE_999").mkdir()

    with pytest.raises(ValueError, match="No candidate segmentation files"):
        adapter.load_segmented_timepoints(input_dir=tmp_path, case_id="DUKE_999")


def test_base_viz_flip_spec_default_matches_current_mamamia_constant() -> None:
    """The adapter default must match graph_extraction's hardcoded MAMA-MIA constant.

    A regression guard: if PROCESSING_VIZ_FLIP_SPEC ever changes, this fails
    loudly instead of silently producing different MIP visualizations for
    adapter=None vs. adapter=MamaMiaDataset(...) runs.
    """
    from graph_extraction.constants import PROCESSING_VIZ_FLIP_SPEC

    assert DatasetAdapter.viz_flip_spec == PROCESSING_VIZ_FLIP_SPEC
    assert MamaMiaDataset.viz_flip_spec == PROCESSING_VIZ_FLIP_SPEC
