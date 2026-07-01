"""Synthetic smoke test for the GNN centerline dataset.

No cluster data is available in CI, so this fabricates a tiny centerline tree
(skeleton + support masks, fake per-timepoint ``vessel`` volumes, a
``run_summary.json`` and a labels CSV) and checks the :class:`Data` invariants
the GNN track relies on.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from gnn.data_loader import VanguardCenterlineDataset  # noqa: E402

VOLUME_SHAPE = (4, 8, 8)  # (z, y, x)
NUM_TIMEPOINTS = 3
SKELETON_Z = 2
SKELETON_Y = 3
SKELETON_XS = (1, 2, 3, 4, 5)  # 5 voxels -> 4 undirected edges


def _write_case(
    studies_dir: Path, case_id: str, *, num_timepoints: int = NUM_TIMEPOINTS
) -> None:
    """Write one synthetic case directory under ``studies_dir/NACT``."""
    case_dir = studies_dir / "NACT" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    skeleton = np.zeros(VOLUME_SHAPE, dtype=np.uint8)
    for x in SKELETON_XS:
        skeleton[SKELETON_Z, SKELETON_Y, x] = 1
    np.save(case_dir / f"{case_id}_skeleton_4d_exam_mask.npy", skeleton)

    # Support = a small neighborhood around the skeleton so the distance
    # transform yields positive radii at the skeleton voxels.
    support = np.zeros(VOLUME_SHAPE, dtype=np.uint8)
    support[
        SKELETON_Z - 1 : SKELETON_Z + 2,
        SKELETON_Y - 1 : SKELETON_Y + 2,
        min(SKELETON_XS) - 1 : max(SKELETON_XS) + 2,
    ] = 1
    np.save(case_dir / f"{case_id}_skeleton_4d_exam_support_mask.npy", support)

    # Monotonically rising contrast -> peak at the last timepoint everywhere.
    study_files: list[str] = []
    for t in range(num_timepoints):
        vessel = np.full(VOLUME_SHAPE, float(t), dtype=np.float32)
        npz_path = case_dir / f"{case_id}_{t:04d}_vessel_segmentation.npz"
        np.savez(npz_path, vessel=vessel)
        study_files.append(str(npz_path))

    (case_dir / "run_summary.json").write_text(
        json.dumps(
            {
                "study_files": study_files,
                "study_timepoints": list(range(num_timepoints)),
            }
        )
    )


@pytest.fixture
def centerline_tree(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Build a two-case tree: one labeled case and one with no label row."""
    studies = tmp_path / "studies"
    _write_case(studies, "NACT_01")
    _write_case(studies, "NACT_99")  # deliberately absent from the labels file

    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text("case_id,pcr\nNACT_01,1\n")

    cache_dir = tmp_path / "cache"
    return studies, labels_csv, cache_dir


def test_builds_labeled_graph_and_skips_unlabeled(
    centerline_tree: tuple[Path, Path, Path],
) -> None:
    """A labeled case yields a valid graph; an unlabeled case is dropped."""
    studies, labels_csv, cache_dir = centerline_tree

    dataset = VanguardCenterlineDataset(
        studies,
        labels_path=labels_csv,
        cache_dir=cache_dir,
        node_features=("peak_time", "radius"),
        profile=True,
    )

    # Only the labeled case survives; NACT_99 is skipped for lacking a label.
    assert len(dataset) == 1
    data = dataset[0]

    assert data.num_nodes == len(SKELETON_XS)
    assert data.x.shape == (len(SKELETON_XS), 2)
    assert data.pos.shape == (len(SKELETON_XS), 3)

    # 4 undirected edges -> 8 directed entries, and every edge has its reverse.
    assert data.edge_index.shape[1] == 2 * (len(SKELETON_XS) - 1)
    edges = {(int(u), int(v)) for u, v in data.edge_index.t().tolist()}
    assert all((v, u) in edges for (u, v) in edges)

    # Peak-contrast index stays within the valid timepoint range.
    assert int(data.peak_time.min()) >= 0
    assert int(data.peak_time.max()) <= NUM_TIMEPOINTS - 1

    assert data.y.tolist() == [1]
    assert data.case_id == "NACT_01"
    assert data.dataset == "NACT"

    # Per-case graph artifact is emitted alongside the collated cache.
    assert (cache_dir / "processed" / "NACT_01_graph.pt").exists()


def test_missing_labels_path_raises(tmp_path: Path) -> None:
    """Labels are mandatory: a missing path fails fast."""
    with pytest.raises(ValueError, match="labels_path is required"):
        VanguardCenterlineDataset(tmp_path, labels_path=None)


def test_segment_mode_not_implemented(tmp_path: Path) -> None:
    """The ``segment`` node mode is an explicit, unimplemented extension point."""
    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text("case_id,pcr\nNACT_01,1\n")
    with pytest.raises(NotImplementedError):
        VanguardCenterlineDataset(tmp_path, labels_path=labels_csv, node_mode="segment")
