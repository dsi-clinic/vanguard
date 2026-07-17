"""Synthetic smoke test for the GNN centerline dataset.

No cluster data is available in CI, so this fabricates a tiny centerline tree
(skeleton + support masks, a ``run_summary.json``, a raw DCE-MRI NIfTI series,
and a labels CSV) and checks the :class:`Data` invariants the GNN track relies
on. The raw DCE series (not the vessel-segmentation NPZ timepoints) is what the
kinetic node features are sampled from -- see ``gnn/README.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
sitk = pytest.importorskip("SimpleITK")

from gnn.data_loader import (  # noqa: E402
    VanguardCenterlineDataset,
    _node_kinetic_features,
    _time_axis_from_study_timepoints,
)

VOLUME_SHAPE = (4, 8, 8)  # (z, y, x)
NUM_TIMEPOINTS = 3
SKELETON_Z = 2
SKELETON_Y = 3
SKELETON_XS = (1, 2, 3, 4, 5)  # 5 voxels -> 4 undirected edges

# curve = t for every voxel -> baseline=0, enhancement=[0, 1, 2]:
# peak at the last timepoint (idx 2), arrival (20% of peak=2 -> threshold 0.4)
# first crossed at idx 1.
EXPECTED_PEAK_IDX = NUM_TIMEPOINTS - 1
EXPECTED_TTE_IDX = 1
EXPECTED_PEAK_ENHANCEMENT = 2.0
PROTOCOL_EXPECTED_PEAK_IDX = 3
PROTOCOL_EXPECTED_TTE_IDX = 2


def test_protocol_kinetics_use_baseline_mean_relative_signal_and_seconds() -> None:
    """UFAST features honor all baselines and the irregular physical time axis."""
    curve = np.asarray([10.0, 12.0, 16.5, 22.0], dtype=np.float32)
    times = np.asarray([0.0, 5.0, 50.0, 55.0], dtype=np.float64)
    result = _node_kinetic_features(
        curve,
        times,
        baseline_frame_count=2,
        relative_enhancement=True,
    )
    assert result["baseline_signal"] == pytest.approx(11.0)
    assert result["peak_idx"] == PROTOCOL_EXPECTED_PEAK_IDX
    assert result["peak_time_seconds"] == pytest.approx(55.0)
    assert result["tte_idx"] == PROTOCOL_EXPECTED_TTE_IDX
    assert result["tte_seconds"] == pytest.approx(50.0)
    assert result["peak_enhancement"] == pytest.approx(1.0)
    assert result["washin_slope"] == pytest.approx(0.1)
    assert result["auc_positive"] == pytest.approx(15.0)


def test_vanguard_kinetics_require_physical_time_sidecar() -> None:
    """New Vanguard cases must never silently substitute frame indices."""
    with pytest.raises(FileNotFoundError, match="ufast_times_seconds.npy"):
        _time_axis_from_study_timepoints([0, 1, 2], require_physical_seconds=True)


def _write_case(
    studies_dir: Path,
    dce_dir: Path,
    case_id: str,
    *,
    num_timepoints: int = NUM_TIMEPOINTS,
    flat_voxel_x: int | None = None,
) -> None:
    """Write one synthetic case: centerline tree under ``studies_dir/NACT`` plus a raw DCE-MRI NIfTI series under ``dce_dir``.

    Every skeleton voxel gets a monotonically rising curve (``enhancement = t``)
    except ``flat_voxel_x`` (if given), which stays at zero for every timepoint
    -- a "no measurable enhancement" voxel, used to exercise the
    ``time_to_enhancement`` NaN path.
    """
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

    # Monotonically rising raw DCE signal -> enhancement = t everywhere, except
    # flat_voxel_x (if given), which stays at zero.
    dce_case_dir = dce_dir / case_id
    dce_case_dir.mkdir(parents=True, exist_ok=True)
    for t in range(num_timepoints):
        phase = np.full(VOLUME_SHAPE, float(t), dtype=np.float32)
        if flat_voxel_x is not None:
            phase[SKELETON_Z, SKELETON_Y, flat_voxel_x] = 0.0
        sitk.WriteImage(
            sitk.GetImageFromArray(phase),
            str(dce_case_dir / f"{case_id}_{t:04d}.nii.gz"),
        )

    (case_dir / "run_summary.json").write_text(
        json.dumps({"study_timepoints": list(range(num_timepoints))})
    )


@pytest.fixture
def centerline_tree(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Build a two-case tree: one labeled case and one with no label row."""
    studies = tmp_path / "studies"
    dce_root = tmp_path / "images"
    _write_case(studies, dce_root, "NACT_01")
    _write_case(studies, dce_root, "NACT_99")  # deliberately absent from labels

    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text("case_id,pcr\nNACT_01,1\n")

    cache_dir = tmp_path / "cache"
    return studies, dce_root, labels_csv, cache_dir


def test_builds_labeled_graph_and_skips_unlabeled(
    centerline_tree: tuple[Path, Path, Path, Path],
) -> None:
    """A labeled case yields a valid graph; an unlabeled case is dropped."""
    studies, dce_root, labels_csv, cache_dir = centerline_tree

    node_features = (
        "peak_time",
        "radius",
        "peak_enhancement",
        "time_to_enhancement",
        "washin_slope",
        "auc_positive",
    )
    dataset = VanguardCenterlineDataset(
        studies,
        labels_path=labels_csv,
        dce_root=dce_root,
        cache_dir=cache_dir,
        node_features=node_features,
        profile=True,
        # This 2-case fixture is 50% unlabeled by construction; the default
        # max_missing_label_frac=0.1 is calibrated for real cohorts, not
        # tiny synthetic ones, so raise it here to exercise the drop path.
        max_missing_label_frac=0.5,
    )

    # Only the labeled case survives; NACT_99 is skipped for lacking a label.
    assert len(dataset) == 1
    data = dataset[0]

    assert data.num_nodes == len(SKELETON_XS)
    assert data.x.shape == (len(SKELETON_XS), len(node_features))
    assert data.pos.shape == (len(SKELETON_XS), 3)

    # 4 undirected edges -> 8 directed entries, and every edge has its reverse.
    assert data.edge_index.shape[1] == 2 * (len(SKELETON_XS) - 1)
    edges = {(int(u), int(v)) for u, v in data.edge_index.t().tolist()}
    assert all((v, u) in edges for (u, v) in edges)

    # Every node has the same synthetic curve, so the kinetic features are
    # deterministic: enhancement = [0, 1, 2] at every voxel.
    assert int(data.peak_time.min()) == EXPECTED_PEAK_IDX
    assert int(data.peak_time.max()) == EXPECTED_PEAK_IDX
    assert int(data.time_to_enhancement.min()) == EXPECTED_TTE_IDX
    assert int(data.time_to_enhancement.max()) == EXPECTED_TTE_IDX
    assert torch.allclose(data.peak_enhancement, torch.full((len(SKELETON_XS),), 2.0))
    assert torch.allclose(data.washin_slope, torch.full((len(SKELETON_XS),), 1.0))
    assert torch.allclose(data.auc_positive, torch.full((len(SKELETON_XS),), 2.0))

    assert data.y.tolist() == [1]
    assert data.case_id == "NACT_01"
    assert data.dataset == "NACT"

    # Per-case graph artifact is emitted alongside the collated cache.
    assert (cache_dir / "processed" / "NACT_01_graph.pt").exists()

    # A fresh build also writes a feature summary: histograms + NaN/inf report.
    summary_dir = cache_dir / "processed" / "feature_summary"
    for name in node_features:
        assert (summary_dir / f"{name}_hist.png").exists()
    assert (summary_dir / "README.md").exists()
    na_report = json.loads((summary_dir / "feature_na_report.json").read_text())
    assert set(na_report) == set(node_features)
    for column_report in na_report.values():
        assert column_report["num_values"] == len(SKELETON_XS)

    # A fresh build also writes a per-graph QC summary for confound auditing.
    qc_path = cache_dir / "processed" / "graph_qc.csv"
    assert qc_path.exists()
    qc = pd.read_csv(qc_path)
    assert len(qc) == 1
    row = qc.iloc[0]
    assert row["case_id"] == "NACT_01"
    assert row["dataset"] == "NACT"
    assert row["pcr"] == 1
    assert row["num_nodes"] == len(SKELETON_XS)
    # 4 undirected edges -> 8 directed entries (matches data.num_edges).
    assert row["num_edges"] == 2 * (len(SKELETON_XS) - 1)
    assert row["num_connected_components"] == 1
    assert row["mean_degree"] == row["num_edges"] / row["num_nodes"]
    assert row["missing_feature_count"] == 0
    assert row["nan_feature_count"] == 0
    for name in node_features:
        assert f"{name}_min" in qc.columns
        assert f"{name}_max" in qc.columns
        assert f"{name}_mean" in qc.columns
        assert f"{name}_std" in qc.columns
    # Every node has the identical synthetic curve -> zero spread.
    assert row["peak_enhancement_std"] == 0.0
    assert (
        row["peak_enhancement_min"]
        == row["peak_enhancement_max"]
        == EXPECTED_PEAK_ENHANCEMENT
    )

    # The 4 build-time-derivable QC plots are rendered automatically too
    # (prediction_vs_num_nodes.png needs a trained model and is written by
    # gnn/train.py instead -- see gnn/graph_qc_plots.py).
    plots_dir = cache_dir / "processed" / "graph_qc_plots"
    for name in (
        "num_nodes_vs_pcr.png",
        "num_nodes_vs_dataset.png",
        "feature_distributions_by_dataset.png",
        "feature_distributions_by_pcr.png",
    ):
        assert (plots_dir / name).exists()


def test_missing_dce_root_raises(
    centerline_tree: tuple[Path, Path, Path, Path],
) -> None:
    """DCE root is mandatory: kinetic features cannot be sampled without it."""
    studies, _dce_root, labels_csv, cache_dir = centerline_tree
    with pytest.raises(ValueError, match="dce_root is required"):
        VanguardCenterlineDataset(
            studies, labels_path=labels_csv, dce_root=None, cache_dir=cache_dir
        )


def test_missing_labels_path_raises(tmp_path: Path) -> None:
    """Labels are mandatory: a missing path fails fast."""
    with pytest.raises(ValueError, match="labels_path is required"):
        VanguardCenterlineDataset(tmp_path, labels_path=None, dce_root=tmp_path)


def test_segment_mode_not_implemented(tmp_path: Path) -> None:
    """The ``segment`` node mode is an explicit, unimplemented extension point."""
    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text("case_id,pcr\nNACT_01,1\n")
    with pytest.raises(NotImplementedError):
        VanguardCenterlineDataset(
            tmp_path, labels_path=labels_csv, dce_root=tmp_path, node_mode="segment"
        )


def test_cache_manifest_written_and_validated_on_reload(
    centerline_tree: tuple[Path, Path, Path, Path],
) -> None:
    """A fresh build writes cache_manifest.json; a matching reload succeeds."""
    studies, dce_root, labels_csv, cache_dir = centerline_tree

    VanguardCenterlineDataset(
        studies,
        labels_path=labels_csv,
        dce_root=dce_root,
        cache_dir=cache_dir,
        node_features=("peak_time", "radius"),
        max_missing_label_frac=0.5,
    )

    manifest_path = cache_dir / "processed" / "cache_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["centerline_root"] == str(studies)
    assert manifest["dce_root"] == str(dce_root)
    assert manifest["labels_path"] == str(labels_csv)
    assert manifest["label_column"] == "pcr"
    assert manifest["node_mode"] == "voxel"
    assert manifest["node_features"] == ["peak_time", "radius"]
    assert manifest["feature_source"] == "raw_dce_protocol_baseline_physical_time_v3"
    assert manifest["num_graphs"] == 1
    assert manifest["label_counts"] == {"1": 1}
    assert manifest["code_commit"]
    assert manifest["built_at"]

    # Loading again with identical settings hits the cache without complaint.
    reloaded = VanguardCenterlineDataset(
        studies,
        labels_path=labels_csv,
        dce_root=dce_root,
        cache_dir=cache_dir,
        node_features=("peak_time", "radius"),
        max_missing_label_frac=0.5,
    )
    assert len(reloaded) == 1


def test_cache_manifest_mismatch_raises_unless_overridden(
    centerline_tree: tuple[Path, Path, Path, Path],
) -> None:
    """Loading a cache with different settings than it was built with fails loudly."""
    studies, dce_root, labels_csv, cache_dir = centerline_tree

    VanguardCenterlineDataset(
        studies,
        labels_path=labels_csv,
        dce_root=dce_root,
        cache_dir=cache_dir,
        node_features=("peak_time", "radius"),
        max_missing_label_frac=0.5,
    )

    with pytest.raises(RuntimeError, match="different settings"):
        VanguardCenterlineDataset(
            studies,
            labels_path=labels_csv,
            dce_root=dce_root,
            cache_dir=cache_dir,
            node_features=(
                "radius",
            ),  # different from the cached ("peak_time", "radius")
            max_missing_label_frac=0.5,
        )

    # allow_manifest_mismatch=True explicitly bypasses the check.
    overridden = VanguardCenterlineDataset(
        studies,
        labels_path=labels_csv,
        dce_root=dce_root,
        cache_dir=cache_dir,
        node_features=("radius",),
        max_missing_label_frac=0.5,
        allow_manifest_mismatch=True,
    )
    assert len(overridden) == 1


def test_cache_manifest_cases_mismatch_raises_unless_overridden(
    centerline_tree: tuple[Path, Path, Path, Path],
) -> None:
    """Changing the ``cases`` whitelist is also a manifest-tracked setting."""
    studies, dce_root, labels_csv, cache_dir = centerline_tree

    VanguardCenterlineDataset(
        studies,
        labels_path=labels_csv,
        dce_root=dce_root,
        cache_dir=cache_dir,
        cases=["NACT_01"],
        node_features=("peak_time", "radius"),
        max_missing_label_frac=0.5,
    )

    with pytest.raises(RuntimeError, match="different settings"):
        VanguardCenterlineDataset(
            studies,
            labels_path=labels_csv,
            dce_root=dce_root,
            cache_dir=cache_dir,
            cases=None,  # different from the cached ["NACT_01"]
            node_features=("peak_time", "radius"),
            max_missing_label_frac=0.5,
        )

    # allow_manifest_mismatch=True explicitly bypasses the check.
    overridden = VanguardCenterlineDataset(
        studies,
        labels_path=labels_csv,
        dce_root=dce_root,
        cache_dir=cache_dir,
        cases=None,
        node_features=("peak_time", "radius"),
        max_missing_label_frac=0.5,
        allow_manifest_mismatch=True,
    )
    assert len(overridden) == 1


def test_cache_manifest_missing_raises(
    centerline_tree: tuple[Path, Path, Path, Path],
) -> None:
    """A cache built before manifest tracking existed fails loudly, not silently."""
    studies, dce_root, labels_csv, cache_dir = centerline_tree

    VanguardCenterlineDataset(
        studies,
        labels_path=labels_csv,
        dce_root=dce_root,
        cache_dir=cache_dir,
        node_features=("peak_time", "radius"),
        max_missing_label_frac=0.5,
    )
    (cache_dir / "processed" / "cache_manifest.json").unlink()

    with pytest.raises(RuntimeError, match="no cache_manifest.json"):
        VanguardCenterlineDataset(
            studies,
            labels_path=labels_csv,
            dce_root=dce_root,
            cache_dir=cache_dir,
            node_features=("peak_time", "radius"),
            max_missing_label_frac=0.5,
        )


def test_no_signal_voxel_reports_nan_time_to_enhancement(tmp_path: Path) -> None:
    """A voxel with no measurable enhancement gets a NaN time_to_enhancement, caught by the feature_na_report audit rather than a silent fallback value."""
    studies = tmp_path / "studies"
    dce_root = tmp_path / "images"
    flat_x = SKELETON_XS[-1]
    _write_case(studies, dce_root, "NACT_01", flat_voxel_x=flat_x)

    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text("case_id,pcr\nNACT_01,1\n")

    dataset = VanguardCenterlineDataset(
        studies,
        labels_path=labels_csv,
        dce_root=dce_root,
        cache_dir=tmp_path / "cache",
        node_features=("time_to_enhancement", "radius"),
    )
    data = dataset[0]

    # The flat voxel has no detected arrival -> sentinel -1 on the raw index,
    # NaN on the normalized feature actually fed into data.x.
    assert int(data.time_to_enhancement.min()) == -1
    assert torch.isnan(data.x[:, 0]).sum().item() == 1

    na_report = json.loads(
        (
            tmp_path
            / "cache"
            / "processed"
            / "feature_summary"
            / "feature_na_report.json"
        ).read_text()
    )
    assert na_report["time_to_enhancement"]["num_nan"] == 1

    qc = pd.read_csv(tmp_path / "cache" / "processed" / "graph_qc.csv")
    assert qc.iloc[0]["nan_feature_count"] == 1
    assert qc.iloc[0]["missing_feature_count"] == 1
