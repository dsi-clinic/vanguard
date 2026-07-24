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

import networkx as nx
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
sitk = pytest.importorskip("SimpleITK")

from gnn.data_loader import (  # noqa: E402
    _BIFURCATION_FEATURE_NAMES,
    _TTE_FEATURE_NAMES,
    NO_BIFURCATION_SENTINEL,
    TTE_NO_ARRIVAL_SENTINEL,
    VanguardCenterlineDataset,
    _attach_node_features,
    _load_study_metadata,
    _node_kinetic_features,
    _raise_on_unexpected_nan,
    _sentinel_fill,
    _time_axis_from_study_timepoints,
)
from gnn.junction_graph import build_junction_graph  # noqa: E402
from gnn.segment_graph import build_segment_line_graph  # noqa: E402

VOLUME_SHAPE = (4, 8, 8)  # (z, y, x)
NUM_TIMEPOINTS = 3
SKELETON_Z = 2
SKELETON_Y = 3
SKELETON_XS = (1, 2, 3, 4, 5)  # 5 voxels -> 4 undirected edges
NUM_CLINICAL_TEST_CASES = 2

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


def test_ufast_kinetics_match_across_all_graph_modes() -> None:
    """Voxel, segment, and junction modes share one physical-time contract."""
    curve = np.asarray([10.0, 12.0, 16.5, 22.0], dtype=np.float32)
    times = np.asarray([0.0, 5.0, 50.0, 55.0], dtype=np.float64)
    graph = nx.path_graph([(x, 3, 0) for x in range(1, 6)])
    radius_map = dict.fromkeys(graph.nodes, 1.0)
    dce_4d = np.empty((len(curve), 1, 8, 8), dtype=np.float32)
    for index, signal in enumerate(curve):
        dce_4d[index] = signal

    voxel_graph = graph.copy()
    _attach_node_features(voxel_graph, radius_map, dce_4d, times, 2, True, 0, ())
    segment_graph = build_segment_line_graph(
        graph,
        radius_map,
        dce_4d,
        times,
        baseline_frame_count=2,
        relative_enhancement=True,
    )
    junction_graph = build_junction_graph(
        graph,
        radius_map,
        dce_4d,
        times,
        baseline_frame_count=2,
        relative_enhancement=True,
    )

    voxel = voxel_graph.nodes[(1, 3, 0)]
    expected = {
        "peak_time": 1.0,
        "peak_enhancement": 1.0,
        "time_to_enhancement": 50.0 / 55.0,
        "washin_slope": 0.1,
        "washout_slope": 0.0,
        "auc_positive": 15.0,
    }
    voxel_values = {
        "peak_time": voxel["peak_time_norm"],
        "peak_enhancement": voxel["peak_enhancement"],
        "time_to_enhancement": voxel["time_to_enhancement_norm"],
        "washin_slope": voxel["washin_slope"],
        "washout_slope": voxel["washout_slope"],
        "auc_positive": voxel["auc_positive"],
    }
    for name, expected_value in expected.items():
        assert voxel_values[name] == pytest.approx(expected_value)
        assert segment_graph.nodes[0][f"seg_{name}_mean"] == pytest.approx(
            expected_value
        )
        assert float(junction_graph[name][0]) == pytest.approx(expected_value)
        assert float(junction_graph[f"seg_{name}_mean"][0]) == pytest.approx(
            expected_value
        )


def test_vanguard_kinetics_require_physical_time_sidecar() -> None:
    """New Vanguard cases must never silently substitute frame indices."""
    with pytest.raises(FileNotFoundError, match="ufast_times_seconds.npy"):
        _time_axis_from_study_timepoints([0, 1, 2], require_physical_seconds=True)


def test_vanguard_kinetics_require_explicit_alignment_approval(tmp_path: Path) -> None:
    """A missing alignment status can't silently authorize HR/UFAST kinetics."""
    summary = {
        "study_timepoints": [0, 1, 2],
        "kinetic_feature_policy": {
            "baseline_frame_count": 1,
            "enhancement": "relative_signal_change",
            "time_axis": "physical_seconds",
        },
    }
    (tmp_path / "run_summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="requires explicit alignment approval"):
        _load_study_metadata("case", tmp_path)


def test_run_summary_policy_defines_the_kinetic_contract(tmp_path: Path) -> None:
    """An approved v5 run_summary.json is the single source of the contract.

    The v5 Vanguard preprocessing writes ``kinetic_feature_policy`` +
    ``alignment_qc_status`` per case (baseline frame count is acquisition
    metadata, not a build knob), so the loader derives the ultrafast contract --
    baseline=5, relative signal change -- straight from the file, with no
    override.
    """
    n_timepoints = 24
    policy_baseline = 5
    summary = {
        "study_timepoints": list(range(n_timepoints)),
        "alignment_qc_status": "pass",
        "kinetic_feature_policy": {
            "baseline_frame_count": policy_baseline,
            "enhancement": "relative_signal_change",
            "time_axis": "physical_seconds",
        },
    }
    (tmp_path / "run_summary.json").write_text(json.dumps(summary))
    timepoints, baseline_frame_count, relative_enhancement = _load_study_metadata(
        "case", tmp_path
    )
    assert timepoints == list(range(n_timepoints))
    assert baseline_frame_count == policy_baseline
    assert relative_enhancement is True


def _write_morphometry_json(path: Path) -> None:
    """Write a tiny synthetic morphometry.json matching the real nested schema."""
    morph = {
        "1": {
            "component_1": [
                {
                    "segment": {"start": [0, 0, 0], "end": [3, 0, 0]},
                    "radius": {
                        "mean": 1.5,
                        "sd": 0.1,
                        "median": 1.5,
                        "min": 1.4,
                        "q1": 1.4,
                        "q3": 1.6,
                        "max": 1.7,
                    },
                    "length": 3.0,
                    "tortuosity": 1.1,
                    "volume": 10.0,
                    "curvature": {
                        "mean": 0.2,
                        "sd": 0.05,
                        "median": 0.2,
                        "min": 0.1,
                        "q1": 0.15,
                        "q3": 0.25,
                        "max": 0.3,
                    },
                }
            ],
            "component_1 bifurcation": [],
        }
    }
    path.write_text(json.dumps(morph))


def _write_case(
    studies_dir: Path,
    dce_dir: Path,
    case_id: str,
    *,
    num_timepoints: int = NUM_TIMEPOINTS,
    flat_voxel_x: int | None = None,
    include_morphometry: bool = False,
) -> None:
    """Write one synthetic case: centerline tree under ``studies_dir/NACT`` plus a raw DCE-MRI NIfTI series under ``dce_dir``.

    Every skeleton voxel gets a monotonically rising curve (``enhancement = t``)
    except ``flat_voxel_x`` (if given), which stays at zero for every timepoint
    -- a "no measurable enhancement" voxel, used to exercise the
    ``time_to_enhancement`` NaN path. ``include_morphometry`` additionally
    writes a synthetic ``<case_id>_morphometry.json`` and records its path in
    ``run_summary.json``, for tests exercising ``gnn.morphometry``.
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

    summary: dict[str, object] = {"study_timepoints": list(range(num_timepoints))}
    if include_morphometry:
        morphometry_path = case_dir / f"{case_id}_morphometry.json"
        _write_morphometry_json(morphometry_path)
        summary["morphometry_path"] = str(morphometry_path)
    (case_dir / "run_summary.json").write_text(json.dumps(summary))


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


def test_builds_junction_graph_with_edge_features(
    centerline_tree: tuple[Path, Path, Path, Path],
) -> None:
    """node_mode='junction' keeps junctions as nodes and segments as edges.

    The fixture skeletons are single unbranched vessels, so each becomes 2
    endpoint nodes joined by one segment (both directions -> 2 edges). This
    checks the junction-mode wiring end to end: junction per-voxel features back
    data.x, the segment summary backs data.edge_attr, and the cache manifest
    records node_mode + edge_features. Topology is covered separately by
    tests/test_gnn_junction_graph.py.
    """
    studies, dce_root, labels_csv, cache_dir = centerline_tree

    node_features = ("peak_time", "radius", "degree")
    edge_features = ("seg_length", "seg_radius_mean", "seg_peak_time_mean")
    dataset = VanguardCenterlineDataset(
        studies,
        labels_path=labels_csv,
        dce_root=dce_root,
        cache_dir=cache_dir,
        node_mode="junction",
        node_features=node_features,
        edge_features=edge_features,
        max_missing_label_frac=0.5,
    )

    assert len(dataset) == 1
    data = dataset[0]

    # Straight vessel -> 2 endpoint nodes, 1 segment -> 2 directed edges.
    n_endpoints, n_dir_edges, degree_col, straight_len = 2, 2, 2, 4.0
    assert data.num_nodes == n_endpoints
    assert data.x.shape == (n_endpoints, len(node_features))
    assert data.edge_index.shape == (2, n_dir_edges)
    assert data.edge_attr.shape == (n_dir_edges, len(edge_features))
    # Both endpoints are degree 1 (degree is the 3rd node feature).
    assert torch.allclose(data.x[:, degree_col], torch.ones(n_endpoints))
    # The segment summary on the edges: 5-voxel line is 4 units long.
    assert torch.allclose(
        data.edge_attr[:, 0], torch.full((n_dir_edges,), straight_len)
    )

    manifest = json.loads((cache_dir / "processed" / "cache_manifest.json").read_text())
    assert manifest["node_mode"] == "junction"
    assert manifest["node_features"] == list(node_features)
    assert manifest["edge_features"] == list(edge_features)


def test_edge_features_rejected_for_non_junction_modes(
    centerline_tree: tuple[Path, Path, Path, Path],
) -> None:
    """edge_features only make sense for junction mode; else fail loud."""
    studies, dce_root, labels_csv, cache_dir = centerline_tree
    with pytest.raises(ValueError, match="only supported for node_mode='junction'"):
        VanguardCenterlineDataset(
            studies,
            labels_path=labels_csv,
            dce_root=dce_root,
            cache_dir=cache_dir,
            node_mode="voxel",
            edge_features=("seg_length",),
            max_missing_label_frac=0.5,
        )


def test_builds_segment_line_graph(
    centerline_tree: tuple[Path, Path, Path, Path],
) -> None:
    """node_mode='segment' contracts each vessel to one segment-node.

    The fixture skeletons are single unbranched vessels, so each collapses to
    exactly one segment-node (0 edges). This checks the segment-mode wiring end
    to end: the segment feature vocabulary backs data.x, and the cache manifest,
    feature summary, and graph QC are all written under node_mode='segment'.
    Exact line-graph topology (junctions -> adjacency) is covered separately by
    tests/test_gnn_segment_graph.py on hand-built graphs.
    """
    studies, dce_root, labels_csv, cache_dir = centerline_tree

    node_features = (
        "seg_length",
        "seg_num_voxels",
        "seg_radius_mean",
        "seg_peak_time_mean",
        "seg_peak_enhancement_mean",
    )
    dataset = VanguardCenterlineDataset(
        studies,
        labels_path=labels_csv,
        dce_root=dce_root,
        cache_dir=cache_dir,
        node_mode="segment",
        node_features=node_features,
        max_missing_label_frac=0.5,
    )

    assert len(dataset) == 1
    data = dataset[0]

    # One unbranched vessel -> one segment-node, no edges.
    assert data.num_nodes == 1
    assert data.num_edges == 0
    assert data.x.shape == (1, len(node_features))
    assert data.pos.shape == (1, 3)

    # Segment features summarize the whole 5-voxel line: 4 unit steps long,
    # every voxel shares enhancement = [0, 1, 2] (peak idx 2 -> norm 1.0, peak
    # enhancement 2.0).
    seg = {name: float(data.x[0, i]) for i, name in enumerate(node_features)}
    assert seg["seg_num_voxels"] == len(SKELETON_XS)
    assert seg["seg_length"] == pytest.approx(len(SKELETON_XS) - 1)
    assert seg["seg_peak_time_mean"] == pytest.approx(1.0)
    assert seg["seg_peak_enhancement_mean"] == pytest.approx(EXPECTED_PEAK_ENHANCEMENT)
    assert seg["seg_radius_mean"] > 0.0

    assert data.y.tolist() == [1]
    assert data.case_id == "NACT_01"

    # Manifest records the mode, so a segment cache and a voxel cache built at
    # the same cache_dir can never be confused for one another.
    manifest = json.loads((cache_dir / "processed" / "cache_manifest.json").read_text())
    assert manifest["node_mode"] == "segment"
    assert manifest["node_features"] == list(node_features)

    # Feature summary + QC are written under the segment vocabulary.
    summary_dir = cache_dir / "processed" / "feature_summary"
    for name in node_features:
        assert (summary_dir / f"{name}_hist.png").exists()
    qc = pd.read_csv(cache_dir / "processed" / "graph_qc.csv")
    assert qc.iloc[0]["num_nodes"] == 1
    for name in node_features:
        assert f"{name}_mean" in qc.columns


def test_segment_mode_rejects_voxel_features(
    centerline_tree: tuple[Path, Path, Path, Path],
) -> None:
    """Voxel-vocabulary features are not valid segment features (fail loud)."""
    studies, dce_root, labels_csv, cache_dir = centerline_tree
    with pytest.raises(ValueError, match="Unknown node_features"):
        VanguardCenterlineDataset(
            studies,
            labels_path=labels_csv,
            dce_root=dce_root,
            cache_dir=cache_dir,
            node_mode="segment",
            node_features=("peak_time", "radius"),  # voxel names
            max_missing_label_frac=0.5,
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
    assert (
        manifest["feature_source"]
        == "raw_dce_protocol_baseline_physical_time_all_modes_v4"
    )
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


def test_cache_subset_node_features_reslices_without_override(
    centerline_tree: tuple[Path, Path, Path, Path],
) -> None:
    """A node_features request that's a genuine subset of the cache succeeds.

    This is the core mechanism LOCO relies on (see evaluation/loco_gnn.py): a
    cache built once with a feature superset can serve any narrower
    node_features request without allow_manifest_mismatch and without a
    rebuild -- see _check_cache_manifest / _reslice_for_requested_features.
    """
    studies, dce_root, labels_csv, cache_dir = centerline_tree

    VanguardCenterlineDataset(
        studies,
        labels_path=labels_csv,
        dce_root=dce_root,
        cache_dir=cache_dir,
        node_features=("peak_time", "radius"),
        max_missing_label_frac=0.5,
    )

    # Subset request succeeds with no allow_manifest_mismatch override.
    subset = VanguardCenterlineDataset(
        studies,
        labels_path=labels_csv,
        dce_root=dce_root,
        cache_dir=cache_dir,
        node_features=("radius",),  # subset of the cached ("peak_time", "radius")
        max_missing_label_frac=0.5,
    )
    assert len(subset) == 1
    assert subset[0].x.shape == (len(SKELETON_XS), 1)

    # Parity, not just shape: the reslice must match a fresh direct build
    # with the same narrower feature list.
    direct = VanguardCenterlineDataset(
        studies,
        labels_path=labels_csv,
        dce_root=dce_root,
        cache_dir=cache_dir / "direct",
        node_features=("radius",),
        max_missing_label_frac=0.5,
    )
    assert torch.allclose(subset[0].x, direct[0].x)


def test_cache_subset_edge_features_reslices_without_override(
    centerline_tree: tuple[Path, Path, Path, Path],
) -> None:
    """The same subset-reslice mechanism applies to junction-mode edge_features."""
    studies, dce_root, labels_csv, cache_dir = centerline_tree
    node_features = ("peak_time", "radius", "degree")
    full_edge_features = ("seg_length", "seg_radius_mean", "seg_peak_time_mean")

    VanguardCenterlineDataset(
        studies,
        labels_path=labels_csv,
        dce_root=dce_root,
        cache_dir=cache_dir,
        node_mode="junction",
        node_features=node_features,
        edge_features=full_edge_features,
        max_missing_label_frac=0.5,
    )

    subset = VanguardCenterlineDataset(
        studies,
        labels_path=labels_csv,
        dce_root=dce_root,
        cache_dir=cache_dir,
        node_mode="junction",
        node_features=node_features,
        edge_features=("seg_radius_mean",),  # subset of full_edge_features
        max_missing_label_frac=0.5,
    )
    assert subset[0].edge_attr.shape[1] == 1

    direct = VanguardCenterlineDataset(
        studies,
        labels_path=labels_csv,
        dce_root=dce_root,
        cache_dir=cache_dir / "direct",
        node_mode="junction",
        node_features=node_features,
        edge_features=("seg_radius_mean",),
        max_missing_label_frac=0.5,
    )
    assert torch.allclose(subset[0].edge_attr, direct[0].edge_attr)


def test_cache_non_subset_node_features_still_raises(
    centerline_tree: tuple[Path, Path, Path, Path],
) -> None:
    """A node_features request that is NOT a subset of the cache still raises.

    Confirms the subset-select fix only relaxes genuine subset requests, not
    arbitrary mismatches -- a name the cache never built is still an error.
    """
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
            node_features=("peak_time", "washin_slope"),  # washin_slope never cached
            max_missing_label_frac=0.5,
        )


def test_cache_manifest_override_with_truly_missing_feature_raises(
    centerline_tree: tuple[Path, Path, Path, Path],
) -> None:
    """allow_manifest_mismatch=True cannot conjure an uncached feature.

    It bypasses the manifest *equality* check, but the reslice step (see
    _reslice_for_requested_features) still fails loud if a requested feature
    was genuinely never cached -- overriding must never silently serve
    mismatched-width data, which is the exact bug this mechanism replaces.
    """
    studies, dce_root, labels_csv, cache_dir = centerline_tree

    VanguardCenterlineDataset(
        studies,
        labels_path=labels_csv,
        dce_root=dce_root,
        cache_dir=cache_dir,
        node_features=("peak_time", "radius"),
        max_missing_label_frac=0.5,
    )

    with pytest.raises(ValueError, match="washin_slope"):
        VanguardCenterlineDataset(
            studies,
            labels_path=labels_csv,
            dce_root=dce_root,
            cache_dir=cache_dir,
            node_features=("peak_time", "washin_slope"),
            max_missing_label_frac=0.5,
            allow_manifest_mismatch=True,
        )


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


def test_no_signal_voxel_gets_tte_sentinel(tmp_path: Path) -> None:
    """A voxel with no measurable enhancement gets the TTE sentinel in data.x.

    The no-arrival case is not silently defaulted to a plausible time: it is
    replaced with ``TTE_NO_ARRIVAL_SENTINEL`` (a distinct out-of-range value)
    so ``data.x`` carries no NaN (which would break training), and the count is
    surfaced by ``graph_qc.csv``'s ``tte_no_arrival_count`` rather than the
    generic NaN audit.
    """
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

    # The flat voxel has no detected arrival -> raw index keeps its -1 sentinel,
    # and the normalized TTE feature in data.x is the sentinel (not NaN).
    assert int(data.time_to_enhancement.min()) == -1
    assert not torch.isnan(data.x).any()
    tte_column = data.x[:, 0]
    assert (tte_column == TTE_NO_ARRIVAL_SENTINEL).sum().item() == 1
    assert int(data.tte_no_arrival_count) == 1

    na_report = json.loads(
        (
            tmp_path
            / "cache"
            / "processed"
            / "feature_summary"
            / "feature_na_report.json"
        ).read_text()
    )
    # No-arrival is sentinel-filled before the feature summary, so the NaN audit
    # is clean; the "no signal" count lives in graph_qc's tte_no_arrival_count.
    assert na_report["time_to_enhancement"]["num_nan"] == 0

    qc = pd.read_csv(tmp_path / "cache" / "processed" / "graph_qc.csv")
    assert qc.iloc[0]["nan_feature_count"] == 0
    assert qc.iloc[0]["missing_feature_count"] == 0
    assert qc.iloc[0]["tte_no_arrival_count"] == 1


def test_sentinel_fill_tte_fills_tte_columns_and_counts() -> None:
    """NaN in a TTE column is replaced with the sentinel; the count is returned."""
    real_arrival = 0.5
    expected_filled = 2
    matrix = torch.tensor(
        [[float("nan"), 1.0], [real_arrival, 2.0], [float("nan"), 3.0]]
    )
    filled = _sentinel_fill(
        matrix,
        ("time_to_enhancement", "radius"),
        _TTE_FEATURE_NAMES,
        TTE_NO_ARRIVAL_SENTINEL,
    )
    assert filled == expected_filled
    assert matrix[0, 0].item() == TTE_NO_ARRIVAL_SENTINEL
    assert matrix[2, 0].item() == TTE_NO_ARRIVAL_SENTINEL
    assert matrix[1, 0].item() == real_arrival  # a real arrival is left untouched
    assert not torch.isnan(matrix).any()


def test_sentinel_fill_tte_handles_segment_summary_column() -> None:
    """The segment/edge ``seg_time_to_enhancement_mean`` column is also filled."""
    matrix = torch.tensor([[float("nan")], [0.2]])
    filled = _sentinel_fill(
        matrix,
        ("seg_time_to_enhancement_mean",),
        _TTE_FEATURE_NAMES,
        TTE_NO_ARRIVAL_SENTINEL,
    )
    assert filled == 1
    assert matrix[0, 0].item() == TTE_NO_ARRIVAL_SENTINEL
    assert matrix[1, 0].item() == pytest.approx(0.2)


def test_sentinel_fill_bifurcation_angle_column() -> None:
    """A NaN ``bifurcation_angle_mean`` (no-bifurcation node) is sentinel-filled."""
    matrix = torch.tensor([[float("nan")], [120.0]])
    filled = _sentinel_fill(
        matrix,
        ("bifurcation_angle_mean",),
        _BIFURCATION_FEATURE_NAMES,
        NO_BIFURCATION_SENTINEL,
    )
    assert filled == 1
    assert matrix[0, 0].item() == NO_BIFURCATION_SENTINEL
    assert matrix[1, 0].item() == pytest.approx(120.0)


def test_raise_on_unexpected_nan_passes_when_clean() -> None:
    """A matrix with no NaN left (after sentinel fills) does not raise."""
    matrix = torch.tensor([[TTE_NO_ARRIVAL_SENTINEL, 0.5]])
    _raise_on_unexpected_nan(matrix)  # must not raise


def test_raise_on_unexpected_nan_raises_on_unregistered_nan() -> None:
    """A NaN outside every registered category is a bug and must raise."""
    matrix = torch.tensor([[0.5, float("nan")]])  # NaN in the radius column
    with pytest.raises(ValueError, match="Unexpected NaN"):
        _raise_on_unexpected_nan(matrix)


def _write_patient_info(
    patient_info_dir: Path,
    case_id: str,
    *,
    age: int,
    menopausal_status: str,
    bilateral: bool | None = None,
) -> None:
    patient_info_dir.mkdir(parents=True, exist_ok=True)
    imaging_data: dict[str, object] = {"dataset": "NACT"}
    if bilateral is not None:
        imaging_data["bilateral"] = bilateral
    (patient_info_dir / f"{case_id}.json").write_text(
        json.dumps(
            {
                "case_id": case_id,
                "clinical_data": {"age": age, "menopausal_status": menopausal_status},
                "primary_lesion": {"tumor_subtype": "luminal_a"},
                "imaging_data": imaging_data,
            }
        )
    )


def test_graph_features_cached_as_raw_inputs_not_baked_on_graphs(
    tmp_path: Path,
) -> None:
    """graph_features are cached as a raw-input sidecar, not baked onto data.graph_features.

    The whole-cohort transform is deliberately NOT applied at build time (it
    would leak the validation distribution across CV folds); the raw per-case
    inputs are cached instead, one row per case, for a per-fold fit in
    gnn.train. So the cached graphs carry no ``graph_features`` attribute, and
    ``load_graph_feature_inputs`` returns the raw values.
    """
    studies = tmp_path / "studies"
    dce_root = tmp_path / "images"
    _write_case(studies, dce_root, "NACT_01")
    _write_case(studies, dce_root, "NACT_02")

    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text("case_id,pcr\nNACT_01,1\nNACT_02,0\n")

    patient_info_dir = tmp_path / "patient_info"
    _write_patient_info(patient_info_dir, "NACT_01", age=40, menopausal_status="pre")
    _write_patient_info(patient_info_dir, "NACT_02", age=60, menopausal_status="post")

    dataset = VanguardCenterlineDataset(
        studies,
        labels_path=labels_csv,
        dce_root=dce_root,
        cache_dir=tmp_path / "cache",
        node_features=("peak_time", "radius"),
        graph_features=("age",),
        patient_info_dir=patient_info_dir,
    )

    assert len(dataset) == NUM_CLINICAL_TEST_CASES
    assert dataset.dropped_case_ids == []
    # Nothing transformed is baked onto the cached graphs any more.
    for i in range(len(dataset)):
        assert getattr(dataset[i], "graph_features", None) is None
    # The raw (un-imputed, un-encoded) inputs are cached instead.
    inputs = dataset.load_graph_feature_inputs()
    assert list(inputs.columns) == ["age"]
    assert inputs.loc["NACT_01", "age"] == pytest.approx(40.0)
    assert inputs.loc["NACT_02", "age"] == pytest.approx(60.0)


def test_graph_features_requires_clinical_source() -> None:
    """A clinical graph_features name without patient_info_dir/clinical_excel fails fast."""
    with pytest.raises(ValueError, match="graph_features includes clinical column"):
        VanguardCenterlineDataset(
            Path("/nonexistent"),
            labels_path=Path("/nonexistent/labels.csv"),
            dce_root=Path("/nonexistent/images"),
            graph_features=("age",),
        )


def test_case_missing_clinical_row_is_dropped(tmp_path: Path) -> None:
    """A labeled case with no clinical row is dropped (missing_clinical), not a hard failure."""
    studies = tmp_path / "studies"
    dce_root = tmp_path / "images"
    _write_case(studies, dce_root, "NACT_01")
    _write_case(studies, dce_root, "NACT_02")  # no patient_info entry below

    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text("case_id,pcr\nNACT_01,1\nNACT_02,0\n")

    patient_info_dir = tmp_path / "patient_info"
    _write_patient_info(patient_info_dir, "NACT_01", age=40, menopausal_status="pre")

    dataset = VanguardCenterlineDataset(
        studies,
        labels_path=labels_csv,
        dce_root=dce_root,
        cache_dir=tmp_path / "cache",
        node_features=("peak_time", "radius"),
        graph_features=("age",),
        patient_info_dir=patient_info_dir,
        max_missing_clinical_frac=0.5,
    )

    assert len(dataset) == 1
    assert dataset.dropped_case_ids == ["NACT_02"]
    assert dataset[0].case_id == "NACT_01"


def test_missing_clinical_frac_exceeded_raises(tmp_path: Path) -> None:
    """Dropping more cases than max_missing_clinical_frac allows raises loudly."""
    studies = tmp_path / "studies"
    dce_root = tmp_path / "images"
    _write_case(studies, dce_root, "NACT_01")
    _write_case(studies, dce_root, "NACT_02")

    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text("case_id,pcr\nNACT_01,1\nNACT_02,0\n")

    patient_info_dir = tmp_path / "patient_info"
    _write_patient_info(patient_info_dir, "NACT_01", age=40, menopausal_status="pre")
    # NACT_02 has no clinical row -> 1/2 = 50% missing, exceeds a 0.1 threshold.

    with pytest.raises(RuntimeError, match="max_missing_clinical_frac"):
        VanguardCenterlineDataset(
            studies,
            labels_path=labels_csv,
            dce_root=dce_root,
            cache_dir=tmp_path / "cache",
            node_features=("peak_time", "radius"),
            graph_features=("age",),
            patient_info_dir=patient_info_dir,
            max_missing_clinical_frac=0.1,
        )


def test_mixed_source_graph_feature_inputs_in_fixed_source_order(
    tmp_path: Path,
) -> None:
    """Clinical + graph-derived + morphometry names land in one raw sidecar, fixed order."""
    studies = tmp_path / "studies"
    dce_root = tmp_path / "images"
    _write_case(studies, dce_root, "NACT_01", include_morphometry=True)

    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text("case_id,pcr\nNACT_01,1\n")

    patient_info_dir = tmp_path / "patient_info"
    _write_patient_info(patient_info_dir, "NACT_01", age=40, menopausal_status="pre")

    dataset = VanguardCenterlineDataset(
        studies,
        labels_path=labels_csv,
        dce_root=dce_root,
        cache_dir=tmp_path / "cache",
        node_features=("peak_time", "radius"),
        # Deliberately interleaved request order; the sidecar must still lay
        # columns out in fixed source order (clinical, graph-derived, morphometry).
        graph_features=("morph_seg_length_n", "age", "num_connected_components"),
        patient_info_dir=patient_info_dir,
    )

    inputs = dataset.load_graph_feature_inputs()
    assert list(inputs.columns) == [
        "age",
        "num_connected_components",
        "morph_seg_length_n",
    ]
    assert dataset.graph_feature_groups() == (
        ["age"],
        ["num_connected_components"],
        ["morph_seg_length_n"],
    )
    row = inputs.loc["NACT_01"]
    assert row["age"] == pytest.approx(40.0)
    assert row["num_connected_components"] == pytest.approx(1.0)  # one straight vessel
    assert row["morph_seg_length_n"] == pytest.approx(1.0)  # one synthetic segment


def test_graph_derived_only_does_not_require_clinical_source(tmp_path: Path) -> None:
    """A graph_features request with no clinical names needs no patient_info_dir/clinical_excel."""
    studies = tmp_path / "studies"
    dce_root = tmp_path / "images"
    _write_case(studies, dce_root, "NACT_01")

    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text("case_id,pcr\nNACT_01,1\n")

    dataset = VanguardCenterlineDataset(
        studies,
        labels_path=labels_csv,
        dce_root=dce_root,
        cache_dir=tmp_path / "cache",
        node_features=("peak_time", "radius"),
        graph_features=("num_connected_components",),
    )

    inputs = dataset.load_graph_feature_inputs()
    assert list(inputs.columns) == ["num_connected_components"]
    assert inputs.loc["NACT_01", "num_connected_components"] == pytest.approx(1.0)


def test_morphometry_only_does_not_require_clinical_source(tmp_path: Path) -> None:
    """A morphometry-only graph_features request needs no clinical source either."""
    studies = tmp_path / "studies"
    dce_root = tmp_path / "images"
    _write_case(studies, dce_root, "NACT_01", include_morphometry=True)

    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text("case_id,pcr\nNACT_01,1\n")

    dataset = VanguardCenterlineDataset(
        studies,
        labels_path=labels_csv,
        dce_root=dce_root,
        cache_dir=tmp_path / "cache",
        node_features=("peak_time", "radius"),
        graph_features=("morph_seg_length_n", "morph_bifurcation_count"),
    )

    inputs = dataset.load_graph_feature_inputs()
    assert list(inputs.columns) == ["morph_seg_length_n", "morph_bifurcation_count"]
    assert inputs.loc["NACT_01"].tolist() == pytest.approx([1.0, 0.0])


def test_missing_morphometry_path_raises(tmp_path: Path) -> None:
    """A case whose run_summary.json lacks morphometry_path is a hard failure, not a drop."""
    studies = tmp_path / "studies"
    dce_root = tmp_path / "images"
    _write_case(studies, dce_root, "NACT_01", include_morphometry=False)

    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text("case_id,pcr\nNACT_01,1\n")

    with pytest.raises(KeyError, match="morphometry_path"):
        VanguardCenterlineDataset(
            studies,
            labels_path=labels_csv,
            dce_root=dce_root,
            cache_dir=tmp_path / "cache",
            node_features=("peak_time", "radius"),
            graph_features=("morph_seg_length_n",),
        )


def _write_single_breast_skeleton(
    breast_split_root: Path,
    dataset_name: str,
    case_id: str,
    xs: tuple[int, ...],
    *,
    num_timepoints: int = NUM_TIMEPOINTS,
) -> None:
    """Write a synthetic precomputed single-breast skeleton, support mask, and run_summary.json.

    Mirrors what ``gnn.build_single_breast_skeletons`` actually writes:
    ``gnn/data_loader.py::_build_case`` looks up both the support mask and
    ``study_timepoints`` (from ``run_summary.json``) next to whichever
    skeleton path it's given, unaware of breast-split substitution --
    omitting either would make this fixture unrepresentative of the real
    precompute output.
    """
    out_dir = breast_split_root / dataset_name / case_id
    out_dir.mkdir(parents=True, exist_ok=True)
    skeleton = np.zeros(VOLUME_SHAPE, dtype=bool)
    for x in xs:
        skeleton[SKELETON_Z, SKELETON_Y, x] = True
    np.save(out_dir / f"{case_id}_skeleton_4d_single_breast_mask.npy", skeleton)

    support = np.zeros(VOLUME_SHAPE, dtype=bool)
    support[
        SKELETON_Z - 1 : SKELETON_Z + 2,
        SKELETON_Y - 1 : SKELETON_Y + 2,
        min(xs) - 1 : max(xs) + 2,
    ] = True
    np.save(out_dir / f"{case_id}_skeleton_4d_exam_support_mask.npy", support)

    (out_dir / "run_summary.json").write_text(
        json.dumps({"study_timepoints": list(range(num_timepoints))})
    )


def test_breast_split_single_substitutes_bilateral_case_skeleton(
    tmp_path: Path,
) -> None:
    """A bilateral case with a precomputed single-breast skeleton builds from that file, not the exam-level one."""
    studies = tmp_path / "studies"
    dce_root = tmp_path / "images"
    _write_case(studies, dce_root, "NACT_01")

    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text("case_id,pcr\nNACT_01,1\n")

    patient_info_dir = tmp_path / "patient_info"
    _write_patient_info(
        patient_info_dir, "NACT_01", age=40, menopausal_status="pre", bilateral=True
    )

    breast_split_root = tmp_path / "breast_split_skeletons"
    kept_xs = (1, 2, 3)  # a strict subset of SKELETON_XS = (1, 2, 3, 4, 5)
    _write_single_breast_skeleton(breast_split_root, "NACT", "NACT_01", kept_xs)

    dataset = VanguardCenterlineDataset(
        studies,
        labels_path=labels_csv,
        dce_root=dce_root,
        cache_dir=tmp_path / "cache",
        node_features=("peak_time", "radius"),
        patient_info_dir=patient_info_dir,
        breast_split_mode="single",
        breast_split_skeleton_root=breast_split_root,
    )

    assert dataset.dropped_case_ids == []
    data = dataset[0]
    assert data.num_nodes == len(kept_xs)  # not len(SKELETON_XS) -- proves substitution


def test_cache_manifest_breast_split_mismatch_raises_unless_overridden(
    tmp_path: Path,
) -> None:
    """A cache built with breast_split_mode="single" can't be silently reloaded as the unsplit cohort."""
    studies = tmp_path / "studies"
    dce_root = tmp_path / "images"
    _write_case(studies, dce_root, "NACT_01")

    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text("case_id,pcr\nNACT_01,1\n")

    patient_info_dir = tmp_path / "patient_info"
    _write_patient_info(
        patient_info_dir, "NACT_01", age=40, menopausal_status="pre", bilateral=True
    )

    breast_split_root = tmp_path / "breast_split_skeletons"
    _write_single_breast_skeleton(breast_split_root, "NACT", "NACT_01", (1, 2, 3))

    cache_dir = tmp_path / "cache"
    VanguardCenterlineDataset(
        studies,
        labels_path=labels_csv,
        dce_root=dce_root,
        cache_dir=cache_dir,
        node_features=("peak_time", "radius"),
        patient_info_dir=patient_info_dir,
        breast_split_mode="single",
        breast_split_skeleton_root=breast_split_root,
    )

    with pytest.raises(RuntimeError, match="different settings"):
        VanguardCenterlineDataset(
            studies,
            labels_path=labels_csv,
            dce_root=dce_root,
            cache_dir=cache_dir,
            node_features=("peak_time", "radius"),
            # breast_split_mode omitted -- this must not silently load the
            # single-breast-built cache as if it were the mixed cohort.
        )

    overridden = VanguardCenterlineDataset(
        studies,
        labels_path=labels_csv,
        dce_root=dce_root,
        cache_dir=cache_dir,
        node_features=("peak_time", "radius"),
        allow_manifest_mismatch=True,
    )
    assert len(overridden) == 1


def test_breast_split_single_leaves_unilateral_case_unchanged(tmp_path: Path) -> None:
    """A case flagged bilateral=False keeps its original exam-level skeleton."""
    studies = tmp_path / "studies"
    dce_root = tmp_path / "images"
    _write_case(studies, dce_root, "NACT_01")

    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text("case_id,pcr\nNACT_01,1\n")

    patient_info_dir = tmp_path / "patient_info"
    _write_patient_info(
        patient_info_dir, "NACT_01", age=40, menopausal_status="pre", bilateral=False
    )

    dataset = VanguardCenterlineDataset(
        studies,
        labels_path=labels_csv,
        dce_root=dce_root,
        cache_dir=tmp_path / "cache",
        node_features=("peak_time", "radius"),
        patient_info_dir=patient_info_dir,
        breast_split_mode="single",
        breast_split_skeleton_root=tmp_path / "breast_split_skeletons",  # empty, unused
    )

    assert dataset.dropped_case_ids == []
    assert dataset[0].num_nodes == len(SKELETON_XS)


def test_breast_split_single_drops_bilateral_case_with_no_precomputed_skeleton(
    tmp_path: Path,
) -> None:
    """A bilateral case with no precomputed single-breast file is dropped, not silently kept mixed."""
    studies = tmp_path / "studies"
    dce_root = tmp_path / "images"
    _write_case(studies, dce_root, "NACT_01")
    _write_case(studies, dce_root, "NACT_02")

    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text("case_id,pcr\nNACT_01,1\nNACT_02,0\n")

    patient_info_dir = tmp_path / "patient_info"
    _write_patient_info(
        patient_info_dir, "NACT_01", age=40, menopausal_status="pre", bilateral=True
    )
    _write_patient_info(
        patient_info_dir, "NACT_02", age=50, menopausal_status="post", bilateral=False
    )

    breast_split_root = tmp_path / "breast_split_skeletons"
    # No single-breast skeleton written for NACT_01 -- it was "excluded by the splitter".

    dataset = VanguardCenterlineDataset(
        studies,
        labels_path=labels_csv,
        dce_root=dce_root,
        cache_dir=tmp_path / "cache",
        node_features=("peak_time", "radius"),
        patient_info_dir=patient_info_dir,
        breast_split_mode="single",
        breast_split_skeleton_root=breast_split_root,
        max_missing_breast_split_frac=0.5,
    )

    assert dataset.dropped_case_ids == ["NACT_01"]
    assert len(dataset) == 1
    assert dataset[0].case_id == "NACT_02"

    dropped_manifest = json.loads(
        (tmp_path / "cache" / "processed" / "dropped_cases.json").read_text()
    )
    assert dropped_manifest["dropped_reasons"]["NACT_01"] == "breast_split_excluded"


def test_breast_split_single_drops_case_with_unknown_laterality(tmp_path: Path) -> None:
    """A case with no clinical row at all can't be laterality-checked, so it's dropped."""
    studies = tmp_path / "studies"
    dce_root = tmp_path / "images"
    _write_case(studies, dce_root, "NACT_01")
    _write_case(studies, dce_root, "NACT_02")

    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text("case_id,pcr\nNACT_01,1\nNACT_02,0\n")

    patient_info_dir = tmp_path / "patient_info"
    # Only NACT_02 has a clinical row; NACT_01's laterality is unknowable.
    _write_patient_info(
        patient_info_dir, "NACT_02", age=50, menopausal_status="post", bilateral=False
    )

    dataset = VanguardCenterlineDataset(
        studies,
        labels_path=labels_csv,
        dce_root=dce_root,
        cache_dir=tmp_path / "cache",
        node_features=("peak_time", "radius"),
        patient_info_dir=patient_info_dir,
        breast_split_mode="single",
        breast_split_skeleton_root=tmp_path / "breast_split_skeletons",
        max_missing_breast_split_frac=0.5,
    )

    assert dataset.dropped_case_ids == ["NACT_01"]
    dropped_manifest = json.loads(
        (tmp_path / "cache" / "processed" / "dropped_cases.json").read_text()
    )
    assert (
        dropped_manifest["dropped_reasons"]["NACT_01"]
        == "breast_split_unknown_laterality"
    )


def test_breast_split_max_missing_frac_exceeded_raises(tmp_path: Path) -> None:
    """Too many breast-split exclusions raises instead of silently shrinking the cohort."""
    studies = tmp_path / "studies"
    dce_root = tmp_path / "images"
    _write_case(studies, dce_root, "NACT_01")

    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text("case_id,pcr\nNACT_01,1\n")

    patient_info_dir = tmp_path / "patient_info"
    _write_patient_info(
        patient_info_dir, "NACT_01", age=40, menopausal_status="pre", bilateral=True
    )
    # No precomputed single-breast skeleton -> excluded; this is the only case,
    # so the exclusion fraction is 100%, exceeding the default 0.1 threshold.

    with pytest.raises(RuntimeError, match="max_missing_breast_split_frac"):
        VanguardCenterlineDataset(
            studies,
            labels_path=labels_csv,
            dce_root=dce_root,
            cache_dir=tmp_path / "cache",
            node_features=("peak_time", "radius"),
            patient_info_dir=patient_info_dir,
            breast_split_mode="single",
            breast_split_skeleton_root=tmp_path / "breast_split_skeletons",
        )


def test_breast_split_requires_skeleton_root() -> None:
    """breast_split_mode='single' without breast_split_skeleton_root fails fast."""
    with pytest.raises(ValueError, match="breast_split_skeleton_root"):
        VanguardCenterlineDataset(
            Path("/nonexistent"),
            labels_path=Path("/nonexistent/labels.csv"),
            dce_root=Path("/nonexistent/images"),
            patient_info_dir=Path("/nonexistent/patient_info"),
            breast_split_mode="single",
        )


def test_breast_split_requires_clinical_source() -> None:
    """breast_split_mode='single' without a clinical source fails fast."""
    with pytest.raises(ValueError, match="patient_info_dir or"):
        VanguardCenterlineDataset(
            Path("/nonexistent"),
            labels_path=Path("/nonexistent/labels.csv"),
            dce_root=Path("/nonexistent/images"),
            breast_split_mode="single",
            breast_split_skeleton_root=Path("/nonexistent/breast_split_skeletons"),
        )


def test_breast_split_rejects_unknown_mode() -> None:
    """An unsupported breast_split_mode value fails fast rather than being silently ignored."""
    with pytest.raises(ValueError, match="breast_split_mode"):
        VanguardCenterlineDataset(
            Path("/nonexistent"),
            labels_path=Path("/nonexistent/labels.csv"),
            dce_root=Path("/nonexistent/images"),
            patient_info_dir=Path("/nonexistent/patient_info"),
            breast_split_mode="both",
            breast_split_skeleton_root=Path("/nonexistent/breast_split_skeletons"),
        )
