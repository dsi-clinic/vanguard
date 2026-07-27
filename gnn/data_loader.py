"""PyTorch-Geometric dataset that builds raw vessel graphs from centerlines.

The tabular / Deep Sets pipelines consume *summarized* vessel-graph features. The
GNN track instead needs the **raw graph** -- one node per skeleton voxel, edges
between 26-connected voxels -- delivered as :class:`torch_geometric.data.Data`
objects. This module walks a saved centerline output tree, rebuilds each case's
graph with the existing ``graph_extraction`` primitives, attaches node features
(DCE enhancement kinetics, local radius), and collates everything into an
:class:`~torch_geometric.data.InMemoryDataset`.

The heavy graph-building work is deliberately shared with the rest of the repo:
we reuse ``mask_to_edges_bitmask``, ``edges_to_segments``, ``segments_to_graph``
and ``obtain_radius_map`` so there is a single source of truth for how a skeleton
mask becomes a graph. Kinetic node features are sampled from the raw DCE-MRI
series (via ``gnn.raw_dce``) and derived using the same enhancement-curve
conventions as ``features/kinematic.py`` (baseline = timepoint 0, arrival via
``graph_extraction.feature_stats._arrival_index_from_enhancement``) -- not from
the vessel-segmentation probability maps, which are a model output rather than
measured signal.
"""

from __future__ import annotations

import json
import logging
import statistics
import subprocess
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import from_networkx

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from clinical_features import load_clinical_from_excel, load_clinical_from_patient_info
from config import DEFAULT_CONFIG
from gnn.breast_split import single_breast_skeleton_path
from gnn.clinical import ALL_CLINICAL_COLUMNS, normalize_clinical_frame
from gnn.graph_derived import GRAPH_DERIVED_COLUMNS, build_graph_derived_feature_matrix
from gnn.graph_qc_plots import GRAPH_QC_PLOTS_DIRNAME, write_build_time_plots
from gnn.junction_graph import (
    JUNCTION_EDGE_FEATURE_ATTR,
    JUNCTION_NODE_FEATURE_ATTR,
    build_junction_graph,
)
from gnn.kinetics import (
    node_kinetic_features as _node_kinetic_features,
)
from gnn.kinetics import (
    time_axis_from_study_timepoints as _time_axis_from_study_timepoints,
)
from gnn.morphometry import MORPHOMETRY_COLUMNS, extract_morphometry_frame
from gnn.raw_dce import (
    discover_raw_dce_paths,
    load_raw_dce_series,
    load_raw_dce_times,
)
from gnn.segment_graph import SEGMENT_FEATURE_ATTR, build_segment_line_graph
from graph_extraction.constants import NDIM_3D
from graph_extraction.feature_stats import mask_to_edges_bitmask
from graph_extraction.skeleton_to_graph_primitives import (
    edges_to_segments,
    obtain_radius_map,
    segments_to_graph,
)
from tabular.cohort import load_labels

# Filename patterns come straight from the shared config so the loader stays in
# sync with how the centerline pipeline writes its outputs.
_CENTERLINE_PATTERN: str = DEFAULT_CONFIG["feature_toggles"]["centerline_file_pattern"]
_SUPPORT_PATTERN: str = DEFAULT_CONFIG["feature_toggles"][
    "deepsets_support_mask_pattern"
]
_CENTERLINE_SUFFIX: str = _CENTERLINE_PATTERN.replace("{case_id}", "", 1)

_RUN_SUMMARY_NAME = "run_summary.json"
_SINGLE_TIMEPOINT = 1
_DROPPED_MANIFEST_NAME = "dropped_cases.json"
_CACHE_MANIFEST_NAME = "cache_manifest.json"
_FEATURE_SUMMARY_DIRNAME = "feature_summary"
_GRAPH_QC_NAME = "graph_qc.csv"
# Raw (un-imputed, un-encoded) graph-level feature inputs, one row per case,
# written when ``graph_features`` are requested. The clinical imputer/one-hot
# encoder and morphometry imputer are fit per cross-validation fold on the
# training split from this file (see ``gnn.train``), never once over the whole
# cohort at build time -- that whole-cohort fit would leak the validation
# distribution into the training features. See ``gnn.clinical`` / ``gnn.morphometry``.
_GRAPH_FEATURE_INPUTS_NAME = "graph_feature_inputs.csv"
_HIST_BINS = 50

# The only node-feature source implemented today (see module docstring): raw
# DCE-MRI signal, not the vessel-segmentation probability maps. Recorded in
# every cache_manifest.json so a manifest from before this migration -- or a
# hypothetical future vessel_segmentation-sourced build -- is never silently
# treated as compatible with the current code.
_FEATURE_SOURCE = "raw_dce_protocol_baseline_physical_time_all_modes_v4"

# Maps a requested node-feature name to the per-node ``Data`` attribute used to
# populate the corresponding column of ``data.x``.
#
# ``pcr_dummy`` is a sanity-check / leakage-canary feature: every node in a
# graph gets the graph's own ``pcr`` label broadcast onto it, making it a
# perfect predictor of ``data.y`` by construction. It exists only to validate
# that the GNN pipeline (data.x -> GCNConv stack -> pooled logit -> loss) can
# learn an end-to-end trivial signal -- it is computed only when explicitly
# requested via ``node_features`` (see ``_attach_node_features``), never as a
# hardcoded default, and must never be used for real modeling.
_FEATURE_ATTR: dict[str, str] = {
    "peak_time": "peak_time_norm",
    "peak_enhancement": "peak_enhancement",
    "time_to_enhancement": "time_to_enhancement_norm",
    "washin_slope": "washin_slope",
    "washout_slope": "washout_slope",
    "auc_positive": "auc_positive",
    "radius": "radius",
    "pcr_dummy": "pcr_dummy",
}
_DEFAULT_NODE_FEATURES: tuple[str, ...] = ("peak_time", "radius")

# "Time to enhancement" is NaN for any voxel / segment / edge with no detected
# arrival (peak enhancement <= 0, i.e. non-enhancing tissue). A raw NaN cannot
# enter the model -- it propagates to a NaN loss -- so at build time
# ``_finalize_data`` replaces these NaNs with a fixed out-of-range sentinel in
# the normalized [0, 1] TTE space. ``-1.0`` reads as a distinct, learnable "no
# detectable arrival" value rather than being imputed to a plausible arrival
# time, and is applied identically to voxel nodes, segment nodes, and junction
# nodes/edges so the three representations agree. Every no-arrival cell is
# counted per graph (``tte_no_arrival_count`` in ``graph_qc.csv``) so the fill
# stays audited rather than silent. See ``AUDITING_RESULTS.md`` and
# ``gnn/DESIGN_segment_graph.md``.
TTE_NO_ARRIVAL_SENTINEL: float = -1.0
# The feature names (across all three modes) whose column may legitimately be
# NaN and therefore gets the sentinel. Any NaN outside these columns is a bug we
# surface loudly rather than fill.
_TTE_FEATURE_NAMES: frozenset[str] = frozenset(
    {
        "time_to_enhancement",
        "seg_time_to_enhancement_mean",
        "seg_time_to_enhancement_std",
    }
)

# Junction-mode bifurcation-angle features (``gnn.junction_graph``) are NaN for
# a degree-1 junction/endpoint node: it has no neighbor pair to measure an
# opening angle from, a real "not a bifurcation" case, not a bug. This mirrors
# the TTE no-arrival policy above -- NaN is legitimate, sentinel-filled rather
# than imputed, and audited -- but is tracked as its own sentinel/count since
# it is an unrelated missingness mechanism (unrelated feature, unrelated
# cause), not folded into ``tte_no_arrival_count``.
NO_BIFURCATION_SENTINEL: float = -1.0
_BIFURCATION_FEATURE_NAMES: frozenset[str] = frozenset(
    {"bifurcation_angle_mean", "bifurcation_angle_min", "bifurcation_angle_max"}
)

# Node-granularity modes. ``"voxel"`` keeps one node per skeleton voxel;
# ``"segment"`` contracts each vessel segment to a single node (line graph, see
# ``gnn.segment_graph``); ``"junction"`` keeps junction/endpoint voxels as nodes
# and each segment as an edge carrying the segment summary as ``edge_attr``
# (segment-as-edge, Option A, see ``gnn.junction_graph``). See
# ``gnn/DESIGN_segment_graph.md``.
_VOXEL_MODE = "voxel"
_SEGMENT_MODE = "segment"
_JUNCTION_MODE = "junction"
_IMPLEMENTED_NODE_MODES = (_VOXEL_MODE, _SEGMENT_MODE, _JUNCTION_MODE)

# Default segment-mode features mirror the voxel default (one kinetic + one
# geometry feature), expressed in the segment vocabulary.
_DEFAULT_SEGMENT_NODE_FEATURES: tuple[str, ...] = (
    "seg_peak_time_mean",
    "seg_radius_mean",
)
# Junction mode splits features across nodes (per-voxel signal at the junction
# + degree) and edges (the segment summary -- same vocabulary segment mode uses
# for its nodes). The edge default mirrors a broad geometry+kinetics set;
# ``seg_time_to_enhancement_*`` is deliberately left out of the default because
# it is NaN for segments with no detected arrival (opt in explicitly if wanted).
_DEFAULT_JUNCTION_NODE_FEATURES: tuple[str, ...] = ("peak_time", "radius", "degree")
_DEFAULT_JUNCTION_EDGE_FEATURES: tuple[str, ...] = (
    "seg_length",
    "seg_tortuosity",
    "seg_radius_mean",
    "seg_peak_time_mean",
    "seg_peak_enhancement_mean",
    "seg_washin_slope_mean",
    "seg_auc_positive_mean",
)

# Per-mode NODE feature vocabulary (name -> ``Data`` attribute backing that
# column of ``data.x``) and default node feature set. Single source of truth so
# ``__init__`` validation, defaults, and ``_finalize_data``'s column stacking
# stay consistent.
_MODE_FEATURE_ATTR: dict[str, dict[str, str]] = {
    _VOXEL_MODE: _FEATURE_ATTR,
    _SEGMENT_MODE: SEGMENT_FEATURE_ATTR,
    _JUNCTION_MODE: JUNCTION_NODE_FEATURE_ATTR,
}
_MODE_DEFAULT_FEATURES: dict[str, tuple[str, ...]] = {
    _VOXEL_MODE: _DEFAULT_NODE_FEATURES,
    _SEGMENT_MODE: _DEFAULT_SEGMENT_NODE_FEATURES,
    _JUNCTION_MODE: _DEFAULT_JUNCTION_NODE_FEATURES,
}
# EDGE feature vocabulary + default. Only junction mode has edge features; the
# other modes carry none (empty), so ``data.edge_attr`` is never set for them.
_MODE_EDGE_FEATURE_ATTR: dict[str, dict[str, str]] = {
    _JUNCTION_MODE: JUNCTION_EDGE_FEATURE_ATTR,
}
_MODE_DEFAULT_EDGE_FEATURES: dict[str, tuple[str, ...]] = {
    _JUNCTION_MODE: _DEFAULT_JUNCTION_EDGE_FEATURES,
}


class _StageTimings:
    """Accumulate per-stage wall times across cases for coarse profiling.

    This mirrors the intent of ``deepsets.runtime.stage_timer`` but *aggregates*
    across cases (mean / median / max) instead of only logging each call, so the
    dominant stage -- expected to be the 4D time-series load for UChicago-scale
    studies -- is visible after a build.
    """

    def __init__(self) -> None:
        self._stages: dict[str, list[float]] = {}

    def merge(self, stage_samples: dict[str, list[float]]) -> None:
        """Fold per-case timing samples (e.g. from a worker process) into the running totals."""
        for stage, samples in stage_samples.items():
            self._stages.setdefault(stage, []).extend(samples)

    def log_summary(self) -> None:
        """Log mean / median / max seconds for every recorded stage."""
        if not self._stages:
            return
        logging.info("GNN build stage timings (seconds):")
        for stage, samples in self._stages.items():
            logging.info(
                "  %-16s n=%d mean=%.3f median=%.3f max=%.3f",
                stage,
                len(samples),
                statistics.fmean(samples),
                statistics.median(samples),
                max(samples),
            )


@contextmanager
def _stage_timer(stage_samples: dict[str, list[float]], stage: str) -> Iterator[None]:
    """Time a code block and append its elapsed seconds under ``stage``."""
    started = time.perf_counter()
    try:
        yield
    finally:
        stage_samples.setdefault(stage, []).append(time.perf_counter() - started)


def _git_commit() -> str:
    """Return the current HEAD commit hash, for cache-manifest provenance."""
    result = subprocess.run(  # noqa: S603
        ["git", "rev-parse", "HEAD"],  # noqa: S607
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _load_study_metadata(case_id: str, study_dir: Path) -> tuple[list[int], int, bool]:
    """Return timepoints and the explicit kinetic baseline contract.

    These are the ``NNNN`` indices used to name both the vessel-segmentation
    NPZ files and the raw DCE NIfTI phases (``<case_id>_NNNN.nii.gz``), so they
    are what resolves the raw DCE series for this case.

    Raises ``FileNotFoundError`` if the summary is missing, ``KeyError`` if
    ``study_timepoints`` is absent or empty, and propagates
    ``json.JSONDecodeError`` if the file is malformed.
    """
    summary_path = study_dir / _RUN_SUMMARY_NAME
    if not summary_path.exists():
        raise FileNotFoundError(
            f"case={case_id}: {summary_path} not found; run_summary.json is required"
        )
    summary = json.loads(summary_path.read_text())
    study_timepoints = summary.get("study_timepoints")
    if not study_timepoints:
        raise KeyError(
            f"case={case_id}: run_summary.json missing or empty 'study_timepoints' key"
        )
    timepoints = [int(t) for t in study_timepoints]
    alignment_status = summary.get("alignment_qc_status")
    if alignment_status not in (None, "pass", "manually_approved"):
        raise ValueError(
            f"case={case_id}: HR-to-UFAST alignment requires review before "
            "kinetic features can be sampled"
        )
    policy = summary.get("kinetic_feature_policy")
    if policy is None:
        # Legacy cohorts don't carry a protocol baseline contract. Preserve their
        # established single-frame, absolute-enhancement behavior explicitly.
        return timepoints, 1, False
    if alignment_status not in ("pass", "manually_approved"):
        raise ValueError(
            f"case={case_id}: Vanguard kinetic metadata requires explicit "
            "alignment approval before kinetic features can be sampled"
        )
    baseline_frame_count = int(policy["baseline_frame_count"])
    if not 1 <= baseline_frame_count < len(timepoints):
        raise ValueError(
            f"case={case_id}: baseline_frame_count must be in [1, n_timepoints)"
        )
    enhancement = str(policy.get("enhancement", ""))
    if enhancement != "relative_signal_change":
        raise ValueError(
            f"case={case_id}: unsupported kinetic enhancement policy {enhancement!r}"
        )
    if policy.get("time_axis") != "physical_seconds":
        raise ValueError(
            f"case={case_id}: Vanguard kinetic features require physical seconds"
        )
    return timepoints, baseline_frame_count, True


def _load_study_timepoints(case_id: str, study_dir: Path) -> list[int]:
    """Return only the timepoint indices for compatibility with existing callers."""
    timepoints, _, _ = _load_study_metadata(case_id, study_dir)
    return timepoints


def _attach_node_features(
    graph: nx.Graph,
    radius_map: dict[tuple[int, int, int], float],
    dce_4d: np.ndarray,
    time_axis: np.ndarray,
    baseline_frame_count: int,
    relative_enhancement: bool,
    label: int | None,
    node_features: tuple[str, ...],
    *,
    baseline_floor_frac: float = 0.0,
) -> None:
    """Set ``radius`` and the DCE-derived kinetic features on every node.

    ``time_to_enhancement_norm`` is ``NaN`` for nodes with no detected arrival
    (no meaningful enhancement). That NaN is later replaced with
    ``TTE_NO_ARRIVAL_SENTINEL`` when the feature is stacked into ``data.x`` /
    ``data.edge_attr`` (see ``_sentinel_fill_tte`` / ``_finalize_data``), so a
    raw NaN never reaches the model; the per-graph no-arrival count is audited
    via ``graph_qc.csv``'s ``tte_no_arrival_count``.

    ``pcr_dummy`` (the label broadcast onto every node) is only computed and
    attached when it is present in ``node_features`` -- it is a leakage
    canary for pipeline sanity checks, not a default feature, so it must stay
    opt-in rather than something every graph carries.
    """
    duration_seconds = float(time_axis[-1] - time_axis[0])
    if not duration_seconds > 0.0:
        duration_seconds = float(_SINGLE_TIMEPOINT)
    include_pcr_dummy = "pcr_dummy" in node_features
    if include_pcr_dummy and label is None:
        raise ValueError(
            "node feature 'pcr_dummy' requires a label, but none was provided "
            "(label-free pretraining build). Drop 'pcr_dummy' for unlabeled cases."
        )
    for node in graph.nodes():
        x, y, z = int(node[0]), int(node[1]), int(node[2])
        curve = dce_4d[:, z, y, x]
        kinetic = _node_kinetic_features(
            curve,
            time_axis,
            baseline_frame_count=baseline_frame_count,
            relative_enhancement=relative_enhancement,
            baseline_floor_frac=baseline_floor_frac,
        )
        tte_idx = kinetic["tte_idx"]

        attrs = graph.nodes[node]
        attrs["radius"] = float(radius_map[node])
        attrs["baseline_signal"] = float(kinetic["baseline_signal"])
        attrs["peak_time"] = int(kinetic["peak_idx"])
        attrs["peak_time_seconds"] = float(kinetic["peak_time_seconds"])
        attrs["peak_time_norm"] = float(kinetic["peak_time_seconds"]) / duration_seconds
        attrs["peak_enhancement"] = float(kinetic["peak_enhancement"])
        attrs["time_to_enhancement"] = -1 if tte_idx is None else int(tte_idx)
        attrs["time_to_enhancement_seconds"] = (
            float("nan") if tte_idx is None else float(kinetic["tte_seconds"])
        )
        attrs["time_to_enhancement_norm"] = (
            float("nan")
            if tte_idx is None
            else float(kinetic["tte_seconds"]) / duration_seconds
        )
        attrs["washin_slope"] = float(kinetic["washin_slope"])
        attrs["washout_slope"] = float(kinetic["washout_slope"])
        attrs["auc_positive"] = float(kinetic["auc_positive"])
        if include_pcr_dummy:
            attrs["pcr_dummy"] = float(label)


def _sentinel_fill(
    matrix: torch.Tensor,
    feature_names: tuple[str, ...],
    legitimate_nan_names: frozenset[str],
    sentinel: float,
) -> int:
    """Replace NaN with ``sentinel`` in columns named in ``legitimate_nan_names``.

    In place; returns the number of cells filled. Called once per registered
    "legitimate NaN" category (TTE no-arrival, junction no-bifurcation) so each
    keeps its own sentinel value and audit count rather than being merged into
    one generic missingness bucket.
    """
    filled = 0
    for col, name in enumerate(feature_names):
        if name in legitimate_nan_names:
            column = matrix[:, col]
            mask = torch.isnan(column)
            filled += int(mask.sum().item())
            column[mask] = sentinel
    return filled


def _raise_on_unexpected_nan(matrix: torch.Tensor) -> None:
    """Fail loudly if a NaN survives every registered sentinel fill (fail-fast).

    Only ``_TTE_FEATURE_NAMES`` and ``_BIFURCATION_FEATURE_NAMES`` columns may
    legitimately be NaN; a NaN anywhere else is a bug, not a "no signal" case.
    """
    if bool(torch.isnan(matrix).any()):
        raise ValueError(
            "Unexpected NaN in a feature column after sentinel fill; only "
            "registered no-arrival (TTE) or no-bifurcation columns may be NaN."
        )


def _finalize_data(
    data: Data,
    case_id: str,
    label: int | None,
    num_timepoints: int,
    node_features: tuple[str, ...],
    num_connected_components: int,
    feature_attr: dict[str, str],
    edge_features: tuple[str, ...] = (),
    edge_feature_attr: dict[str, str] | None = None,
) -> Data:
    """Assemble ``data.x`` (+ ``data.edge_attr``), the label, and metadata.

    ``feature_attr`` maps each requested node-feature name to the ``Data``
    attribute backing its ``data.x`` column -- ``_FEATURE_ATTR`` (voxel),
    ``SEGMENT_FEATURE_ATTR`` (segment), or ``JUNCTION_NODE_FEATURE_ATTR``
    (junction) -- so the node stacking is identical across modes.

    ``edge_features`` is non-empty only in junction mode (segment-as-edge):
    those columns are stacked into ``data.edge_attr`` from ``edge_feature_attr``
    (``JUNCTION_EDGE_FEATURE_ATTR``), aligned with ``data.edge_index``. Voxel and
    segment modes pass no edge features, so ``data.edge_attr`` is never set.

    Any no-arrival NaN in a time-to-enhancement column of ``data.x`` /
    ``data.edge_attr`` is replaced with ``TTE_NO_ARRIVAL_SENTINEL``, and any
    no-bifurcation NaN in a junction bifurcation-angle column is replaced with
    ``NO_BIFURCATION_SENTINEL`` (see those constants); the counts of filled
    cells are recorded on ``data.tte_no_arrival_count`` /
    ``data.no_bifurcation_count`` for the QC audit, and any NaN outside those
    two registered categories raises.
    """
    columns = [data[feature_attr[name]] for name in node_features]
    data.x = torch.stack([column.float() for column in columns], dim=1)
    no_arrival = _sentinel_fill(
        data.x, node_features, _TTE_FEATURE_NAMES, TTE_NO_ARRIVAL_SENTINEL
    )
    no_bifurcation = _sentinel_fill(
        data.x, node_features, _BIFURCATION_FEATURE_NAMES, NO_BIFURCATION_SENTINEL
    )
    if edge_features:
        edge_columns = [data[edge_feature_attr[name]] for name in edge_features]
        data.edge_attr = torch.stack([col.float() for col in edge_columns], dim=1)
        no_arrival += _sentinel_fill(
            data.edge_attr, edge_features, _TTE_FEATURE_NAMES, TTE_NO_ARRIVAL_SENTINEL
        )
        no_bifurcation += _sentinel_fill(
            data.edge_attr,
            edge_features,
            _BIFURCATION_FEATURE_NAMES,
            NO_BIFURCATION_SENTINEL,
        )
    _raise_on_unexpected_nan(data.x)
    if edge_features:
        _raise_on_unexpected_nan(data.edge_attr)
    data.tte_no_arrival_count = no_arrival
    data.no_bifurcation_count = no_bifurcation
    # Label-free pretraining (design review issue 2): unlabeled cases build a
    # forecasting graph with no ``data.y``. The classification path always passes
    # a label, so its ``data.y`` is unchanged.
    if label is not None:
        data.y = torch.tensor([int(label)], dtype=torch.long)
    data.case_id = case_id
    data.num_timepoints = num_timepoints
    data.num_connected_components = num_connected_components

    # Case-id prefix (e.g. "DUKE_001" -> "DUKE"), not directory structure --
    # the same convention used elsewhere (cohorts/base.py::case_dataset_name,
    # evaluation/selection.py). Robust to a case's skeleton being substituted
    # from a different root entirely (breast_split_mode="single" points
    # mask_path at a workspace directory that isn't under centerline_root, so
    # a directory-relative derivation would raise there).
    dataset = case_id.split("_")[0]
    data.dataset = dataset
    data.site = dataset
    return data


def _build_case(
    case_id: str,
    mask_path: Path,
    label: int | None = None,
    *,
    dce_root: Path,
    node_features: tuple[str, ...],
    node_mode: str,
    edge_features: tuple[str, ...] = (),
    attach_node_series: bool = False,
    baseline_floor_frac: float = 0.0,
) -> tuple[Data, dict[str, list[float]]]:
    """Build one labeled :class:`Data` graph for ``case_id`` in ``node_mode``.

    Standalone (no dataset instance state) so it can run, unmodified, inside a
    worker process when the build is parallelized across cases -- each case's
    graph depends only on its own files, so there is no cross-case state to
    share.

    The kinetic contract (baseline frame count, relative vs. absolute
    enhancement) is acquisition metadata, read per case from the case's
    ``run_summary.json`` (``kinetic_feature_policy``) via ``_load_study_metadata``.

    ``node_mode="voxel"`` keeps one node per skeleton voxel with per-voxel
    features; ``node_mode="segment"`` contracts each vessel segment to a single
    node via ``gnn.segment_graph.build_segment_line_graph`` (Option B, line
    graph); ``node_mode="junction"`` keeps junction/endpoint voxels as nodes and
    each segment as an edge carrying the segment summary as ``edge_attr`` via
    ``gnn.junction_graph.build_junction_graph`` (Option A). Voxel and segment
    mode go through ``from_networkx``; junction mode builds its ``Data``
    directly (it must emit edge features too). All three converge on
    ``_finalize_data``. ``edge_features`` is non-empty only for junction mode.
    """
    stage_samples: dict[str, list[float]] = {}
    # The baseline floor is threaded through the voxel, segment, and junction
    # kinetic paths. The forecasting node-series path (attach_node_series) still
    # calls baseline_relative_curve with the default (unfloored) denominator, so
    # refuse rather than silently half-apply it there.
    if baseline_floor_frac > 0.0 and attach_node_series:
        raise NotImplementedError(
            "baseline_floor_frac > 0 is not yet wired into the forecasting "
            "node-series path (attach_node_series=True). Thread it through "
            "gnn/pretrain/node_series.py before using it there."
        )
    study_dir = mask_path.parent

    with _stage_timer(stage_samples, "mask_load"):
        skeleton = np.load(mask_path).astype(bool, copy=False)
        support_path = study_dir / _SUPPORT_PATTERN.format(case_id=case_id)
        if not support_path.exists():
            raise FileNotFoundError(
                f"Support mask not found for {case_id}: {support_path}"
            )
        support = np.load(support_path).astype(bool, copy=False)

    if skeleton.ndim != NDIM_3D or not skeleton.any():
        raise ValueError(f"Empty or non-3D skeleton for {case_id}")
    if skeleton.shape != support.shape:
        raise ValueError(
            f"Skeleton/support shape mismatch for {case_id}: "
            f"{skeleton.shape} vs {support.shape}"
        )

    with _stage_timer(stage_samples, "graph_build"):
        segments = edges_to_segments(mask_to_edges_bitmask(skeleton))
        if segments.size == 0:
            raise ValueError(f"Skeleton for {case_id} has zero segments")
        voxel_graph = segments_to_graph(segments)
    if voxel_graph.number_of_nodes() == 0:
        raise ValueError(f"Graph for {case_id} has zero nodes")

    radius_map = obtain_radius_map(support, voxel_graph)

    (
        study_timepoints,
        baseline_frame_count,
        relative_enhancement,
    ) = _load_study_metadata(case_id, study_dir)
    with _stage_timer(stage_samples, "timeseries_load"):
        dce_paths = discover_raw_dce_paths(dce_root, case_id, study_timepoints)
        dce_4d = load_raw_dce_series(dce_paths, expected_shape_zyx=support.shape)
        physical_times_seconds = load_raw_dce_times(dce_root, case_id, study_timepoints)
    if dce_4d.shape[1:] != support.shape:
        raise ValueError(
            f"Aligned raw DCE shape for {case_id} {dce_4d.shape[1:]} does not "
            f"match support mask shape {support.shape}"
        )
    num_timepoints = int(dce_4d.shape[0])
    time_axis = _time_axis_from_study_timepoints(
        study_timepoints,
        physical_times_seconds,
        require_physical_seconds=relative_enhancement,
    )

    if node_mode == _JUNCTION_MODE:
        # Junction mode builds Data directly (node + edge features) and records
        # its own connected-component count on the junction graph.
        with _stage_timer(stage_samples, "junction_build"):
            data = build_junction_graph(
                voxel_graph,
                radius_map,
                dce_4d,
                time_axis,
                baseline_frame_count=baseline_frame_count,
                relative_enhancement=relative_enhancement,
                baseline_floor_frac=baseline_floor_frac,
            )
        num_connected_components = int(data.num_connected_components)
    else:
        if node_mode == _SEGMENT_MODE:
            with _stage_timer(stage_samples, "segment_build"):
                graph = build_segment_line_graph(
                    voxel_graph,
                    radius_map,
                    dce_4d,
                    time_axis,
                    baseline_frame_count=baseline_frame_count,
                    relative_enhancement=relative_enhancement,
                    baseline_floor_frac=baseline_floor_frac,
                )
        else:
            with _stage_timer(stage_samples, "peak_time"):
                _attach_node_features(
                    voxel_graph,
                    radius_map,
                    dce_4d,
                    time_axis,
                    baseline_frame_count,
                    relative_enhancement,
                    label,
                    node_features,
                    baseline_floor_frac=baseline_floor_frac,
                )
            graph = voxel_graph

        # Must be counted on the (modeled) nx.Graph itself -- from_networkx()
        # below discards it, and edge_index on the resulting Data is
        # directed-both-ways, which is not a valid input to
        # nx.connected_components without reconstruction. Counted on ``graph``
        # (voxel or line graph) so it stays consistent with the
        # num_nodes/num_edges QC reports for the same object.
        num_connected_components = nx.number_connected_components(graph)
        with _stage_timer(stage_samples, "from_networkx"):
            data = from_networkx(graph)

    data = _finalize_data(
        data,
        case_id,
        label,
        num_timepoints,
        node_features,
        num_connected_components,
        feature_attr=_MODE_FEATURE_ATTR[node_mode],
        edge_features=edge_features,
        edge_feature_attr=_MODE_EDGE_FEATURE_ATTR.get(node_mode),
    )

    # Opt-in per-node contrast time-series for the forecasting pretext task
    # (gnn.pretrain). Each row aligns with the corresponding ``data.x`` row:
    #   - voxel: ``list(voxel_graph.nodes())`` is exactly the order
    #     ``from_networkx`` used to stack ``data.x``.
    #   - segment: ``segment_node_series`` uses the same
    #     ``enumerate(extract_segments(voxel_graph))`` order that
    #     ``build_segment_line_graph`` numbered its segment-nodes with.
    #   - junction: ``junction_node_series`` samples the raw curve at each
    #     junction voxel, ordered by ``data.pos`` (== ``build_junction_graph``'s
    #     ``ordered_nodes``). First-pass target = raw junction-voxel curve
    #     (design doc §4.3; flow/derivative alternative deferred).
    # Default off -> the classification build path is byte-for-byte unchanged.
    if attach_node_series:
        from gnn.pretrain.node_series import (
            junction_node_series,
            segment_node_series,
            voxel_node_series,
        )

        if node_mode == _VOXEL_MODE:
            node_series = voxel_node_series(
                dce_4d,
                list(voxel_graph.nodes()),
                baseline_frame_count=baseline_frame_count,
                relative_enhancement=relative_enhancement,
            )
        elif node_mode == _SEGMENT_MODE:
            node_series = segment_node_series(
                voxel_graph,
                dce_4d,
                baseline_frame_count=baseline_frame_count,
                relative_enhancement=relative_enhancement,
            )
        elif node_mode == _JUNCTION_MODE:
            node_series = junction_node_series(
                dce_4d,
                data.pos.numpy(),
                baseline_frame_count=baseline_frame_count,
                relative_enhancement=relative_enhancement,
            )
        else:
            raise ValueError(
                f"attach_node_series=True: unknown node_mode {node_mode!r}."
            )
        data.node_series = torch.tensor(node_series, dtype=torch.float)
        # Physical acquisition seconds for the forecasting time axis (issue 3a):
        # the same ``time_axis`` the kinetic path uses, so forecasting sees the
        # real (irregular) UFAST cadence rather than frame indices.
        data.node_times = torch.tensor(time_axis, dtype=torch.float)
        # Protocol baseline length, so the forecasting tiler can drop precontrast
        # baseline frames from its windows (issue 3b) while still using them for S0.
        data.baseline_frame_count = int(baseline_frame_count)
    return data, stage_samples


class VanguardCenterlineDataset(InMemoryDataset):
    """Raw vessel-graph dataset built from saved centerline outputs.

    One graph is produced per case (named ``<case_id>_graph``): nodes are
    skeleton voxels keyed by ``(x, y, z)``, edges connect 26-connected voxels,
    and node features are stacked into ``data.x`` in the order given by
    ``node_features``. A binary ``data.y`` label is required to build a graph;
    cases with no matching label are **dropped** (not built), which is logged
    loudly and recorded in ``dropped_case_ids`` / ``processed/dropped_cases.json``
    every time the dataset is built or loaded. If the dropped fraction exceeds
    ``max_missing_label_frac`` the whole build raises instead of silently
    training on a shrunken cohort. Degenerate geometry (empty skeleton, zero
    segments, shape mismatch, ...) still raises immediately -- that is a data
    problem, not an expected missing-label case.

    A fresh build also writes ``processed/cache_manifest.json`` recording the
    settings that determine the cached graphs' content (roots, labels, node
    features, feature source, ...), plus the code commit, graph count, label
    counts, and build timestamp. Every later load compares the requested
    settings against this manifest and raises ``RuntimeError`` on a mismatch
    (see ``allow_manifest_mismatch``), so a stale cache built under different
    settings can never be silently reused.

    A fresh build also writes ``processed/graph_qc.csv``, one row per graph
    (``case_id``, ``dataset``, ``pcr``, ``num_nodes``, ``num_edges``,
    ``num_connected_components``, ``mean_degree``, missing/NaN feature
    counts, and per-feature min/max/mean/std), plus the confound-audit plots
    derivable from it under ``processed/graph_qc_plots/`` -- see
    ``_write_graph_qc``.

    Args:
        root: The centerline ``studies`` tree containing per-case output
            directories with ``*_skeleton_4d_exam_mask.npy`` files.
        labels_path: CSV/JSON labels file passed to
            :func:`tabular.cohort.load_labels`.
        dce_root: Root of the raw DCE-MRI NIfTI tree
            (``<dce_root>/<case_id>/<case_id>_NNNN.nii.gz``), used to compute the
            DCE-derived kinetic node features (see ``gnn.raw_dce``).
        cache_dir: Where the collated ``processed/`` cache is written. Defaults
            to ``<root>/gnn_cache`` so the source tree can stay read-only when an
            explicit path is given.
        cases: Optional whitelist of case IDs to include.
        no_cache: Skip reading and writing the on-disk cache; always rebuild from
            source. Useful during development to avoid stale-cache surprises.
        node_mode: Node granularity. ``"voxel"`` (default) keeps one node per
            skeleton voxel; ``"segment"`` contracts each vessel segment to a single
            node (line graph, see ``gnn.segment_graph``); ``"junction"`` keeps
            junction/endpoint voxels as nodes and each segment as an edge carrying
            the segment summary as ``edge_attr`` (segment-as-edge, Option A, see
            ``gnn.junction_graph``). The mode selects the node (and, for junction,
            edge) feature vocabulary and defaults. See ``gnn/DESIGN_segment_graph.md``.
        node_features: Node-feature names, in ``data.x`` column order. ``None``
            (default) resolves to the mode's default feature set. Supported names
            depend on ``node_mode``:

            - **voxel:** ``"peak_time"`` (normalized time-to-peak enhancement),
              ``"peak_enhancement"``, ``"time_to_enhancement"`` (normalized arrival
              time; ``NaN`` for nodes with no detected enhancement),
              ``"washin_slope"``, ``"auc_positive"``, ``"radius"``, and
              ``"pcr_dummy"`` (the graph's ``pcr`` label broadcast onto every node
              -- a leakage-canary feature for pipeline sanity checks only; opt-in,
              never included unless named explicitly). The kinetic features are
              sampled from the raw DCE curve (``curve = dce_4d[:, z, y, x]``).
              Vanguard UFAST uses its protocol baseline mean, relative signal
              change, and physical seconds; legacy cohorts retain single-frame
              absolute enhancement. See ``gnn.kinetics.node_kinetic_features``.
            - **segment:** geometry (``"seg_length"``, ``"seg_tortuosity"``,
              ``"seg_volume"``, ``"seg_radius_{mean,std,median,min,max}"``,
              ``"seg_curvature_{mean,std,max}"``), the same per-voxel kinetics
              summarized (mean/std) along the segment
              (``"seg_{peak_time,peak_enhancement,time_to_enhancement,washin_slope,
              auc_positive}_{mean,std}"``), and ``"seg_num_voxels"``. See
              ``gnn.segment_graph.SEGMENT_FEATURE_ATTR``. (``"pcr_dummy"`` is
              voxel-only for now.)
            - **junction:** per-voxel signal at the junction voxel
              (``"peak_time"``, ``"peak_enhancement"``, ``"time_to_enhancement"``,
              ``"washin_slope"``, ``"auc_positive"``, ``"radius"``) plus
              ``"degree"``. The segment summary goes on ``edge_features``, not
              here. See ``gnn.junction_graph.JUNCTION_NODE_FEATURE_ATTR``.
        edge_features: Edge-feature names, in ``data.edge_attr`` column order.
            Only valid (and required) for ``node_mode="junction"``, where each
            segment's summary rides on its edge -- the same ``seg_*`` vocabulary
            segment mode uses for its nodes
            (``gnn.junction_graph.JUNCTION_EDGE_FEATURE_ATTR``). ``None`` resolves
            to the junction default; must be empty/unset for voxel and segment
            mode (their graphs carry no edge features).
        label_column: Binary label column in the labels file.
        max_missing_label_frac: Maximum fraction of discovered cases allowed to
            be dropped for lacking a ``label_column`` value. Every drop is
            logged regardless; exceeding this fraction raises ``RuntimeError``
            instead of building a silently-shrunken cohort. Default 0.1 (10%).
        profile: When true, accumulate and log per-stage timings.
        num_workers: Number of process-pool workers used to build cases in
            parallel (each case's graph depends only on its own files, so
            cases are built independently). Default 1 (sequential -- safe to
            run on a login node); pass more to match allocated Slurm CPUs for
            a full cluster build.
        allow_manifest_mismatch: A fresh build always writes
            ``processed/cache_manifest.json`` recording the settings that
            determine the cached graphs' content (roots, labels, node
            features, ...); every later load compares the requested settings
            against it and raises ``RuntimeError`` on a mismatch, so a config
            change can never be silently served from a stale cache built
            under different settings. Set this to ``True`` to explicitly
            override that check (e.g. you know the mismatch is benign) --
            never use it to paper over an unexplained mismatch.
        breast_split_mode: ``None`` (default, current exam-level-graph
            behavior, fully backward compatible) or ``"single"`` to harmonize
            bilateral cases down to their tumor-bearing breast (see
            ``gnn.breast_split``, ``gnn.build_single_breast_skeletons``).
            Native unilateral cases are always used unchanged -- there is
            nothing to split. Requires ``breast_split_skeleton_root`` and a
            clinical source (``patient_info_dir`` or ``clinical_excel``) to
            look up each case's ``bilateral`` flag.
        breast_split_skeleton_root: Root of precomputed single-breast
            skeletons (``<root>/<dataset>/<case_id>/<case_id>_skeleton_4d_single_breast_mask.npy``),
            written by ``gnn.build_single_breast_skeletons``. Required when
            ``breast_split_mode="single"``.
        max_missing_breast_split_frac: Maximum fraction of discovered cases
            allowed to be dropped because they're bilateral but were excluded
            by the splitter (unknown tumor side, tumor straddling both
            sides, ...) or have no clinical row to determine laterality from.
            Every drop is logged regardless; exceeding this fraction raises
            ``RuntimeError`` instead of silently training on a
            harmonized-cohort-mismatched cohort. Default 0.1 (10%), only
            consulted when ``breast_split_mode`` is set.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        labels_path: str | Path,
        dce_root: str | Path,
        cache_dir: str | Path | None = None,
        cases: Sequence[str] | None = None,
        no_cache: bool = False,
        node_mode: str = _VOXEL_MODE,
        node_features: Sequence[str] | None = None,
        edge_features: Sequence[str] | None = None,
        graph_features: Sequence[str] | None = None,
        patient_info_dir: str | Path | None = None,
        clinical_excel: str | Path | None = None,
        max_missing_clinical_frac: float = 0.1,
        id_column: str = "case_id",
        label_column: str = "pcr",
        max_missing_label_frac: float = 0.1,
        kinetic_baseline_floor_frac: float = 0.0,
        profile: bool = False,
        num_workers: int = 1,
        allow_manifest_mismatch: bool = False,
        breast_split_mode: str | None = None,
        breast_split_skeleton_root: str | Path | None = None,
        max_missing_breast_split_frac: float = 0.1,
        transform: object = None,
        pre_transform: object = None,
    ) -> None:
        if labels_path is None:
            raise ValueError("labels_path is required; every graph must carry a label.")
        if dce_root is None:
            raise ValueError(
                "dce_root is required; kinetic node features are sampled from the "
                "raw DCE series, not the vessel-segmentation NPZ timepoints."
            )
        if node_mode not in _IMPLEMENTED_NODE_MODES:
            raise NotImplementedError(
                f"node_mode={node_mode!r} is not implemented; use one of "
                f"{list(_IMPLEMENTED_NODE_MODES)}. ('junction' / segment-as-edge "
                "is planned next -- see gnn/DESIGN_segment_graph.md.)"
            )
        feature_attr = _MODE_FEATURE_ATTR[node_mode]
        # Each mode has its own default and vocabulary: voxel-mode features are
        # not valid segment-mode features and vice versa, so a None here resolves
        # to the *mode's* default rather than a single global one.
        if node_features is None:
            node_features = _MODE_DEFAULT_FEATURES[node_mode]
        unknown = [f for f in node_features if f not in feature_attr]
        if unknown:
            raise ValueError(
                f"Unknown node_features {unknown} for node_mode={node_mode!r}; "
                f"supported: {sorted(feature_attr)}"
            )
        # Edge features exist only for junction mode (segment-as-edge). Resolve a
        # None to the mode's default there; require it be empty everywhere else,
        # since voxel/segment graphs carry no edge features.
        edge_feature_attr = _MODE_EDGE_FEATURE_ATTR.get(node_mode)
        if edge_feature_attr is None:
            if edge_features:
                raise ValueError(
                    f"edge_features are only supported for node_mode='junction', "
                    f"not {node_mode!r}; got {list(edge_features)}."
                )
            edge_features = ()
        else:
            if edge_features is None:
                edge_features = _MODE_DEFAULT_EDGE_FEATURES[node_mode]
            if not edge_features:
                raise ValueError(
                    "node_mode='junction' requires at least one edge feature "
                    "(the segment summary rides on the edges)."
                )
            unknown_edges = [f for f in edge_features if f not in edge_feature_attr]
            if unknown_edges:
                raise ValueError(
                    f"Unknown edge_features {unknown_edges} for "
                    f"node_mode={node_mode!r}; supported: {sorted(edge_feature_attr)}"
                )
        if not 0.0 <= max_missing_label_frac <= 1.0:
            raise ValueError(
                f"max_missing_label_frac must be in [0, 1], got {max_missing_label_frac}"
            )
        if num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {num_workers}")
        graph_features = tuple(graph_features) if graph_features else ()
        # graph_features spans three independent vocabularies -- clinical
        # (external, joined by case_id), graph-derived (already on the built
        # Data object), and morphometry (per-case JSON) -- so a name's source
        # is resolved by membership, not by which mechanism the caller
        # intended. See gnn.clinical / gnn.graph_derived / gnn.morphometry.
        _ALL_GRAPH_FEATURE_COLUMNS = (
            ALL_CLINICAL_COLUMNS | set(GRAPH_DERIVED_COLUMNS) | set(MORPHOMETRY_COLUMNS)
        )
        unknown_graph = [
            f for f in graph_features if f not in _ALL_GRAPH_FEATURE_COLUMNS
        ]
        if unknown_graph:
            raise ValueError(
                f"Unknown graph_features {unknown_graph}; supported: "
                f"{sorted(_ALL_GRAPH_FEATURE_COLUMNS)}"
            )
        _requests_clinical = any(f in ALL_CLINICAL_COLUMNS for f in graph_features)
        if _requests_clinical and not (patient_info_dir or clinical_excel):
            raise ValueError(
                "graph_features includes clinical column(s), which require "
                "patient_info_dir or clinical_excel (a clinical data source) "
                "to be set."
            )
        if not 0.0 <= max_missing_clinical_frac <= 1.0:
            raise ValueError(
                f"max_missing_clinical_frac must be in [0, 1], got "
                f"{max_missing_clinical_frac}"
            )
        if breast_split_mode is not None:
            if breast_split_mode != "single":
                raise ValueError(
                    f"breast_split_mode={breast_split_mode!r} is not supported; "
                    "only 'single' (or None to disable) is implemented."
                )
            if breast_split_skeleton_root is None:
                raise ValueError(
                    "breast_split_mode='single' requires breast_split_skeleton_root "
                    "(see gnn.build_single_breast_skeletons)."
                )
            if not (patient_info_dir or clinical_excel):
                raise ValueError(
                    "breast_split_mode='single' requires patient_info_dir or "
                    "clinical_excel to look up each case's bilateral flag."
                )
        if not 0.0 <= max_missing_breast_split_frac <= 1.0:
            raise ValueError(
                f"max_missing_breast_split_frac must be in [0, 1], got "
                f"{max_missing_breast_split_frac}"
            )

        self._centerline_root = Path(root)
        self._labels_path = Path(labels_path)
        self._dce_root = Path(dce_root)
        self._cases = set(cases) if cases is not None else None
        self._no_cache = no_cache
        self._data_list_cache: list[Data] | None = None
        self._node_mode = node_mode
        self._node_features = tuple(node_features)
        self._edge_features = tuple(edge_features)
        self._graph_features = graph_features
        self._patient_info_dir = Path(patient_info_dir) if patient_info_dir else None
        self._clinical_excel = Path(clinical_excel) if clinical_excel else None
        self._max_missing_clinical_frac = max_missing_clinical_frac
        self._id_column = id_column
        self._label_column = label_column
        self._max_missing_label_frac = max_missing_label_frac
        if kinetic_baseline_floor_frac < 0.0:
            raise ValueError("kinetic_baseline_floor_frac must be >= 0")
        self._kinetic_baseline_floor_frac = float(kinetic_baseline_floor_frac)
        self._profile = profile
        self._num_workers = num_workers
        self._allow_manifest_mismatch = allow_manifest_mismatch
        self._breast_split_mode = breast_split_mode
        self._breast_split_skeleton_root = (
            Path(breast_split_skeleton_root) if breast_split_skeleton_root else None
        )
        self._max_missing_breast_split_frac = max_missing_breast_split_frac
        self._timings = _StageTimings()
        self.dropped_case_ids: list[str] = []
        # Populated by _check_cache_manifest() from cache_manifest.json; used by
        # _reslice_for_requested_features() to narrow a cached feature superset
        # down to what was actually requested (see both methods' docstrings).
        self._cached_node_features: list[str] | None = None
        self._cached_edge_features: list[str] | None = None
        # Raw graph-level feature inputs held in memory for no_cache runs; on a
        # cached build they are written to / read from _GRAPH_FEATURE_INPUTS_NAME
        # instead (see _write_graph_feature_inputs / load_graph_feature_inputs).
        self._graph_feature_inputs: pd.DataFrame | None = None

        resolved_cache = (
            Path(cache_dir)
            if cache_dir is not None
            else self._centerline_root / "gnn_cache"
        )
        super().__init__(
            str(resolved_cache),
            transform=transform,
            pre_transform=pre_transform,
        )
        self._load_processed()

    # -- InMemoryDataset plumbing ------------------------------------------

    @property
    def raw_dir(self) -> str:
        """Raw data lives in the centerline tree, not under ``root/raw``."""
        return str(self._centerline_root)

    @property
    def raw_file_names(self) -> list[str]:
        """No fixed raw manifest; ``process`` globs the tree directly."""
        return []

    @property
    def processed_file_names(self) -> list[str]:
        """Single collated cache file."""
        return ["data.pt"]

    def download(self) -> None:
        """No-op: centerline outputs are produced upstream, never downloaded."""

    def _process(self) -> None:
        if self._no_cache:
            self.process()
        else:
            super()._process()

    def _load_processed(self) -> None:
        """Restore the collated tensors."""
        if self._no_cache:
            if self._data_list_cache is None:
                raise RuntimeError("no_cache=True but process() has not run yet")
            self.data, self.slices = self.collate(self._data_list_cache)
        else:
            self._check_cache_manifest()
            self.data, self.slices = torch.load(self.processed_paths[0])
            self._reslice_for_requested_features()
            self._reload_dropped_manifest()

    def _reslice_for_requested_features(self) -> None:
        """Narrow ``data.x``/``data.edge_attr`` to the requested feature subset.

        ``_check_cache_manifest`` allows ``node_features``/``edge_features`` to
        be any *subset* of what the cache was built with (see its docstring),
        so a cache built once with a feature superset can serve any narrower
        request without a rebuild -- this is the other half of that contract:
        the tensors loaded from disk still have every cached column, so a
        genuine subset request needs an explicit column-index reslice here.
        No-op when the request matches the cache exactly (the common case), or
        when no cached feature list is known (e.g. a pre-manifest cache loaded
        via ``allow_manifest_mismatch``).
        """
        cached_nf = self._cached_node_features
        if cached_nf is not None and list(self._node_features) != cached_nf:
            keep_idx = [cached_nf.index(name) for name in self._node_features]
            self.data.x = self.data.x[:, keep_idx]
        cached_ef = self._cached_edge_features
        if cached_ef and list(self._edge_features) != cached_ef:
            keep_idx = [cached_ef.index(name) for name in self._edge_features]
            self.data.edge_attr = self.data.edge_attr[:, keep_idx]

    def _reload_dropped_manifest(self) -> None:
        """Restore and re-log dropped-case bookkeeping on a cache hit.

        ``process()`` only runs once per cache; every later load of the same
        cache still goes through here, so this is what keeps missing-label
        (and missing-clinical) drops visible instead of only being logged the
        one time the cache was built.
        """
        manifest_path = Path(self.processed_dir) / _DROPPED_MANIFEST_NAME
        if not manifest_path.exists():
            return
        manifest = json.loads(manifest_path.read_text())
        self.dropped_case_ids = manifest["dropped_case_ids"]
        if self.dropped_case_ids:
            # dropped_reasons is absent on caches built before graph_features
            # existed; fall back to the original label-only phrasing so old
            # caches still log sensibly.
            reasons = manifest.get("dropped_reasons")
            reason_text = (
                dict(Counter(reasons.values()))
                if reasons
                else f"missing {manifest['label_column']!r} label"
            )
            logging.warning(
                "GNN dataset (cached): %d/%d case(s) (%.1f%%) were dropped (%s): %s",
                len(self.dropped_case_ids),
                manifest["num_discovered"],
                manifest["dropped_frac"] * 100,
                reason_text,
                self.dropped_case_ids,
            )

    def _manifest_settings(self) -> dict[str, object]:
        """Settings that determine what the cached graphs actually contain.

        This is the single source of truth for both writing
        ``cache_manifest.json`` on a fresh build and validating it on every
        later load -- see ``_write_cache_manifest`` / ``_check_cache_manifest``.
        Execution-only knobs (``num_workers``, ``profile``,
        ``max_missing_label_frac``) are deliberately excluded: they don't
        change what ends up in the cache, only how it's built.
        """
        settings: dict[str, object] = {
            "centerline_root": str(self._centerline_root),
            "dce_root": str(self._dce_root),
            "labels_path": str(self._labels_path),
            "id_column": self._id_column,
            "label_column": self._label_column,
            "cases": sorted(self._cases) if self._cases is not None else None,
            "node_mode": self._node_mode,
            "node_features": list(self._node_features),
            "feature_source": _FEATURE_SOURCE,
        }
        # Kinetic baseline floor: recorded only when active so pre-floor caches
        # (which have no such key) stay schema-compatible. _check_cache_manifest
        # normalizes a missing cached value to 0.0, so a floored cache never
        # silently serves an unfloored request or vice versa.
        if self._kinetic_baseline_floor_frac > 0.0:
            settings["kinetic_baseline_floor_frac"] = self._kinetic_baseline_floor_frac
        # Only junction mode has edge features. Recording the key only when it's
        # non-empty keeps voxel/segment manifests (and the caches already built
        # under them) schema-compatible -- a pre-edge_features cache has no such
        # key and must still validate against a voxel/segment request.
        if self._edge_features:
            settings["edge_features"] = list(self._edge_features)
        # graph_features (clinical covariates) are opt-in and, unlike
        # node_features/edge_features, are NOT eligible for the subset-request
        # reslice (see _check_cache_manifest's subset_keys) -- the raw
        # graph-feature-input sidecar only stores the columns that were
        # requested at build time, so a "narrower" request still needs its own
        # sidecar. Requesting a different graph_features list always requires a
        # rebuild (or a sidecar regeneration -- see regenerate_graph_feature_inputs).
        if self._graph_features:
            settings["graph_features"] = list(self._graph_features)
            settings["clinical_source"] = str(
                self._patient_info_dir or self._clinical_excel
            )
        # breast_split_mode changes which skeleton backs a bilateral case's
        # graph (see _resolve_breast_split_paths) -- a cache built with it
        # must never be silently reused as if it were the unsplit cohort, or
        # vice versa. Always recorded (even as None) so a mismatch is caught
        # in *both* directions: unlike graph_features/edge_features below,
        # there's no valid "narrower request" reading of a missing
        # breast_split_mode against a cache that was actually built with one.
        settings["breast_split_mode"] = self._breast_split_mode
        if self._breast_split_mode is not None:
            settings["breast_split_skeleton_root"] = str(
                self._breast_split_skeleton_root
            )
        return settings

    def _check_cache_manifest(self) -> None:
        """Fail loudly if an on-disk cache was built under different settings.

        Without this, a stale cache (different centerline root, labels,
        node features, ...) could be silently reused just because
        ``processed/data.pt`` happens to exist at ``cache_dir`` -- see the
        code-review item this implements. Raises if the manifest is missing
        (a cache built before this check existed) or if any recorded setting
        differs from what's currently requested, unless
        ``allow_manifest_mismatch=True`` was passed.

        ``node_features``/``edge_features`` are the one exception to "differs
        means raise": a request for a *subset* of the cached feature list is
        allowed even without ``allow_manifest_mismatch``, since
        ``_reslice_for_requested_features`` narrows the loaded tensors down to
        exactly that subset afterwards. This lets a cache built once with a
        feature superset serve any narrower request (e.g. leave-one-covariate
        LOCO sweeps) without a rebuild. Every other setting (roots, labels,
        node_mode, ...) still requires exact equality.
        """
        manifest_path = Path(self.processed_dir) / _CACHE_MANIFEST_NAME
        if not manifest_path.exists() and not self._allow_manifest_mismatch:
            raise RuntimeError(
                f"Cache at {self.processed_dir} has no {_CACHE_MANIFEST_NAME} "
                "(built before cache-manifest tracking was added). Its "
                "settings can't be verified. Rebuild the cache (e.g. "
                "--force-rebuild in gnn/build_dataset.py) so a manifest is "
                "recorded, or pass allow_manifest_mismatch=True to bypass "
                "this check."
            )
        if not manifest_path.exists():
            return
        manifest = json.loads(manifest_path.read_text())
        requested = self._manifest_settings()
        self._cached_node_features = list(manifest.get("node_features") or [])
        self._cached_edge_features = list(manifest.get("edge_features") or [])
        subset_keys = {"node_features", "edge_features"}
        mismatched: dict[str, object] = {}
        for key, value in requested.items():
            cached_value = manifest.get(key)
            if key in subset_keys:
                if not set(value) <= set(cached_value or []):
                    mismatched[key] = {"cached": cached_value, "requested": value}
            elif cached_value != value:
                mismatched[key] = {"cached": cached_value, "requested": value}
        # Kinetic baseline floor is recorded only when active, so it may be
        # absent from either side; compare it explicitly with absent==0.0 so a
        # floored cache can't serve an unfloored request (or vice versa) even
        # though the key is missing from the requested settings.
        floor_key = "kinetic_baseline_floor_frac"
        cached_floor = manifest.get(floor_key) or 0.0
        requested_floor = requested.get(floor_key) or 0.0
        if cached_floor != requested_floor:
            mismatched[floor_key] = {
                "cached": cached_floor,
                "requested": requested_floor,
            }
        if mismatched and not self._allow_manifest_mismatch:
            raise RuntimeError(
                f"Cache at {self.processed_dir} was built with different "
                f"settings than requested, so it may not reflect the data/"
                f"features you asked for: {json.dumps(mismatched, indent=2)}. "
                "Point at a different cache_dir, rebuild (--force-rebuild), "
                "or pass allow_manifest_mismatch=True to explicitly override."
            )

    def _write_cache_manifest(self, data_list: list[Data]) -> None:
        """Persist the settings, code commit, and summary stats for a fresh build.

        Written once per cache build (never on a cache hit) so every later
        load of this cache has something to validate itself against -- see
        ``_check_cache_manifest``.
        """
        if self._no_cache:
            return
        label_counts = Counter(int(data.y.item()) for data in data_list)
        manifest = {
            **self._manifest_settings(),
            "code_commit": _git_commit(),
            "num_graphs": len(data_list),
            "label_counts": {str(k): v for k, v in sorted(label_counts.items())},
            "built_at": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path = Path(self.processed_dir) / _CACHE_MANIFEST_NAME
        manifest_path.write_text(json.dumps(manifest, indent=2))

    def _save_processed(self, data_list: list[Data]) -> None:
        """Persist the collated dataset, or keep in memory when no_cache=True."""
        if self._no_cache:
            self._data_list_cache = data_list
            return
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    # -- build --------------------------------------------------------------

    def process(self) -> None:
        """Discover cases, build one labeled graph each, and collate the cache."""
        labels = self._load_label_map()
        discovered = self._discover_cases()
        total_discovered = len(discovered)
        logging.info(
            "GNN build: %d candidate case(s) under %s",
            total_discovered,
            self._centerline_root,
        )

        # graph_features spans three independent vocabularies -- see
        # gnn.clinical / gnn.graph_derived / gnn.morphometry and
        # _build_graph_feature_inputs's docstring.
        requested_clinical, requested_graph_derived, requested_morphometry = (
            self._requested_graph_feature_groups()
        )
        # Computed from the pre-breast-split discovery: morphometry/graph_qc
        # always resolve against the original exam-level study directory,
        # even for a case whose *skeleton* gets substituted below.
        case_id_to_study_dir = {
            case_id: mask_path.parent for case_id, mask_path in discovered
        }

        dropped: list[str] = []
        dropped_reasons: dict[str, str] = {}
        if self._breast_split_mode == "single":
            resolved_paths, breast_split_dropped = self._resolve_breast_split_paths(
                discovered
            )
            dropped.extend(breast_split_dropped)
            dropped_reasons.update(breast_split_dropped)
            discovered = [
                (case_id, resolved_paths[case_id])
                for case_id, _ in discovered
                if case_id in resolved_paths
            ]

        # Clinical covariates need the clinical source loaded upfront: a case
        # with no clinical row is dropped before the expensive per-case graph
        # build runs, the same way a missing label is -- not as a post-build
        # filter, so we don't waste compute on cases we'll drop. Graph-derived
        # and morphometry features need no such upfront drop (graph-derived is
        # always present on a built graph; morphometry is present for 100% of
        # the real cohort and hard-fails instead, see
        # _resolve_morphometry_paths).
        clinical_df: pd.DataFrame | None = None
        clinical_case_ids: set[str] | None = None
        if requested_clinical:
            clinical_df = self._load_clinical_df()
            clinical_case_ids = set(clinical_df["case_id"])

        tasks: list[tuple[str, Path, int]] = []
        for case_id, mask_path in discovered:
            label = labels.get(case_id)
            if label is None:
                dropped.append(case_id)
                dropped_reasons[case_id] = "missing_label"
                continue
            if clinical_case_ids is not None and case_id not in clinical_case_ids:
                dropped.append(case_id)
                dropped_reasons[case_id] = "missing_clinical"
                continue
            tasks.append((case_id, mask_path, label))

        data_list: list[Data] = []
        for case_id, data in self._build_cases(tasks):
            torch.save(data, Path(self.processed_dir) / f"{case_id}_graph.pt")
            data_list.append(data)

        self.dropped_case_ids = dropped
        self._write_dropped_manifest(dropped, dropped_reasons, total_discovered)
        if dropped:
            dropped_frac = len(dropped) / total_discovered
            by_reason = dict(Counter(dropped_reasons.values()))
            logging.warning(
                "GNN build: dropped %d/%d case(s) (%.1f%%): %s: %s",
                len(dropped),
                total_discovered,
                dropped_frac * 100,
                by_reason,
                dropped,
            )
            missing_label = [
                c for c in dropped if dropped_reasons[c] == "missing_label"
            ]
            if missing_label:
                label_frac = len(missing_label) / total_discovered
                if label_frac > self._max_missing_label_frac:
                    raise RuntimeError(
                        f"{len(missing_label)}/{total_discovered} cases "
                        f"({label_frac:.1%}) are missing a {self._label_column!r} "
                        f"label in {self._labels_path}, exceeding "
                        f"max_missing_label_frac={self._max_missing_label_frac}. "
                        "Fix the labels file, or pass a higher "
                        "max_missing_label_frac if this many missing labels is "
                        "expected."
                    )
            missing_clinical = [
                c for c in dropped if dropped_reasons[c] == "missing_clinical"
            ]
            if missing_clinical:
                clinical_frac = len(missing_clinical) / total_discovered
                if clinical_frac > self._max_missing_clinical_frac:
                    raise RuntimeError(
                        f"{len(missing_clinical)}/{total_discovered} cases "
                        f"({clinical_frac:.1%}) have no clinical row for "
                        f"graph_features={requested_clinical}, exceeding "
                        f"max_missing_clinical_frac={self._max_missing_clinical_frac}. "
                        "Check the clinical data source, or pass a higher "
                        "max_missing_clinical_frac if this many missing records "
                        "is expected."
                    )
            breast_split_dropped_cases = [
                c
                for c in dropped
                if dropped_reasons[c]
                in ("breast_split_excluded", "breast_split_unknown_laterality")
            ]
            if breast_split_dropped_cases:
                breast_split_frac = len(breast_split_dropped_cases) / total_discovered
                if breast_split_frac > self._max_missing_breast_split_frac:
                    raise RuntimeError(
                        f"{len(breast_split_dropped_cases)}/{total_discovered} cases "
                        f"({breast_split_frac:.1%}) were dropped for "
                        "breast_split_mode='single' (excluded by the splitter or "
                        "missing a clinical row for laterality), exceeding "
                        f"max_missing_breast_split_frac={self._max_missing_breast_split_frac}. "
                        "Check gnn.build_single_breast_skeletons's manifest, or pass "
                        "a higher max_missing_breast_split_frac if this many "
                        "exclusions is expected."
                    )

        if not data_list:
            raise RuntimeError(
                "No graphs were built; check the centerline tree and labels file."
            )

        if self._graph_features:
            self._write_graph_feature_inputs(
                data_list, clinical_df, case_id_to_study_dir
            )

        self._write_feature_summary(data_list)
        self._write_graph_qc(data_list)
        self._write_cache_manifest(data_list)
        self._save_processed(data_list)

        logging.info(
            "GNN build complete: %d graphs built, %d dropped",
            len(data_list),
            len(dropped),
        )
        if self._profile:
            self._timings.log_summary()

    def _load_clinical_df(self) -> pd.DataFrame:
        """Load clinical metadata, preferring ``patient_info_dir`` over ``clinical_excel``.

        Mirrors ``clinical_features.get_clinical_features``'s preference order,
        without requiring the caller to build a full training ``config`` object.
        """
        if self._patient_info_dir is not None:
            return load_clinical_from_patient_info(self._patient_info_dir)
        return load_clinical_from_excel(self._clinical_excel)

    def _resolve_morphometry_paths(
        self, data_list: list[Data], case_id_to_study_dir: dict[str, Path]
    ) -> dict[str, Path]:
        """Read each surviving case's ``morphometry_path`` out of ``run_summary.json``.

        ``morphometry_path`` is present for 100% of the real MAMA-MIA cohort
        (see ``gnn/morphometry.py``'s module docstring) -- a missing summary,
        missing key, or nonexistent file is treated as a hard failure (fail
        loudly), not a third drop-threshold mechanism like
        ``max_missing_clinical_frac``, since this is not an observed
        real-world case and a build-time regression here should surface
        immediately rather than silently shrinking the cohort.
        """
        paths: dict[str, Path] = {}
        for data in data_list:
            case_id = str(data.case_id)
            summary_path = case_id_to_study_dir[case_id] / _RUN_SUMMARY_NAME
            summary = json.loads(summary_path.read_text())
            morphometry_path = summary.get("morphometry_path")
            if not morphometry_path:
                raise KeyError(
                    f"case={case_id}: run_summary.json missing 'morphometry_path' "
                    "-- expected present for every case (see gnn/morphometry.py)."
                )
            resolved = Path(morphometry_path)
            if not resolved.exists():
                raise FileNotFoundError(
                    f"case={case_id}: morphometry_path {resolved} does not exist"
                )
            paths[case_id] = resolved
        return paths

    def _requested_graph_feature_groups(self) -> tuple[list[str], list[str], list[str]]:
        """Split the requested ``graph_features`` into their three vocabularies.

        ``graph_features`` spans three independent sources -- clinical
        (``gnn.clinical``), graph-derived (``gnn.graph_derived``), and
        morphometry (``gnn.morphometry``). Membership resolves each name's
        source, and each group preserves the requested order. This is the
        single source of truth for the group split, used both when writing the
        raw-input sidecar and by ``graph_feature_groups`` for the per-fold
        transform in ``gnn.train``.
        """
        clinical = [f for f in self._graph_features if f in ALL_CLINICAL_COLUMNS]
        graph_derived = [f for f in self._graph_features if f in GRAPH_DERIVED_COLUMNS]
        morphometry = [f for f in self._graph_features if f in MORPHOMETRY_COLUMNS]
        return clinical, graph_derived, morphometry

    def graph_feature_groups(self) -> tuple[list[str], list[str], list[str]]:
        """Public: requested (clinical, graph_derived, morphometry) names, in order.

        The per-fold transform in ``gnn.train`` uses this to know which columns
        of :meth:`load_graph_feature_inputs` get a clinical impute+one-hot
        transformer, which get a morphometry imputer, and which pass through
        (graph-derived), and to rebuild the graph-feature vector in the same
        fixed source order the sidecar columns are laid out in.
        """
        return self._requested_graph_feature_groups()

    def _build_graph_feature_inputs(
        self,
        data_list: list[Data],
        clinical_df: pd.DataFrame | None,
        case_id_to_study_dir: dict[str, Path],
    ) -> pd.DataFrame:
        """Assemble the RAW (un-imputed, un-encoded) graph-level feature inputs.

        One row per case (indexed by ``case_id``), columns laid out in a fixed
        source order -- clinical, then graph-derived, then morphometry --
        regardless of how the caller interleaved names in ``gnn_graph_features``
        (each name's source is always recoverable by vocabulary membership).

        This is the Option-1 cross-validation-leakage fix: rather than fitting
        the clinical imputer/one-hot encoder and the morphometry imputer once
        over the whole cohort here and baking the transformed vector into each
        graph, we cache only the raw inputs and defer every cross-case fit to
        per-fold code in ``gnn.train`` (see module-level
        ``_GRAPH_FEATURE_INPUTS_NAME``). Clinical values are normalized
        (deterministic per case, no fit) and morphometry scalars are read raw
        with their ``NaN``s intact; graph-derived features are pure per-graph
        reads (never missing, no cross-case fit) and pass through as-is.

        Every case in ``data_list`` is already confirmed present in
        ``clinical_df`` (the clinical drop happens in ``process()`` before the
        graph build), so this is a pure read, not a further filter.
        """
        clinical_cols, graph_derived_cols, morphometry_cols = (
            self._requested_graph_feature_groups()
        )
        case_ids = [str(data.case_id) for data in data_list]
        index = pd.Index(case_ids, name="case_id")
        blocks: list[pd.DataFrame] = []
        if clinical_cols:
            blocks.append(
                normalize_clinical_frame(clinical_df, case_ids, clinical_cols)
            )
        if graph_derived_cols:
            graph_derived = build_graph_derived_feature_matrix(
                data_list, graph_derived_cols
            )
            blocks.append(
                pd.DataFrame(
                    graph_derived, columns=list(graph_derived_cols), index=index
                )
            )
        if morphometry_cols:
            morphometry_paths_by_case = self._resolve_morphometry_paths(
                data_list, case_id_to_study_dir
            )
            blocks.append(
                extract_morphometry_frame(
                    morphometry_paths_by_case, case_ids, morphometry_cols
                )
            )
        return pd.concat(blocks, axis=1)

    def _write_graph_feature_inputs(
        self,
        data_list: list[Data],
        clinical_df: pd.DataFrame | None,
        case_id_to_study_dir: dict[str, Path],
    ) -> None:
        """Persist (or, for ``no_cache``, hold in memory) the raw graph-feature inputs."""
        frame = self._build_graph_feature_inputs(
            data_list, clinical_df, case_id_to_study_dir
        )
        if self._no_cache:
            self._graph_feature_inputs = frame
            return
        frame.to_csv(Path(self.processed_dir) / _GRAPH_FEATURE_INPUTS_NAME)

    def load_graph_feature_inputs(self) -> pd.DataFrame | None:
        """Return the cached raw graph-level feature inputs (index=case_id), or None.

        ``None`` when no ``graph_features`` were requested. The per-fold
        transform in ``gnn.train`` fits the clinical/morphometry preprocessing
        on the training split of this frame and transforms the validation
        split -- so the imputer means/modes and one-hot vocabulary never see
        validation cases.

        Raises if ``graph_features`` were requested but the sidecar is absent
        -- e.g. a cache built before this leakage fix, which baked the
        whole-cohort-transformed vector into the graphs instead. Rebuild
        (``--force-rebuild``) or regenerate the sidecar
        (``gnn.build_dataset --regenerate-graph-feature-inputs``) rather than
        silently falling back to the old leaky features.
        """
        if not self._graph_features:
            return None
        if self._no_cache:
            if self._graph_feature_inputs is None:
                raise RuntimeError(
                    "no_cache=True but process() has not produced graph feature "
                    "inputs yet"
                )
            return self._graph_feature_inputs
        path = Path(self.processed_dir) / _GRAPH_FEATURE_INPUTS_NAME
        if not path.exists():
            raise RuntimeError(
                f"graph_features were requested but {path} is missing. This cache "
                "predates the per-fold graph-feature preprocessing fix (it baked "
                "whole-cohort-fit features into the graphs, leaking the validation "
                "distribution into training). Rebuild with --force-rebuild, or "
                "regenerate the raw-input sidecar without a full rebuild via "
                "`python -m gnn.build_dataset --regenerate-graph-feature-inputs ...`."
            )
        frame = pd.read_csv(path, index_col="case_id")
        frame.index = frame.index.astype(str)
        return frame

    def regenerate_graph_feature_inputs(self) -> Path:
        """Rebuild the raw graph-feature-input sidecar for an already-built cache.

        Lightweight migration for caches built before the cross-validation
        leakage fix: the raw inputs depend only on the surviving cases, the
        clinical source, and the morphometry JSONs -- none of the expensive
        raw-DCE graph construction -- so they are recomputed from the cached
        graphs without a full rebuild. Returns the sidecar path.
        """
        if not self._graph_features:
            raise ValueError(
                "dataset was built without graph_features; nothing to regenerate."
            )
        if self._no_cache:
            raise ValueError(
                "regenerate_graph_feature_inputs is for on-disk caches; a "
                "no_cache dataset holds its raw inputs in memory already."
            )
        data_list = [self[i] for i in range(len(self))]
        discovered = self._discover_cases()
        case_id_to_study_dir = {
            case_id: mask_path.parent for case_id, mask_path in discovered
        }
        clinical_cols, _, _ = self._requested_graph_feature_groups()
        clinical_df = self._load_clinical_df() if clinical_cols else None
        frame = self._build_graph_feature_inputs(
            data_list, clinical_df, case_id_to_study_dir
        )
        path = Path(self.processed_dir) / _GRAPH_FEATURE_INPUTS_NAME
        frame.to_csv(path)
        logging.info("Regenerated %s (%d cases)", path, len(frame))
        return path

    def _build_cases(
        self, tasks: list[tuple[str, Path, int]]
    ) -> Iterator[tuple[str, Data]]:
        """Build every ``(case_id, mask_path, label)`` task, in discovery order.

        Sequential when ``num_workers <= 1`` (the default -- safe on a login
        node); otherwise fans the per-case builds out across a process pool,
        submitting ``num_workers`` at a time, since each case is built
        independently from its own files. Every stage-timing sample collected
        in a worker is folded back into ``self._timings`` so profiling output
        is identical either way.

        Yields each ``(case_id, data)`` as soon as it is ready rather than
        collecting the full cohort into a list first -- ``process()`` saves
        ``<case_id>_graph.pt`` off this generator immediately, so an
        interrupted build (time limit, crash) keeps whatever it already
        finished instead of losing the entire run, and the log shows genuine
        per-case progress instead of going silent until the whole cohort is
        done.
        """
        total = len(tasks)
        if self._num_workers <= 1:
            for done, (case_id, mask_path, label) in enumerate(tasks, start=1):
                data, stage_samples = _build_case(
                    case_id,
                    mask_path,
                    label,
                    dce_root=self._dce_root,
                    node_features=self._node_features,
                    node_mode=self._node_mode,
                    edge_features=self._edge_features,
                    baseline_floor_frac=self._kinetic_baseline_floor_frac,
                )
                self._timings.merge(stage_samples)
                logging.info("GNN build: built %s (%d/%d)", case_id, done, total)
                yield case_id, data
            return

        logging.info(
            "GNN build: building %d case(s) across %d worker process(es)",
            total,
            self._num_workers,
        )
        done = 0
        with ProcessPoolExecutor(max_workers=self._num_workers) as executor:
            for batch_start in range(0, total, self._num_workers):
                batch = tasks[batch_start : batch_start + self._num_workers]
                futures = [
                    executor.submit(
                        _build_case,
                        case_id,
                        mask_path,
                        label,
                        dce_root=self._dce_root,
                        node_features=self._node_features,
                        node_mode=self._node_mode,
                        edge_features=self._edge_features,
                        baseline_floor_frac=self._kinetic_baseline_floor_frac,
                    )
                    for case_id, mask_path, label in batch
                ]
                for (case_id, _, _), future in zip(batch, futures, strict=True):
                    data, stage_samples = future.result()
                    self._timings.merge(stage_samples)
                    done += 1
                    logging.info("GNN build: built %s (%d/%d)", case_id, done, total)
                    yield case_id, data

    def _write_dropped_manifest(
        self,
        dropped: list[str],
        dropped_reasons: dict[str, str],
        num_discovered: int,
    ) -> None:
        """Persist dropped-case bookkeeping so cache hits can re-surface it."""
        if self._no_cache:
            return
        manifest = {
            "dropped_case_ids": dropped,
            "dropped_reasons": dropped_reasons,
            "num_discovered": num_discovered,
            "dropped_frac": len(dropped) / num_discovered if num_discovered else 0.0,
            "label_column": self._label_column,
            "labels_path": str(self._labels_path),
        }
        manifest_path = Path(self.processed_dir) / _DROPPED_MANIFEST_NAME
        manifest_path.write_text(json.dumps(manifest, indent=2))

    def _write_feature_summary(self, data_list: list[Data]) -> None:
        """Save per-feature histograms and a NaN/inf report for a fresh cache build.

        Runs once per cache build (never on a cache hit), so these always
        reflect the data that was actually collated -- not a stale snapshot
        from an earlier build.
        """
        if self._no_cache:
            return
        feature_matrix = torch.cat([data.x for data in data_list], dim=0).numpy()
        frame = pd.DataFrame(feature_matrix, columns=self._node_features)

        summary_dir = Path(self.processed_dir) / _FEATURE_SUMMARY_DIRNAME
        summary_dir.mkdir(exist_ok=True)

        na_report: dict[str, dict[str, float]] = {}
        for name in self._node_features:
            column = frame[name]
            na_report[name] = {
                "num_values": int(column.shape[0]),
                "num_nan": int(column.isna().sum()),
                "num_inf": int(np.isinf(column.to_numpy()).sum()),
                "min": float(column.min()),
                "max": float(column.max()),
                "mean": float(column.mean()),
            }

            fig, ax = plt.subplots(figsize=(6, 4))
            column.plot.hist(ax=ax, bins=_HIST_BINS)
            ax.set_xlabel(name)
            ax.set_title(f"{name} ({len(data_list)} graphs, {len(column)} nodes)")
            fig.tight_layout()
            fig.savefig(summary_dir / f"{name}_hist.png", dpi=150)
            plt.close(fig)

        (summary_dir / "feature_na_report.json").write_text(
            json.dumps(na_report, indent=2)
        )
        self._write_feature_summary_readme(summary_dir, na_report, len(data_list))

    def _write_feature_summary_readme(
        self,
        summary_dir: Path,
        na_report: dict[str, dict[str, float]],
        num_graphs: int,
    ) -> None:
        """Write a short auditing README alongside the histograms/NaN report."""
        lines = [
            "# GNN node-feature summary",
            "",
            f"Per-feature histograms and NaN/inf counts over the "
            f"`data.x` node features of the {num_graphs} graphs cached at "
            f"`{self.processed_dir}` (built from node_features="
            f"{self._node_features}). Regenerated automatically on every "
            "cache build (not on a cache hit) by "
            "`VanguardCenterlineDataset._write_feature_summary`.",
            "",
            "## NaN / inf report",
            "",
            "```json",
            json.dumps(na_report, indent=2),
            "```",
            "",
            "## Histograms",
            "",
        ]
        for name in self._node_features:
            lines.append(f"![{name} histogram]({name}_hist.png)")
        lines.append("")
        summary_dir.joinpath("README.md").write_text("\n".join(lines))

    def _write_graph_qc(self, data_list: list[Data]) -> None:
        """Write one ``graph_qc.csv`` row per graph for confound auditing.

        Graph size and per-feature ranges are possible confounders for this
        GNN (e.g. larger tumors correlating with pCR, or a dataset/site with
        systematically different feature ranges), not side details -- so
        every fresh build (never a cache hit) writes a row per graph with
        enough to plot ``num_nodes`` vs ``pcr``, ``num_nodes`` vs
        ``dataset``, and per-dataset / per-``pcr`` feature distributions
        directly from this one file.

        ``num_edges`` matches the ``data.num_edges`` convention already used
        in ``split_manifest.csv`` (``torch_geometric`` stores each undirected
        edge in both directions, so this is 2x the true undirected edge
        count). ``mean_degree`` (``num_edges / num_nodes``) is the correct
        average node degree under that same doubled convention.
        ``missing_feature_count`` counts every non-finite (NaN or inf) entry
        in ``data.x``; ``nan_feature_count`` is the NaN-only subset of that.
        Because time-to-enhancement no-arrival and junction no-bifurcation NaNs
        are sentinel-filled before ``data.x`` is finalized (see
        ``_sentinel_fill``), both are normally ``0``; the "no signal detected"
        counts are instead carried explicitly by ``tte_no_arrival_count`` and
        ``no_bifurcation_count`` (cells across ``data.x`` and, in junction
        mode, ``data.edge_attr``). A non-zero ``nan_feature_count`` here
        therefore signals an unexpected NaN, not a missing arrival/bifurcation.

        Also writes ``processed/graph_qc_plots/`` (via
        ``gnn.graph_qc_plots.write_build_time_plots``): num_nodes vs pcr,
        num_nodes vs dataset, and feature distributions by dataset/pcr. The
        fifth requested plot, prediction vs num_nodes, needs a trained
        model's predictions, which don't exist at build time -- it's written
        by ``gnn/train.py`` instead (into both the run's own output dir and
        back into this cache's ``graph_qc_plots/``, see
        ``run_gnn_pipeline``).
        """
        if self._no_cache:
            return
        rows: list[dict[str, object]] = []
        node_rows: list[dict[str, object]] = []
        for data in data_list:
            x = data.x.numpy()
            finite = np.isfinite(x)
            row: dict[str, object] = {
                "case_id": data.case_id,
                "dataset": data.dataset,
                "pcr": int(data.y.item()),
                "num_nodes": int(data.num_nodes),
                "num_edges": int(data.num_edges),
                "num_connected_components": int(data.num_connected_components),
                "mean_degree": float(data.num_edges) / float(data.num_nodes),
                "missing_feature_count": int((~finite).sum()),
                "nan_feature_count": int(np.isnan(x).sum()),
                "tte_no_arrival_count": int(getattr(data, "tte_no_arrival_count", 0)),
                "no_bifurcation_count": int(getattr(data, "no_bifurcation_count", 0)),
            }
            for i, name in enumerate(self._node_features):
                column = x[:, i]
                row[f"{name}_min"] = float(np.nanmin(column))
                row[f"{name}_max"] = float(np.nanmax(column))
                row[f"{name}_mean"] = float(np.nanmean(column))
                row[f"{name}_std"] = float(np.nanstd(column))
            rows.append(row)
            for node_idx in range(x.shape[0]):
                node_row = {
                    "case_id": data.case_id,
                    "dataset": data.dataset,
                    "pcr": int(data.y.item()),
                }
                node_row.update(
                    {name: x[node_idx, i] for i, name in enumerate(self._node_features)}
                )
                node_rows.append(node_row)

        qc = pd.DataFrame(rows)
        qc_path = Path(self.processed_dir) / _GRAPH_QC_NAME
        qc.to_csv(qc_path, index=False)

        node_df = pd.DataFrame(node_rows)
        plots_dir = Path(self.processed_dir) / GRAPH_QC_PLOTS_DIRNAME
        write_build_time_plots(qc, node_df, list(self._node_features), plots_dir)

    def _load_label_map(self) -> dict[str, int]:
        """Load labels into a ``case_id -> {0, 1}`` mapping."""
        frame = load_labels(self._labels_path, self._id_column, self._label_column)
        return {
            str(case_id): int(value)
            for case_id, value in zip(frame["case_id"], frame[self._label_column])
        }

    def _resolve_breast_split_paths(
        self, discovered: list[tuple[str, Path]]
    ) -> tuple[dict[str, Path], dict[str, str]]:
        """Map each discovered case to the skeleton path ``breast_split_mode="single"`` should use.

        Native unilateral cases (``bilateral=False``) keep their original
        exam-level skeleton unchanged -- there is nothing to split, per the
        harmonized-cohort plan. Bilateral cases are substituted with their
        precomputed single-breast skeleton
        (``gnn.build_single_breast_skeletons``) when one exists; a bilateral
        case with no precomputed skeleton (excluded by the splitter -- e.g.
        unknown tumor side or a tumor straddling both sides) or with no
        clinical row to determine laterality from at all is dropped, not
        silently kept with its original mixed-breast skeleton, which would
        defeat the point of the harmonized cohort.

        Returns:
            ``(resolved_paths, drop_reasons)`` -- ``resolved_paths`` maps
            every case that should still be built to the skeleton path to
            use; ``drop_reasons`` maps every other case to why it's excluded.
        """
        clinical_df = self._load_clinical_df()
        bilateral_by_case = dict(zip(clinical_df["case_id"], clinical_df["bilateral"]))

        resolved_paths: dict[str, Path] = {}
        drop_reasons: dict[str, str] = {}
        for case_id, mask_path in discovered:
            bilateral = bilateral_by_case.get(case_id)
            if bilateral is None:
                drop_reasons[case_id] = "breast_split_unknown_laterality"
                continue
            if not bilateral:
                resolved_paths[case_id] = mask_path
                continue
            dataset = case_id.split("_")[0]
            single_breast_path = single_breast_skeleton_path(
                self._breast_split_skeleton_root, dataset, case_id
            )
            if single_breast_path.exists():
                resolved_paths[case_id] = single_breast_path
            else:
                drop_reasons[case_id] = "breast_split_excluded"
        return resolved_paths, drop_reasons

    def _discover_cases(self) -> list[tuple[str, Path]]:
        """Find ``(case_id, mask_path)`` pairs under the centerline tree."""
        pairs: list[tuple[str, Path]] = []
        for mask_path in sorted(self._centerline_root.rglob(f"*{_CENTERLINE_SUFFIX}")):
            case_id = mask_path.name[: -len(_CENTERLINE_SUFFIX)]
            if self._cases is not None and case_id not in self._cases:
                continue
            pairs.append((case_id, mask_path))
        return pairs
