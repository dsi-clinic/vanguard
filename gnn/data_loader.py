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

from config import DEFAULT_CONFIG
from gnn.graph_qc_plots import GRAPH_QC_PLOTS_DIRNAME, write_build_time_plots
from gnn.raw_dce import (
    discover_raw_dce_paths,
    load_raw_dce_series,
    load_raw_dce_times,
)
from graph_extraction.constants import NDIM_3D
from graph_extraction.feature_stats import (
    _arrival_index_from_enhancement,
    mask_to_edges_bitmask,
)
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
_HIST_BINS = 50

# The only node-feature source implemented today (see module docstring): raw
# DCE-MRI signal, not the vessel-segmentation probability maps. Recorded in
# every cache_manifest.json so a manifest from before this migration -- or a
# hypothetical future vessel_segmentation-sourced build -- is never silently
# treated as compatible with the current code.
_FEATURE_SOURCE = "raw_dce_physical_time_sidecar_v2"

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
    "auc_positive": "auc_positive",
    "radius": "radius",
    "pcr_dummy": "pcr_dummy",
}
_DEFAULT_NODE_FEATURES: tuple[str, ...] = ("peak_time", "radius")


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


def _load_study_timepoints(case_id: str, study_dir: Path) -> list[int]:
    """Return the timepoint indices from ``run_summary.json["study_timepoints"]``.

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
    return [int(t) for t in study_timepoints]


def _time_axis_from_study_timepoints(
    study_timepoints: list[int],
    physical_times_seconds: np.ndarray | None = None,
) -> np.ndarray:
    """Build a strictly increasing time axis, mirroring ``features.kinematic``.

    Physical acquisition seconds take precedence when the Vanguard sidecar is
    present. Legacy cohorts use their recorded timepoint indices. Either input
    falls back to ``0..T-1`` if it is not finite and strictly increasing.
    """
    time_axis = (
        np.asarray(physical_times_seconds, dtype=float)
        if physical_times_seconds is not None
        else np.asarray(study_timepoints, dtype=float)
    )
    if not np.all(np.isfinite(time_axis)) or np.any(np.diff(time_axis) <= 0.0):
        return np.arange(len(study_timepoints), dtype=float)
    return time_axis


def _node_kinetic_features(
    curve: np.ndarray, time_axis: np.ndarray
) -> dict[str, object]:
    """Derive enhancement-curve features for one node's raw DCE signal.

    Mirrors the per-segment convention in
    ``features.kinematic.compute_tumor_kinematic_feature_payload``: baseline is
    the timepoint-0 value (no per-timepoint normalization, which would destroy
    the kinetic meaning of the curve), arrival is estimated with
    ``graph_extraction.feature_stats._arrival_index_from_enhancement``, and
    washin/AUC use the same formulas.

    ``tte_idx`` is ``None`` when the node shows no meaningful enhancement
    (peak <= 0) -- a real "no signal" voxel, not a bug -- and the caller is
    responsible for choosing a sentinel for the tensor-facing feature.
    """
    baseline = float(curve[0])
    enh = np.asarray(curve, dtype=float) - baseline
    peak_idx = int(np.argmax(enh))
    peak_enhancement = float(enh[peak_idx])
    tte_idx = _arrival_index_from_enhancement(enh)
    start_idx = 0 if tte_idx is None else int(tte_idx)
    washin_den = float(time_axis[peak_idx] - time_axis[start_idx])
    washin_slope = (
        float((enh[peak_idx] - enh[start_idx]) / washin_den)
        if washin_den > 0.0
        else 0.0
    )
    auc_positive = float(np.trapz(np.maximum(enh, 0.0), x=time_axis))
    return {
        "peak_idx": peak_idx,
        "peak_enhancement": peak_enhancement,
        "tte_idx": tte_idx,
        "washin_slope": washin_slope,
        "auc_positive": auc_positive,
    }


def _attach_node_features(
    graph: nx.Graph,
    radius_map: dict[tuple[int, int, int], float],
    dce_4d: np.ndarray,
    time_axis: np.ndarray,
    label: int,
    node_features: tuple[str, ...],
) -> None:
    """Set ``radius`` and the DCE-derived kinetic features on every node.

    ``time_to_enhancement_norm`` is ``NaN`` for nodes with no detected arrival
    (no meaningful enhancement) -- this is caught and reported per-feature by
    ``_write_feature_summary``'s NaN/inf audit rather than silently defaulted.

    ``pcr_dummy`` (the label broadcast onto every node) is only computed and
    attached when it is present in ``node_features`` -- it is a leakage
    canary for pipeline sanity checks, not a default feature, so it must stay
    opt-in rather than something every graph carries.
    """
    num_timepoints = int(dce_4d.shape[0])
    denom = max(num_timepoints - 1, _SINGLE_TIMEPOINT)
    include_pcr_dummy = "pcr_dummy" in node_features
    for node in graph.nodes():
        x, y, z = int(node[0]), int(node[1]), int(node[2])
        curve = dce_4d[:, z, y, x]
        kinetic = _node_kinetic_features(curve, time_axis)
        tte_idx = kinetic["tte_idx"]

        attrs = graph.nodes[node]
        attrs["radius"] = float(radius_map[node])
        attrs["peak_time"] = int(kinetic["peak_idx"])
        attrs["peak_time_norm"] = float(kinetic["peak_idx"]) / float(denom)
        attrs["peak_enhancement"] = float(kinetic["peak_enhancement"])
        attrs["time_to_enhancement"] = -1 if tte_idx is None else int(tte_idx)
        attrs["time_to_enhancement_norm"] = (
            float("nan") if tte_idx is None else float(tte_idx) / float(denom)
        )
        attrs["washin_slope"] = float(kinetic["washin_slope"])
        attrs["auc_positive"] = float(kinetic["auc_positive"])
        if include_pcr_dummy:
            attrs["pcr_dummy"] = float(label)


def _finalize_data(
    data: Data,
    case_id: str,
    study_dir: Path,
    label: int,
    num_timepoints: int,
    centerline_root: Path,
    node_features: tuple[str, ...],
    num_connected_components: int,
) -> Data:
    """Assemble ``data.x``, the label, and provenance metadata."""
    columns = [data[_FEATURE_ATTR[name]] for name in node_features]
    data.x = torch.stack([column.float() for column in columns], dim=1)
    data.y = torch.tensor([int(label)], dtype=torch.long)
    data.case_id = case_id
    data.num_timepoints = num_timepoints
    data.num_connected_components = num_connected_components

    rel_parts = study_dir.relative_to(centerline_root).parts
    dataset = rel_parts[0] if rel_parts else "unknown"
    data.dataset = dataset
    data.site = dataset
    return data


def _build_case(
    case_id: str,
    mask_path: Path,
    label: int,
    *,
    centerline_root: Path,
    dce_root: Path,
    node_features: tuple[str, ...],
) -> tuple[Data, dict[str, list[float]]]:
    """Build one labeled :class:`Data` graph for ``case_id``.

    Standalone (no dataset instance state) so it can run, unmodified, inside a
    worker process when the build is parallelized across cases -- each case's
    graph depends only on its own files, so there is no cross-case state to
    share.
    """
    stage_samples: dict[str, list[float]] = {}
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
        graph = segments_to_graph(segments)
    if graph.number_of_nodes() == 0:
        raise ValueError(f"Graph for {case_id} has zero nodes")

    # Must be counted on the nx.Graph itself -- from_networkx() below discards
    # it, and edge_index on the resulting Data is directed-both-ways, which is
    # not a valid input to nx.connected_components without reconstruction.
    num_connected_components = nx.number_connected_components(graph)

    radius_map = obtain_radius_map(support, graph)

    study_timepoints = _load_study_timepoints(case_id, study_dir)
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
        study_timepoints, physical_times_seconds
    )

    with _stage_timer(stage_samples, "peak_time"):
        _attach_node_features(
            graph, radius_map, dce_4d, time_axis, label, node_features
        )

    with _stage_timer(stage_samples, "from_networkx"):
        data = from_networkx(graph)

    data = _finalize_data(
        data,
        case_id,
        study_dir,
        label,
        num_timepoints,
        centerline_root,
        node_features,
        num_connected_components,
    )
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
        node_mode: Node granularity. Only ``"voxel"`` is implemented; ``"segment"``
            raises :class:`NotImplementedError` as an explicit extension point.
        node_features: Node-feature names, in ``data.x`` column order. Supported:
            ``"peak_time"`` (normalized time-to-peak enhancement),
            ``"peak_enhancement"``, ``"time_to_enhancement"`` (normalized arrival
            time; ``NaN`` for nodes with no detected enhancement), ``"washin_slope"``,
            ``"auc_positive"``, ``"radius"``, and ``"pcr_dummy"`` (the graph's ``pcr``
            label broadcast onto every node -- a leakage-canary feature for pipeline
            sanity checks only; opt-in, never included unless named explicitly here).
            The kinetic features are all sampled from the raw DCE enhancement curve
            (``curve = dce_4d[:, z, y, x]``, ``enhancement = curve - curve[0]``), using
            the same conventions as ``features/kinematic.py`` -- see
            ``_node_kinetic_features``.
        id_column: Case-ID column in the labels file.
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
        node_mode: str = "voxel",
        node_features: Sequence[str] = _DEFAULT_NODE_FEATURES,
        id_column: str = "case_id",
        label_column: str = "pcr",
        max_missing_label_frac: float = 0.1,
        profile: bool = False,
        num_workers: int = 1,
        allow_manifest_mismatch: bool = False,
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
        if node_mode != "voxel":
            raise NotImplementedError(
                f"node_mode={node_mode!r} is not implemented yet; use 'voxel'."
            )
        unknown = [f for f in node_features if f not in _FEATURE_ATTR]
        if unknown:
            raise ValueError(
                f"Unknown node_features {unknown}; supported: {sorted(_FEATURE_ATTR)}"
            )
        if not 0.0 <= max_missing_label_frac <= 1.0:
            raise ValueError(
                f"max_missing_label_frac must be in [0, 1], got {max_missing_label_frac}"
            )
        if num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {num_workers}")

        self._centerline_root = Path(root)
        self._labels_path = Path(labels_path)
        self._dce_root = Path(dce_root)
        self._cases = set(cases) if cases is not None else None
        self._no_cache = no_cache
        self._data_list_cache: list[Data] | None = None
        self._node_mode = node_mode
        self._node_features = tuple(node_features)
        self._id_column = id_column
        self._label_column = label_column
        self._max_missing_label_frac = max_missing_label_frac
        self._profile = profile
        self._num_workers = num_workers
        self._allow_manifest_mismatch = allow_manifest_mismatch
        self._timings = _StageTimings()
        self.dropped_case_ids: list[str] = []

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
            self._reload_dropped_manifest()

    def _reload_dropped_manifest(self) -> None:
        """Restore and re-log dropped-case bookkeeping on a cache hit.

        ``process()`` only runs once per cache; every later load of the same
        cache still goes through here, so this is what keeps missing-label
        drops visible instead of only being logged the one time the cache was
        built.
        """
        manifest_path = Path(self.processed_dir) / _DROPPED_MANIFEST_NAME
        if not manifest_path.exists():
            return
        manifest = json.loads(manifest_path.read_text())
        self.dropped_case_ids = manifest["dropped_case_ids"]
        if self.dropped_case_ids:
            logging.warning(
                "GNN dataset (cached): %d/%d case(s) (%.1f%%) were dropped for "
                "missing %r label in %s: %s",
                len(self.dropped_case_ids),
                manifest["num_discovered"],
                manifest["dropped_frac"] * 100,
                manifest["label_column"],
                manifest["labels_path"],
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
        return {
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

    def _check_cache_manifest(self) -> None:
        """Fail loudly if an on-disk cache was built under different settings.

        Without this, a stale cache (different centerline root, labels,
        node features, ...) could be silently reused just because
        ``processed/data.pt`` happens to exist at ``cache_dir`` -- see the
        code-review item this implements. Raises if the manifest is missing
        (a cache built before this check existed) or if any recorded setting
        differs from what's currently requested, unless
        ``allow_manifest_mismatch=True`` was passed.
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
        mismatched = {
            key: {"cached": manifest.get(key), "requested": value}
            for key, value in requested.items()
            if manifest.get(key) != value
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
        logging.info(
            "GNN build: %d candidate case(s) under %s",
            len(discovered),
            self._centerline_root,
        )

        dropped: list[str] = []
        tasks: list[tuple[str, Path, int]] = []
        for case_id, mask_path in discovered:
            label = labels.get(case_id)
            if label is None:
                dropped.append(case_id)
                continue
            tasks.append((case_id, mask_path, label))

        data_list: list[Data] = []
        for case_id, data in self._build_cases(tasks):
            torch.save(data, Path(self.processed_dir) / f"{case_id}_graph.pt")
            data_list.append(data)

        self.dropped_case_ids = dropped
        self._write_dropped_manifest(dropped, len(discovered))
        if dropped:
            dropped_frac = len(dropped) / len(discovered)
            logging.warning(
                "GNN build: dropped %d/%d case(s) (%.1f%%) with no %r label in %s: %s",
                len(dropped),
                len(discovered),
                dropped_frac * 100,
                self._label_column,
                self._labels_path,
                dropped,
            )
            if dropped_frac > self._max_missing_label_frac:
                raise RuntimeError(
                    f"{len(dropped)}/{len(discovered)} cases ({dropped_frac:.1%}) "
                    f"are missing a {self._label_column!r} label in "
                    f"{self._labels_path}, exceeding max_missing_label_frac="
                    f"{self._max_missing_label_frac}. Fix the labels file, or "
                    "pass a higher max_missing_label_frac if this many missing "
                    "labels is expected."
                )

        if not data_list:
            raise RuntimeError(
                "No graphs were built; check the centerline tree and labels file."
            )

        self._write_feature_summary(data_list)
        self._write_graph_qc(data_list)
        self._write_cache_manifest(data_list)
        self._save_processed(data_list)

        logging.info(
            "GNN build complete: %d graphs built, %d dropped for missing label",
            len(data_list),
            len(dropped),
        )
        if self._profile:
            self._timings.log_summary()

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
                    centerline_root=self._centerline_root,
                    dce_root=self._dce_root,
                    node_features=self._node_features,
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
                        centerline_root=self._centerline_root,
                        dce_root=self._dce_root,
                        node_features=self._node_features,
                    )
                    for case_id, mask_path, label in batch
                ]
                for (case_id, _, _), future in zip(batch, futures, strict=True):
                    data, stage_samples = future.result()
                    self._timings.merge(stage_samples)
                    done += 1
                    logging.info("GNN build: built %s (%d/%d)", case_id, done, total)
                    yield case_id, data

    def _write_dropped_manifest(self, dropped: list[str], num_discovered: int) -> None:
        """Persist dropped-case bookkeeping so cache hits can re-surface it."""
        if self._no_cache:
            return
        manifest = {
            "dropped_case_ids": dropped,
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
        in ``data.x``; ``nan_feature_count`` is the NaN-only subset of that,
        distinguishing "no signal detected" (NaN, e.g. ``time_to_enhancement``
        with no arrival) from a genuine inf.

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

    def _discover_cases(self) -> list[tuple[str, Path]]:
        """Find ``(case_id, mask_path)`` pairs under the centerline tree."""
        pairs: list[tuple[str, Path]] = []
        for mask_path in sorted(self._centerline_root.rglob(f"*{_CENTERLINE_SUFFIX}")):
            case_id = mask_path.name[: -len(_CENTERLINE_SUFFIX)]
            if self._cases is not None and case_id not in self._cases:
                continue
            pairs.append((case_id, mask_path))
        return pairs
