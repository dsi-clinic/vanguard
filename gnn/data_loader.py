"""PyTorch-Geometric dataset that builds raw vessel graphs from centerlines.

The tabular / Deep Sets pipelines consume *summarized* vessel-graph features. The
GNN track instead needs the **raw graph** -- one node per skeleton voxel, edges
between 26-connected voxels -- delivered as :class:`torch_geometric.data.Data`
objects. This module walks a saved centerline output tree, rebuilds each case's
graph with the existing ``graph_extraction`` primitives, attaches node features
(peak-contrast time, local radius), and collates everything into an
:class:`~torch_geometric.data.InMemoryDataset`.

The heavy graph-building work is deliberately shared with the rest of the repo:
we reuse ``mask_to_edges_bitmask``, ``edges_to_segments``, ``segments_to_graph``
and ``obtain_radius_map`` so there is a single source of truth for how a skeleton
mask becomes a graph.
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import from_networkx

from config import DEFAULT_CONFIG
from graph_extraction.constants import NDIM_3D
from graph_extraction.core4d import load_time_series_from_files
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

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        """Time a code block and record its elapsed seconds under ``stage``."""
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            self._stages.setdefault(stage, []).append(elapsed)

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

    Args:
        root: The centerline ``studies`` tree containing per-case output
            directories with ``*_skeleton_4d_exam_mask.npy`` files.
        labels_path: CSV/JSON labels file passed to
            :func:`tabular.cohort.load_labels`.
        cache_dir: Where the collated ``processed/`` cache is written. Defaults
            to ``<root>/gnn_cache`` so the source tree can stay read-only when an
            explicit path is given.
        cases: Optional whitelist of case IDs to include.
        no_cache: Skip reading and writing the on-disk cache; always rebuild from
            source. Useful during development to avoid stale-cache surprises.
        node_mode: Node granularity. Only ``"voxel"`` is implemented; ``"segment"``
            raises :class:`NotImplementedError` as an explicit extension point.
        node_features: Node-feature names, in ``data.x`` column order. Supported:
            ``"peak_time"`` (normalized peak-contrast time), ``"radius"``, and
            ``"pcr_dummy"`` (the graph's ``pcr`` label broadcast onto every
            node -- a leakage-canary feature for pipeline sanity checks only;
            opt-in, never included unless named explicitly here).
        id_column: Case-ID column in the labels file.
        label_column: Binary label column in the labels file.
        max_missing_label_frac: Maximum fraction of discovered cases allowed to
            be dropped for lacking a ``label_column`` value. Every drop is
            logged regardless; exceeding this fraction raises ``RuntimeError``
            instead of building a silently-shrunken cohort. Default 0.1 (10%).
        profile: When true, accumulate and log per-stage timings.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        labels_path: str | Path,
        cache_dir: str | Path | None = None,
        cases: Sequence[str] | None = None,
        no_cache: bool = False,
        node_mode: str = "voxel",
        node_features: Sequence[str] = _DEFAULT_NODE_FEATURES,
        id_column: str = "case_id",
        label_column: str = "pcr",
        max_missing_label_frac: float = 0.1,
        profile: bool = False,
        transform: object = None,
        pre_transform: object = None,
    ) -> None:
        if labels_path is None:
            raise ValueError("labels_path is required; every graph must carry a label.")
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

        self._centerline_root = Path(root)
        self._labels_path = Path(labels_path)
        self._cases = set(cases) if cases is not None else None
        self._no_cache = no_cache
        self._data_list_cache: list[Data] | None = None
        self._node_mode = node_mode
        self._node_features = tuple(node_features)
        self._id_column = id_column
        self._label_column = label_column
        self._max_missing_label_frac = max_missing_label_frac
        self._profile = profile
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

        data_list: list[Data] = []
        dropped: list[str] = []

        for case_id, mask_path in discovered:
            label = labels.get(case_id)
            if label is None:
                dropped.append(case_id)
                continue
            data = self._build_one_case(case_id, mask_path, label)
            torch.save(data, Path(self.processed_dir) / f"{case_id}_graph.pt")
            data_list.append(data)

        self.dropped_case_ids = dropped
        self._write_dropped_manifest(dropped, len(discovered))
        if dropped:
            dropped_frac = len(dropped) / len(discovered)
            logging.warning(
                "GNN build: dropped %d/%d case(s) (%.1f%%) with no %r label in "
                "%s: %s",
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

        self._save_processed(data_list)

        logging.info(
            "GNN build complete: %d graphs built, %d dropped for missing label",
            len(data_list),
            len(dropped),
        )
        if self._profile:
            self._timings.log_summary()

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

    def _build_one_case(self, case_id: str, mask_path: Path, label: int) -> Data:
        """Build one :class:`Data` graph for ``case_id``."""
        study_dir = mask_path.parent

        with self._timings.measure("mask_load"):
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

        with self._timings.measure("graph_build"):
            segments = edges_to_segments(mask_to_edges_bitmask(skeleton))
            if segments.size == 0:
                raise ValueError(f"Skeleton for {case_id} has zero segments")
            graph = segments_to_graph(segments)
        if graph.number_of_nodes() == 0:
            raise ValueError(f"Graph for {case_id} has zero nodes")

        radius_map = obtain_radius_map(support, graph)

        signal_4d = self._load_time_series(case_id, study_dir)
        num_timepoints = int(signal_4d.shape[0])

        with self._timings.measure("peak_time"):
            self._attach_node_features(graph, radius_map, signal_4d, label)

        with self._timings.measure("from_networkx"):
            data = from_networkx(graph)

        return self._finalize_data(data, case_id, study_dir, label, num_timepoints)

    def _load_time_series(self, case_id: str, study_dir: Path) -> np.ndarray:
        """Load the stacked 4D signal for a case."""
        paths = self._resolve_timepoint_paths(case_id, study_dir)
        with self._timings.measure("timeseries_load"):
            return load_time_series_from_files(paths)

    def _resolve_timepoint_paths(self, case_id: str, study_dir: Path) -> list[Path]:
        """Return timepoint paths from ``run_summary.json["study_files"]``.

        Raises ``FileNotFoundError`` if the summary is missing, ``KeyError`` if
        ``study_files`` is absent or empty, and propagates ``json.JSONDecodeError``
        if the file is malformed.
        """
        summary_path = study_dir / _RUN_SUMMARY_NAME
        if not summary_path.exists():
            raise FileNotFoundError(
                f"case={case_id}: {summary_path} not found; "
                "run_summary.json is required"
            )
        summary = json.loads(summary_path.read_text())
        study_files = summary.get("study_files")
        if not study_files:
            raise KeyError(
                f"case={case_id}: run_summary.json missing or empty 'study_files' key"
            )
        return [Path(p) for p in study_files]

    def _attach_node_features(
        self,
        graph: nx.Graph,
        radius_map: dict[tuple[int, int, int], float],
        signal_4d: np.ndarray,
        label: int,
    ) -> None:
        """Set ``radius``, ``peak_time`` and ``peak_time_norm`` on every node.

        ``pcr_dummy`` (the label broadcast onto every node) is only computed
        and attached when it is present in ``self._node_features`` -- it is a
        leakage canary for pipeline sanity checks, not a default feature, so
        it must stay opt-in rather than something every graph carries.
        """
        num_timepoints = int(signal_4d.shape[0])
        denom = max(num_timepoints - 1, _SINGLE_TIMEPOINT)
        baseline = signal_4d[0]
        include_pcr_dummy = "pcr_dummy" in self._node_features
        for node in graph.nodes():
            x, y, z = int(node[0]), int(node[1]), int(node[2])
            enhancement = signal_4d[:, z, y, x] - baseline[z, y, x]
            peak_idx = int(np.argmax(enhancement))
            attrs = graph.nodes[node]
            attrs["radius"] = float(radius_map[node])
            attrs["peak_time"] = peak_idx
            attrs["peak_time_norm"] = float(peak_idx) / float(denom)
            if include_pcr_dummy:
                attrs["pcr_dummy"] = float(label)

    def _finalize_data(
        self,
        data: Data,
        case_id: str,
        study_dir: Path,
        label: int,
        num_timepoints: int,
    ) -> Data:
        """Assemble ``data.x``, the label, and provenance metadata."""
        columns = [data[_FEATURE_ATTR[name]] for name in self._node_features]
        data.x = torch.stack([column.float() for column in columns], dim=1)
        data.y = torch.tensor([int(label)], dtype=torch.long)
        data.case_id = case_id
        data.num_timepoints = num_timepoints

        rel_parts = study_dir.relative_to(self._centerline_root).parts
        dataset = rel_parts[0] if rel_parts else "unknown"
        data.dataset = dataset
        data.site = dataset
        return data
