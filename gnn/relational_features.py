"""Graph-*relational* per-case features: how kinetics vary ACROSS the vessel tree.

Every node feature the GNN currently trains on is node-intrinsic -- a property of
one voxel computed in isolation (``gnn.kinetics``). Nothing is a difference between
connected nodes. That is why graph structure has not paid off: if the target
depends only on the distribution of a node function, pooling that function is
sufficient and the adjacency is irrelevant. Topology earns its keep only when the
target depends on *how* a node function varies across edges.

This module computes that missing quantity. The physiological target is contrast
bolus propagation: contrast enters at the inlet and travels distally, so arrival
time is a function on the tree that increases with distance from the root. Its
rate is transit velocity, its scatter is how orderly propagation is, and its
asymmetry between sibling branches is differential perfusion. Tumor-associated
angiogenesis is expected to perturb exactly these.

Two properties make these features worth the trouble at N=179:

- **Not computable from per-case means.** Each one is a joint property of
  (kinetics, tree position), so a tabular model on these scalars is a direct test
  of whether the graph carries signal -- with a model too simple to be accused of
  winning on capacity.
- **Protocol-invariant.** They are expressed in differences of physical seconds
  over physical millimetres. The per-case duration normalization that makes
  ``peak_time``/``time_to_enhancement`` track acquisition protocol
  (``experiments/uchicago_protocol_confound``) cancels out.

Everything is read from an already-built voxel-mode cache: ``pos`` (voxel
coordinates), ``edge_index`` (skeleton adjacency),
``time_to_enhancement_seconds`` / ``peak_time_seconds`` (physical seconds, set by
``gnn.data_loader._attach_node_features``), and ``radius``. No DCE reload and no
rebuild. UChicago v5 volumes are 1x1x1 mm isotropic, so voxel-coordinate distance
*is* millimetres.

**Root definition.** Each connected component is rooted at its largest-radius
node: proximal vessels are thickest, and unlike "earliest arrival" this is a
geometry-only rule, so it does not make the arrival-based features circular.

**Documented data handling** (see ``AUDITING_RESULTS.md``):

- ``time_to_enhancement_seconds`` is NaN for a voxel with no detected arrival
  (non-enhancing tissue -- a real case, not a bug). Such nodes are excluded from
  the arrival-based fits rather than imputed, and the retained fraction is
  reported as ``frac_nodes_with_arrival`` so the exclusion stays audited.
- Components smaller than ``MIN_COMPONENT_NODES`` are skipped: a handful of voxels
  has no meaningful transit geometry, and these graphs carry 11-24 components of
  which most are small fragments. The number kept is reported as
  ``n_components_used``.
- Slopes are estimated **within component then pooled** (component-demeaned, i.e.
  a fixed-effects fit), so a component whose root happens to enhance late cannot
  masquerade as a shallow transit slope.

A case that yields no usable component at all raises rather than returning zeros.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
from torch_geometric.data import Data

# Per-case relational feature vocabulary, in the order ``case_relational_features``
# emits them.
RELATIONAL_FEATURE_COLUMNS: tuple[str, ...] = (
    "transit_slope_s_per_mm",
    "transit_r2",
    "peak_transit_slope_s_per_mm",
    "peak_transit_r2",
    "arrival_monotonic_frac",
    "sibling_arrival_asymmetry_s",
    "radius_taper_per_mm",
)
# Bookkeeping emitted alongside the features so every exclusion above is visible
# in the output CSV rather than implicit.
RELATIONAL_QC_COLUMNS: tuple[str, ...] = (
    "arrival_tie_frac",
    "frac_peak_censored",
    "n_components_used",
    "n_nodes_used",
    "frac_nodes_with_arrival",
    "n_bifurcations_used",
)

# A component with fewer nodes than this has no meaningful transit geometry.
MIN_COMPONENT_NODES = 20
# A slope needs at least this many points, and a sibling comparison at least this
# many downstream branches, to be defined at all.
_MIN_FIT_POINTS = 10
_MIN_SIBLINGS = 2
# Degree at which a node is treated as a branch point.
_BIFURCATION_DEGREE = 3


def _skeleton_graph(data: Data) -> nx.Graph:
    """Undirected skeleton graph with Euclidean (millimetre) edge weights.

    ``edge_index`` stores each undirected skeleton edge in both directions;
    ``nx.Graph`` collapses the duplicates. Edge weight is the physical distance
    between the two voxels (1, sqrt(2) or sqrt(3) mm under 26-connectivity), so
    geodesic distance along a vessel is a real path length rather than a hop count.
    """
    pos = data.pos.numpy().astype(float)
    edge_index = data.edge_index.numpy()
    graph = nx.Graph()
    graph.add_nodes_from(range(int(data.num_nodes)))
    for source, target in zip(edge_index[0], edge_index[1], strict=True):
        if source == target:
            continue
        graph.add_edge(
            int(source),
            int(target),
            weight=float(np.linalg.norm(pos[source] - pos[target])),
        )
    return graph


def _demeaned_slope(
    x: np.ndarray, y: np.ndarray, groups: list[np.ndarray]
) -> tuple[float, float]:
    """Within-group slope of ``y`` on ``x``, pooled across groups, plus its R^2.

    Each group (connected component) is centered on its own means before pooling,
    which removes between-component offsets in both variables -- the fixed-effects
    estimator. Returns ``(nan, nan)`` when fewer than ``_MIN_FIT_POINTS`` usable
    points survive, so a case with no estimable slope is visibly missing rather
    than silently zero.
    """
    xs, ys = [], []
    for index in groups:
        if index.size < _MIN_FIT_POINTS:
            continue
        xi, yi = x[index], y[index]
        keep = np.isfinite(xi) & np.isfinite(yi)
        if keep.sum() < _MIN_FIT_POINTS:
            continue
        xi, yi = xi[keep], yi[keep]
        xs.append(xi - xi.mean())
        ys.append(yi - yi.mean())
    if not xs:
        return float("nan"), float("nan")
    x_all = np.concatenate(xs)
    y_all = np.concatenate(ys)
    denominator = float(np.sum(x_all**2))
    if x_all.size < _MIN_FIT_POINTS or denominator <= 0.0:
        return float("nan"), float("nan")
    slope = float(np.sum(x_all * y_all) / denominator)
    total = float(np.sum(y_all**2))
    if total <= 0.0:
        return slope, float("nan")
    residual = float(np.sum((y_all - slope * x_all) ** 2))
    return slope, float(1.0 - residual / total)


def case_relational_features(data: Data) -> dict[str, float]:
    """All relational features + QC bookkeeping for one built voxel-mode graph.

    Raises if ``data`` is not a voxel-mode graph carrying the physical-seconds
    kinetic attributes, or if no component is large enough to measure -- both are
    build/config errors rather than expected data conditions.
    """
    for attribute in (
        "pos",
        "time_to_enhancement_seconds",
        "peak_time_seconds",
        "radius",
    ):
        if not hasattr(data, attribute):
            raise AttributeError(
                f"Graph is missing {attribute!r}. Relational features need a "
                "voxel-mode cache built with the physical-seconds kinetic "
                "attributes (gnn.data_loader._attach_node_features)."
            )
    arrival = data.time_to_enhancement_seconds.numpy().astype(float)
    peak = data.peak_time_seconds.numpy().astype(float)
    radius = data.radius.numpy().astype(float)

    graph = _skeleton_graph(data)
    components = [
        np.asarray(sorted(component), dtype=int)
        for component in nx.connected_components(graph)
        if len(component) >= MIN_COMPONENT_NODES
    ]
    if not components:
        raise ValueError(
            f"case={getattr(data, 'case_id', '?')}: no connected component with "
            f">= {MIN_COMPONENT_NODES} nodes; cannot measure transit geometry."
        )

    # Distance from each component's own root (its thickest node).
    distance = np.full(int(data.num_nodes), np.nan, dtype=float)
    for component in components:
        root = int(component[int(np.argmax(radius[component]))])
        lengths = nx.single_source_dijkstra_path_length(graph, root, weight="weight")
        for node, length in lengths.items():
            distance[node] = float(length)

    transit_slope, transit_r2 = _demeaned_slope(distance, arrival, components)
    peak_slope, peak_r2 = _demeaned_slope(distance, peak, components)
    taper_slope, _ = _demeaned_slope(distance, radius, components)

    used = np.concatenate(components)
    in_use = np.zeros(int(data.num_nodes), dtype=bool)
    in_use[used] = True

    # Arrival monotonicity: along each skeleton edge, does the node farther from
    # the root enhance later? Only edges whose endpoints are both in a usable
    # component, both have a detected arrival, and sit at different distances can
    # answer that.
    #
    # Adjacent voxels very often land in the SAME frame, because the DCE series is
    # sampled every ~3-7 s while bolus transit across a vessel is far faster. Those
    # ties carry no ordering information, so they are excluded from the fraction
    # and reported separately as ``arrival_tie_frac`` -- which is itself the honest
    # measure of how much arrival ordering this acquisition can resolve at all.
    agree = 0
    comparable = 0
    tied = 0
    for source, target in graph.edges():
        if not (in_use[source] and in_use[target]):
            continue
        d_distance = distance[target] - distance[source]
        d_arrival = arrival[target] - arrival[source]
        if not np.isfinite(d_arrival) or d_distance == 0.0:
            continue
        if d_arrival == 0.0:
            tied += 1
            continue
        comparable += 1
        agree += int(np.sign(d_distance) == np.sign(d_arrival))
    monotonic_frac = float(agree / comparable) if comparable else float("nan")
    total_edges = comparable + tied
    tie_frac = float(tied / total_edges) if total_edges else float("nan")

    # Right-censoring check. A voxel whose enhancement is still rising when
    # acquisition stops reports its peak at the final frame, so peak_time (and the
    # washout slope measured from it) describe when the scan ended rather than the
    # physiology. Reported per case because the affected fraction is large and
    # varies with protocol.
    peak_index = data.peak_time.numpy().astype(int)
    frac_peak_censored = float(
        np.mean(peak_index[used] == int(data.num_timepoints) - 1)
    )

    # Sibling asymmetry: at a branch point, how differently do its downstream
    # branches enhance? Measured on the immediate downstream neighbours, so this is
    # a local (one-voxel) estimate; a segment-level version is the natural
    # refinement once this feature set proves useful.
    spreads = []
    for node in used:
        if graph.degree(int(node)) < _BIFURCATION_DEGREE:
            continue
        downstream = [
            arrival[neighbor]
            for neighbor in graph.neighbors(int(node))
            if in_use[neighbor]
            and distance[neighbor] > distance[node]
            and np.isfinite(arrival[neighbor])
        ]
        if len(downstream) < _MIN_SIBLINGS:
            continue
        spreads.append(float(max(downstream) - min(downstream)))
    sibling_asymmetry = float(np.mean(spreads)) if spreads else float("nan")

    return {
        "transit_slope_s_per_mm": transit_slope,
        "transit_r2": transit_r2,
        "peak_transit_slope_s_per_mm": peak_slope,
        "peak_transit_r2": peak_r2,
        "arrival_monotonic_frac": monotonic_frac,
        "sibling_arrival_asymmetry_s": sibling_asymmetry,
        "radius_taper_per_mm": taper_slope,
        "arrival_tie_frac": tie_frac,
        "frac_peak_censored": frac_peak_censored,
        "n_components_used": float(len(components)),
        "n_nodes_used": float(used.size),
        "frac_nodes_with_arrival": float(np.isfinite(arrival[used]).mean()),
        "n_bifurcations_used": float(len(spreads)),
    }
