"""Does the arrival-time field have ANY spatial structure on the vessel tree?

The Stage 2 relational features came out at chance
(``experiments/uchicago_graph_relational``), with ``arrival_monotonic_frac`` sitting
at 0.490 +/- 0.011 -- a coin flip in every case. Two very different things produce
that number, and they imply opposite next steps:

1. **The root definition is wrong.** Arrival times are spatially organized, but not
   around the largest-radius node, so distance-from-root is the wrong coordinate.
   Fixable: pick a better root.
2. **The arrival field is spatially unstructured.** Adjacent voxels' arrival times
   are essentially independent, so no root, no feature, and no architecture can
   recover propagation -- the information is not in the acquisition.

This separates them without assuming either. For each case it measures how much
adjacent voxels' arrival times agree, against two references:

- **A shuffled null.** Arrival values permuted across nodes within the case,
  keeping the graph fixed. This is what zero spatial structure looks like.
- **A positive control: ``radius``.** Vessel calibre is necessarily smooth along a
  centreline, so it must show strong edge-wise agreement. If radius does and
  arrival does not, the graph reconstruction is sound and the arrival field
  specifically is unstructured -- which rules out "my skeleton/adjacency is broken"
  as the explanation.

Also reports ``arrival_monotonic_frac`` under the real (largest-radius) root
against a randomly chosen root: if the real root is no better than random, the
distance-from-root coordinate carries no information regardless of the field.

Interpretation hinges on the acquisition, which is why this matters: UChicago DCE
series are 5 fast precontrast frames, then a 29-107 s injection gap, then ~19 frames
at ~3.3 s. Bolus wash-in happens *inside the gap*, so by the first postcontrast
frame contrast has already reached everywhere, and "time to enhancement" is measured
during the plateau/washout rather than during arrival.

Reads the built voxel cache only -- no DCE reload, no rebuild.

Usage:
    python -m analysis.arrival_field_structure \
        --cache-dir /ess/.../gnn_cache_voxel_richfeat_floored \
        --out-dir experiments/uchicago_arrival_field
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch

from gnn.relational_features import MIN_COMPONENT_NODES, _skeleton_graph

_GRAPH_SUFFIX = "_graph.pt"
_RANDOM_PAIRS = 20000
# A correlation needs at least two pairs to be defined at all.
_MIN_CORR_POINTS = 2


def _edgewise_correlation(
    graph: nx.Graph, values: np.ndarray, in_use: np.ndarray
) -> float:
    """Pearson r between the two endpoint values of every usable skeleton edge.

    High r means the field varies smoothly along vessels; r near 0 means adjacent
    voxels are essentially independent. Edges with a non-finite value at either end
    are skipped (no detected arrival is a real condition, not a bug).
    """
    left, right = [], []
    for source, target in graph.edges():
        if not (in_use[source] and in_use[target]):
            continue
        a, b = values[source], values[target]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        left.append(a)
        right.append(b)
    if len(left) < _MIN_CORR_POINTS:
        return float("nan")
    left_arr, right_arr = np.asarray(left), np.asarray(right)
    if left_arr.std() == 0.0 or right_arr.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(left_arr, right_arr)[0, 1])


def _random_pair_correlation(
    values: np.ndarray, used: np.ndarray, rng: np.random.Generator
) -> float:
    """Pearson r between random *non-adjacent* node pairs -- the no-structure anchor."""
    finite = used[np.isfinite(values[used])]
    if finite.size < _MIN_CORR_POINTS:
        return float("nan")
    a = rng.choice(finite, size=_RANDOM_PAIRS)
    b = rng.choice(finite, size=_RANDOM_PAIRS)
    keep = a != b
    if keep.sum() < _MIN_CORR_POINTS:
        return float("nan")
    return float(np.corrcoef(values[a[keep]], values[b[keep]])[0, 1])


def _monotonic_fraction(
    graph: nx.Graph, distance: np.ndarray, arrival: np.ndarray, in_use: np.ndarray
) -> float:
    """Fraction of edges where the node farther from the root enhances later.

    Ties in arrival carry no ordering and are excluded, matching
    ``gnn.relational_features``.
    """
    agree = comparable = 0
    for source, target in graph.edges():
        if not (in_use[source] and in_use[target]):
            continue
        d_distance = distance[target] - distance[source]
        d_arrival = arrival[target] - arrival[source]
        if not np.isfinite(d_arrival) or d_distance == 0.0 or d_arrival == 0.0:
            continue
        comparable += 1
        agree += int(np.sign(d_distance) == np.sign(d_arrival))
    return float(agree / comparable) if comparable else float("nan")


def _distance_from_roots(
    graph: nx.Graph,
    components: list[np.ndarray],
    num_nodes: int,
    pick_root: Callable[[np.ndarray], int],
) -> np.ndarray:
    """Geodesic distance from each component's root, chosen by ``pick_root``."""
    distance = np.full(num_nodes, np.nan, dtype=float)
    for component in components:
        root = int(pick_root(component))
        for node, length in nx.single_source_dijkstra_path_length(
            graph, root, weight="weight"
        ).items():
            distance[node] = float(length)
    return distance


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for the arrival-field structure diagnostic."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    """Measure arrival-field spatial structure per case and summarize the cohort."""
    args = build_parser().parse_args()
    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted((args.cache_dir / "processed").glob(f"*{_GRAPH_SUFFIX}"))
    if not paths:
        raise FileNotFoundError(f"No {_GRAPH_SUFFIX} files under {args.cache_dir}")

    rows = []
    for path in paths:
        cached = torch.load(path)
        data = cached[0] if isinstance(cached, tuple | list) else cached
        arrival = data.time_to_enhancement_seconds.numpy().astype(float)
        radius = data.radius.numpy().astype(float)
        graph = _skeleton_graph(data)
        components = [
            np.asarray(sorted(component), dtype=int)
            for component in nx.connected_components(graph)
            if len(component) >= MIN_COMPONENT_NODES
        ]
        if not components:
            raise ValueError(f"case={data.case_id}: no usable component")
        used = np.concatenate(components)
        in_use = np.zeros(int(data.num_nodes), dtype=bool)
        in_use[used] = True

        shuffled = arrival.copy()
        shuffled[used] = rng.permutation(arrival[used])

        real_distance = _distance_from_roots(
            graph,
            components,
            int(data.num_nodes),
            lambda c, r=radius: c[int(np.argmax(r[c]))],
        )
        random_distance = _distance_from_roots(
            graph,
            components,
            int(data.num_nodes),
            lambda c: rng.choice(c),
        )

        rows.append(
            {
                "case_id": str(data.case_id),
                "pcr": int(data.y.item()),
                "edge_r_arrival": _edgewise_correlation(graph, arrival, in_use),
                "edge_r_arrival_shuffled": _edgewise_correlation(
                    graph, shuffled, in_use
                ),
                "randompair_r_arrival": _random_pair_correlation(arrival, used, rng),
                "edge_r_radius": _edgewise_correlation(graph, radius, in_use),
                "edge_r_radius_shuffled": _edgewise_correlation(
                    graph, radius[rng.permutation(len(radius))], in_use
                ),
                "monotonic_real_root": _monotonic_fraction(
                    graph, real_distance, arrival, in_use
                ),
                "monotonic_random_root": _monotonic_fraction(
                    graph, random_distance, arrival, in_use
                ),
            }
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(args.out_dir / "arrival_field_structure.csv", index=False)
    summary = (
        frame.drop(columns=["case_id", "pcr"])
        .describe()
        .loc[["mean", "std", "min", "50%", "max"]]
    )
    summary.to_csv(args.out_dir / "summary.csv")
    print(f"n_cases={len(frame)}")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
