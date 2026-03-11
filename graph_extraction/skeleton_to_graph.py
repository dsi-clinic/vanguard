"""Utilities to convert skeleton volumes into graph representations."""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from scipy.ndimage import distance_transform_edt

Point3D = tuple[int, int, int]
Segment = tuple[Point3D, Point3D]
PathList = list[Point3D]

_OFFSETS_3D = np.array(
    [
        (dz, dy, dx)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if not (dz == 0 and dy == 0 and dx == 0)
    ],
    dtype=np.int64,
)

SEGMENT_DEGREE = 2
BIFURCATION_DEGREE = 3
MIN_CURVATURE_POINTS = 3


def edges_to_segments(edges: np.ndarray) -> np.ndarray:
    """Convert 3D edge bitmask into an array of line segments."""
    depth, height, width = edges.shape
    segments: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                mask = int(edges[z, y, x])
                if mask == 0:
                    continue
                for bit in range(26):
                    if (mask >> bit) & 1:
                        dz, dy, dx = _OFFSETS_3D[bit]
                        nz, ny, nx = z + int(dz), y + int(dy), x + int(dx)
                        if 0 <= nz < depth and 0 <= ny < height and 0 <= nx < width:
                            # Avoid duplicate segments by canonical ordering.
                            if (nz, ny, nx) > (z, y, x):
                                segments.append(((x, y, z), (nx, ny, nz)))
    return np.asarray(segments, dtype=np.int64)


def segments_to_graph(segments: Iterable[Segment]) -> nx.Graph:
    """Build a NetworkX graph from pairs of segment endpoints."""
    graph = nx.Graph()
    for p1, p2 in segments:
        p1_tuple = tuple(int(x) for x in p1)
        p2_tuple = tuple(int(x) for x in p2)

        graph.add_node(p1_tuple, pos=np.array(p1_tuple))
        graph.add_node(p2_tuple, pos=np.array(p2_tuple))
        graph.add_edge(p1_tuple, p2_tuple)
    return graph


def obtain_radius_map(vessels: np.ndarray, graph: nx.Graph) -> dict[Point3D, float]:
    """Compute a radius map for each node based on the distance transform."""
    dist = distance_transform_edt(vessels)
    return {node: float(dist[node[2], node[1], node[0]]) for node in graph.nodes()}


def extract_segments(graph: nx.Graph) -> list[PathList]:
    """Return polylines that trace each vessel segment between bifurcations."""
    segments: list[PathList] = []
    visited: set[tuple[Point3D, ...]] = set()

    for node in graph.nodes():
        if graph.degree(node) != SEGMENT_DEGREE:
            for nei in graph.neighbors(node):
                path: PathList = [node]
                prev = node
                curr = nei

                while graph.degree(curr) == SEGMENT_DEGREE:
                    path.append(curr)
                    nxt = [n for n in graph.neighbors(curr) if n != prev][0]
                    prev, curr = curr, nxt

                path.append(curr)

                tup = tuple(path)
                if tup not in visited:
                    visited.add(tup)
                    segments.append(path)
    return segments


def compute_segment_metrics(
    path: PathList, radius_map: dict[Point3D, float]
) -> dict[str, Any]:
    """Calculate geometry and radius statistics for a vessel segment."""
    coords = np.array([tuple(int(v) for v in p) for p in path], dtype=float)
    radii = np.array([float(radius_map[p]) for p in path], dtype=float)

    diffs = np.diff(coords, axis=0)
    length = float(np.sum(np.linalg.norm(diffs, axis=1)))

    straight = float(np.linalg.norm(coords[-1] - coords[0]))
    tortuosity = float(length / straight) if straight > 0 else 1.0

    volume = float(np.sum(np.pi * radii**2))

    if len(coords) >= MIN_CURVATURE_POINTS:
        v1 = coords[2:] - coords[1:-1]
        v2 = coords[1:-1] - coords[:-2]
        angles = []
        for a, b in zip(v1, v2):
            dot = float(np.dot(a, b))
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            cosang = max(-1.0, min(1.0, dot / denom))
            angles.append(np.arccos(cosang))
        curvature = np.array(angles, dtype=float)
    else:
        curvature = np.array([0.0], dtype=float)

    return {
        "segment": {
            "start": [int(v) for v in coords[0]],
            "end": [int(v) for v in coords[-1]],
        },
        "radius": {
            "mean": float(np.mean(radii)),
            "sd": float(np.std(radii)),
            "median": float(np.median(radii)),
            "min": float(np.min(radii)),
            "q1": float(np.percentile(radii, 25)),
            "q3": float(np.percentile(radii, 75)),
            "max": float(np.max(radii)),
        },
        "length": float(length),
        "tortuosity": float(tortuosity),
        "volume": float(volume),
        "curvature": {
            "mean": float(np.mean(curvature)),
            "sd": float(np.std(curvature)),
            "median": float(np.median(curvature)),
            "min": float(np.min(curvature)),
            "q1": float(np.percentile(curvature, 25)),
            "q3": float(np.percentile(curvature, 75)),
            "max": float(np.max(curvature)),
        },
    }


def detect_bifurcations(graph: nx.Graph) -> list[dict[str, Any]]:
    """Detect bifurcation points and compute opening angles."""
    bifurcations: list[dict[str, Any]] = []

    for node in graph.nodes():
        if graph.degree(node) == BIFURCATION_DEGREE:
            neigh = list(graph.neighbors(node))
            p0 = np.array(node)
            vecs = [np.array(n) - p0 for n in neigh]

            def angle(a: np.ndarray, b: np.ndarray) -> float:
                return float(
                    np.degrees(
                        np.arccos(
                            np.clip(
                                np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)),
                                -1,
                                1,
                            )
                        )
                    )
                )

            bdict = {
                "bifurcation": {
                    "midpoint": [int(a) for a in node],
                    "points_angle": [[int(a) for a in p] for p in neigh],
                },
                "angles": {
                    "pair1": angle(vecs[0], vecs[1]),
                    "pair2": angle(vecs[0], vecs[2]),
                    "pair3": angle(vecs[1], vecs[2]),
                },
            }

            bifurcations.append(bdict)

    return bifurcations


def assign_component_labels(graph: nx.Graph) -> dict[int, str]:
    """Identify connected components and assign default labels."""
    components = list(nx.connected_components(graph))

    vessel_labels = {i + 1: f"component_{i+1}" for i in range(len(components))}

    for idx, comp in enumerate(components, start=1):
        for node in comp:
            graph.nodes[node]["component"] = idx

    return vessel_labels


def build_vessel_json(
    graph: nx.Graph,
    vessel_labels: dict[int, str],
    segment_paths: list[PathList],
    radius_map: dict[Point3D, float],
    bifurcations: list[dict[str, Any]],
    output_path: str | Path = Path("vessels_morphometry.json"),
) -> dict[str, Any]:
    """Build final vessel morphometry JSON grouped by connected component."""
    final_json: dict[str, Any] = {}

    for idx in vessel_labels:
        comp_nodes = {
            n for n in graph.nodes() if graph.nodes[n].get("component") == idx
        }

        segs = [p for p in segment_paths if tuple(p[0]) in comp_nodes]

        seg_metrics = [compute_segment_metrics(p, radius_map) for p in segs]

        bif = []

        for entry in bifurcations:
            if not isinstance(entry, dict):
                continue

            if "bifurcation" not in entry or not isinstance(entry["bifurcation"], dict):
                continue

            midpoint = entry["bifurcation"].get("midpoint")
            if midpoint is None:
                continue

            midpoint_tuple = tuple(midpoint)

            if midpoint_tuple in comp_nodes:
                bif.append(entry)

        entry: dict[str, Any] = {vessel_labels[idx]: seg_metrics}

        if bif:
            entry[f"{vessel_labels[idx]} bifurcation"] = bif

        final_json[str(idx)] = entry

    output_path = Path(output_path)
    with output_path.open("w") as file_handle:
        json.dump(final_json, file_handle, indent=4)

    print(f"✔ Vessel morphometry JSON saved to: {output_path}")
    return final_json
