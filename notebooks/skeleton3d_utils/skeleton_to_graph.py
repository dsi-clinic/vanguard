import networkx as nx
import numpy as np

from scipy.ndimage import distance_transform_edt
import json


def segments_to_graph(segments):
    G = nx.Graph()
    for p1, p2 in segments:
        p1 = tuple(int(x) for x in p1)  # ensure hashable ints
        p2 = tuple(int(x) for x in p2)
        
        G.add_node(p1, pos=np.array(p1))
        G.add_node(p2, pos=np.array(p2))
        G.add_edge(p1, p2)
    return G

def obtain_radius_map(vessels, G):
    dist = distance_transform_edt(vessels)
    radius_map = {node: float(dist[node[2], node[1], node[0]]) for node in G.nodes()}
    return radius_map

def extract_segments(G):
    segments = []
    visited = set()

    for node in G.nodes():
        # endpoints or bifurcations
        if G.degree(node) != 2:
            for nei in G.neighbors(node):
                path = [node]
                prev = node
                curr = nei

                while G.degree(curr) == 2:
                    path.append(curr)
                    nxt = [n for n in G.neighbors(curr) if n != prev][0]
                    prev, curr = curr, nxt

                path.append(curr)

                tup = tuple(path)
                if tup not in visited:
                    visited.add(tup)
                    segments.append(path)
    return segments

def compute_segment_metrics(path, radius_map):
    import numpy as np
    
    # ensure Python ints for coordinates
    coords = np.array([tuple(int(v) for v in p) for p in path], dtype=float)
    radii = np.array([float(radius_map[p]) for p in path], dtype=float)

    # Length
    diffs = np.diff(coords, axis=0)
    length = float(np.sum(np.linalg.norm(diffs, axis=1)))

    # Tortuosity
    straight = float(np.linalg.norm(coords[-1] - coords[0]))
    tort = float(length / straight) if straight > 0 else 1.0

    # Volume
    vol = float(np.sum(np.pi * radii**2))

    # Curvature
    if len(coords) >= 3:
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
            "end":   [int(v) for v in coords[-1]]
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
        "tortuosity": float(tort),
        "volume": float(vol),
        "curvature": {
            "mean": float(np.mean(curvature)),
            "sd": float(np.std(curvature)),
            "median": float(np.median(curvature)),
            "min": float(np.min(curvature)),
            "q1": float(np.percentile(curvature, 25)),
            "q3": float(np.percentile(curvature, 75)),
            "max": float(np.max(curvature)),
        }
    }

def detect_bifurcations(G):
    bif = []

    for node in G.nodes():
        if G.degree(node) == 3:
            neigh = list(G.neighbors(node))
            p0 = np.array(node)
            vecs = [np.array(n) - p0 for n in neigh]

            # angle helper
            def angle(a, b):
                return float(np.degrees(
                    np.arccos(
                        np.clip(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b)), -1, 1)
                    )
                ))

            bdict = {
                "bifurcation": {
                    "midpoint": [int(a) for a in node],
                    "points_angle": [[int(a) for a in p] for p in neigh]
                },
                "angles": {
                    "pair1": angle(vecs[0], vecs[1]),
                    "pair2": angle(vecs[0], vecs[2]),
                    "pair3": angle(vecs[1], vecs[2]),
                }
            }

            bif.append(bdict)

    return bif


def assign_component_labels(G):
    """
    Identify connected components in the skeleton graph and assign
    each node a component ID. Also return a {id: name} label map.

    Parameters
    ----------
    G : networkx.Graph
        The skeleton graph.

    Returns
    -------
    vessel_labels : dict
        Maps component index → default vessel label ("component_1", ...)
    """
    components = list(nx.connected_components(G))

    # Default name: component_1, component_2, ...
    vessel_labels = {i+1: f"component_{i+1}" for i in range(len(components))}

    # Assign component id to each node
    for idx, comp in enumerate(components, start=1):
        for node in comp:
            G.nodes[node]["component"] = idx

    return vessel_labels


import json

def build_vessel_json(
    G,
    vessel_labels,
    segment_paths,
    radius_map,
    bifurcations,
    output_path="vessels_morphometry.json"
):
    """
    Build final vessel morphometry JSON grouped by connected component
    and save the output to a predefined file path.

    Parameters
    ----------
    G : networkx.Graph
        Skeleton graph with node["component"] already assigned.
    vessel_labels : dict
        Mapping {component_id: vessel_name}.
    segment_paths : list[list[tuple]]
        Each entry is a list of (x,y,z) positions defining a curved segment.
    radius_map : dict
        Mapping {(x,y,z): radius}.
    bifurcations : list[dict]
        Output from detect_bifurcations(). May contain malformed entries.
    output_path : str
        File path where the JSON file will be saved.

    Returns
    -------
    dict
        The full JSON dictionary that is also saved to disk.
    """

    final_json = {}

    for idx in vessel_labels:

        # --------------------------
        # Collect nodes in this component
        # --------------------------
        comp_nodes = {
            n for n in G.nodes()
            if G.nodes[n].get("component") == idx
        }

        # --------------------------
        # Collect segments in this component
        # --------------------------
        segs = [p for p in segment_paths if tuple(p[0]) in comp_nodes]

        seg_metrics = [
            compute_segment_metrics(p, radius_map)
            for p in segs
        ]

        # --------------------------
        # Collect bifurcations in this component
        # (robust filtering to avoid malformed entries)
        # --------------------------
        bif = []

        for b in bifurcations:

            # must be dict
            if not isinstance(b, dict):
                continue

            # must contain "bifurcation" block
            if "bifurcation" not in b:
                continue
            if not isinstance(b["bifurcation"], dict):
                continue

            # must contain midpoint
            midpoint = b["bifurcation"].get("midpoint")
            if midpoint is None:
                continue

            midpoint = tuple(midpoint)

            # is this bifurcation inside this component?
            if midpoint in comp_nodes:
                bif.append(b)

        # --------------------------
        # Build JSON entry
        # --------------------------
        entry = {vessel_labels[idx]: seg_metrics}

        if len(bif) > 0:
            entry[f"{vessel_labels[idx]} bifurcation"] = bif

        final_json[str(idx)] = entry

    # --------------------------
    # Save to file
    # --------------------------
    with open(output_path, "w") as f:
        json.dump(final_json, f, indent=4)

    print(f"✔ Vessel morphometry JSON saved to: {output_path}")