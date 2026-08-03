"""Export one real vessel subgraph + real DCE frames for the TikZ architecture figure.

The architecture figure must show the *same* graph at every stage (inputs, node
features, per-frame message passing, forecast targets, fine-tuning readout) so a
reader can follow one object through the pipeline. That means the graph has to be
extracted once, here, and handed to TikZ as coordinates -- not redrawn by hand in
each panel.

Everything exported is measured, not schematic:

  * the MRI panels are real DCE-MRI frames from one MAMA-MIA **Duke** case
    (public cohort -- the poster must not print UChicago slices),
  * the graph is a connected ball of the real 26-connected centerline voxel
    graph, grown around the case's busiest branch point,
  * the junction and segment graphs are *derived from that same subgraph* with
    the repo's own primitives, so the three representations are three views of
    one object rather than three drawings,
  * the node curves are the real baseline-referenced DCE enhancement curves at
    those voxels, through ``gnn.kinetics.baseline_relative_curve`` -- the same
    transform ``gnn.pretrain.node_series`` uses for the forecasting task.

The 2D layout is the subgraph's own principal plane -- the first two principal
components of its real 3D voxel coordinates -- not a hand-drawn arrangement, so
every drawn node sits where the vessel actually runs. An axis-aligned (x, y)
projection was tried first and collapses distinct voxels onto one point wherever
a vessel runs along z. Near-coincident nodes are still refused rather than
jittered, since a jittered node is a node whose position is no longer measured.

The MRI panels keep the honest raster treatment instead: they show the real slab
projection with the real centerline painted on it, and make no claim that a drawn
graph node sits at a particular pixel.

Loads one case's raw 4D DCE -- run via Slurm, not the head node.

Usage:
    python -m analysis.poster_architecture_data --case-id DUKE_107 \
        --centerline-root /gpfs/data/karczmar-lab/workspaces/saritbose/centerlines_tc4d/studies \
        --dce-root /gpfs/data/karczmar-lab/MAMA-MIA-syn60868042/images \
        --out-dir presentations/symposium_poster/figures/architecture_data
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from analysis.floor_activation_map import _resolve_case  # noqa: E402
from analysis.poster_pipeline_figure import _OFFSETS_3D  # noqa: E402
from gnn.data_loader import _load_study_metadata  # noqa: E402
from gnn.kinetics import baseline_relative_curve  # noqa: E402
from gnn.raw_dce import discover_raw_dce_paths, load_raw_dce_series  # noqa: E402
from graph_extraction.skeleton_to_graph_primitives import (  # noqa: E402
    extract_segments,
    obtain_radius_map,
)

Point3D = tuple[int, int, int]

JUNCTION_DEGREE = 2  # deg > 2 is a branch point; deg == 1 is a free end


def _voxel_graph(skeleton: np.ndarray) -> nx.Graph:
    """The real 26-connected centerline voxel graph over the whole skeleton.

    Same adjacency the training pipeline builds with
    ``edges_to_segments(mask_to_edges_bitmask(...))``, but evaluated over the
    ~10^4 skeleton voxels directly instead of looping all 448x448x160 volume
    positions, which is the only reason this is cheap enough to iterate on.
    """
    # Node keys are (x, y, z), the order used everywhere else in the repo
    # (``gnn.data_loader``, ``obtain_radius_map``, ``gnn.pretrain.node_series``),
    # while ``np.argwhere`` on a (z, y, x) volume yields the reverse.
    nodes = {(int(x), int(y), int(z)) for z, y, x in np.argwhere(skeleton)}

    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    for x, y, z in nodes:
        for dz, dy, dx in _OFFSETS_3D:
            neighbor = (x + dx, y + dy, z + dz)
            if neighbor in nodes and neighbor > (x, y, z):
                graph.add_edge((x, y, z), neighbor)
    return graph


def _pick_subgraph(graph: nx.Graph, target_nodes: int) -> nx.Graph:
    """Breadth-first ball of ~``target_nodes`` voxels around the busiest branch.

    Grown one full BFS ring at a time so the result never stops mid-segment with
    a half-drawn branch: the ball is truncated at ring boundaries, which keeps
    every included branch either complete to its next junction or clearly running
    off the edge of the window.
    """
    component = max(nx.connected_components(graph), key=len)
    core = graph.subgraph(component).copy()
    seed = max(core.degree, key=lambda item: item[1])[0]

    visited: set[Point3D] = {seed}
    frontier: deque[Point3D] = deque([seed])
    while frontier and len(visited) < target_nodes:
        ring = list(frontier)
        frontier.clear()
        for node in ring:
            for neighbor in core.neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    frontier.append(neighbor)
    return core.subgraph(visited).copy()


def _principal_plane_layout(
    nodes: list[Point3D], min_separation_frac: float
) -> dict[Point3D, tuple[float, float]]:
    """Lay the subgraph out on its own principal plane, rescaled to a unit box.

    The two leading principal components of the real (x, y, z) voxel coordinates
    are the plane that loses the least of the subgraph's actual shape, so the
    drawing is a projection of measured anatomy rather than an arrangement chosen
    for looks. Coordinates are then rescaled with a single factor -- longest side
    1.0 -- which preserves aspect ratio and keeps every panel's copy of the graph
    the same shape.

    Two nodes closer than ``min_separation_frac`` of the median edge length would
    print as one dot at poster size; that raises rather than nudging them apart.
    """
    coords = np.array(nodes, dtype=float)
    centered = coords - coords.mean(axis=0)
    _, _, components = np.linalg.svd(centered, full_matrices=False)
    planar = centered @ components[:2].T

    span = float((planar.max(axis=0) - planar.min(axis=0)).max())
    unit_coords = (planar - planar.min(axis=0)) / span

    deltas = unit_coords[:, None, :] - unit_coords[None, :, :]
    distances = np.linalg.norm(deltas, axis=-1)
    np.fill_diagonal(distances, np.inf)
    median_edge = float(np.median(distances.min(axis=1)))
    closest = float(distances.min())
    if closest < min_separation_frac * median_edge:
        raise ValueError(
            f"two nodes are {closest:.4f} apart in the layout, below "
            f"{min_separation_frac:.2f} x the median {median_edge:.4f} node "
            "spacing; they would print as one dot. Lower --target-nodes or pick "
            "another case rather than nudging them apart."
        )
    return {node: (float(x), float(y)) for node, (x, y) in zip(nodes, unit_coords)}


def _junction_graph(voxel: nx.Graph) -> tuple[list[Point3D], list[list[Point3D]]]:
    """Branch/end voxels, and the segment paths that connect them.

    ``extract_segments`` is the repo's own segment definition, so the junction and
    segment views here are the same ones ``gnn.junction_graph`` / ``gnn.segment_graph``
    build for training.
    """
    junctions = [node for node, degree in voxel.degree if degree != JUNCTION_DEGREE]
    return junctions, extract_segments(voxel)


def _write_mri_panels(
    dce_4d: np.ndarray,
    skeleton: np.ndarray,
    slab: tuple[int, int],
    box: tuple[int, int, int, int],
    out_dir: Path,
) -> tuple[float, float]:
    """Write one PNG per DCE frame, all on one shared intensity window.

    Per ``AGENTS.md`` the window is computed once over the whole slab-projected
    4D crop; per-frame windowing would flatten exactly the enhancement change the
    figure is about. A final ``mri_centerline.png`` repeats the last frame with
    the real centerline painted on, which is how the figure shows "the graph came
    from this image" without claiming pixel positions for the drawn nodes.
    """
    z0, z1 = slab
    x0, x1, y0, y1 = box
    projected = dce_4d[:, z0:z1, y0 : y1 + 1, x0 : x1 + 1].max(axis=1)
    finite = projected[np.isfinite(projected)]
    vmin = float(np.percentile(finite, 1.0))
    vmax = float(np.percentile(finite, 99.5))

    def _canvas() -> tuple[plt.Figure, plt.Axes]:
        figure = plt.figure(figsize=(3.0, 3.0), dpi=220)
        axes = figure.add_axes((0, 0, 1, 1))
        axes.axis("off")
        return figure, axes

    for frame in range(projected.shape[0]):
        fig, ax = _canvas()
        ax.imshow(projected[frame], cmap="gray", vmin=vmin, vmax=vmax, origin="upper")
        fig.savefig(out_dir / f"mri_frame_{frame:02d}.png", dpi=220)
        plt.close(fig)

    centerline_2d = skeleton[z0:z1, y0 : y1 + 1, x0 : x1 + 1].any(axis=0)
    fig, ax = _canvas()
    ax.imshow(projected[-1], cmap="gray", vmin=vmin, vmax=vmax, origin="upper")
    overlay = np.zeros((*centerline_2d.shape, 4))
    overlay[centerline_2d] = matplotlib.colors.to_rgba("#f2c14e")
    ax.imshow(overlay, origin="upper", interpolation="nearest")
    fig.savefig(out_dir / "mri_centerline.png", dpi=220)
    plt.close(fig)
    return vmin, vmax


def _tex_points(points: list[tuple[float, float]]) -> str:
    """Format a coordinate list as TikZ ``x/y`` foreach items."""
    return ", ".join(f"{x:.4f}/{y:.4f}" for x, y in points)


def _write_graph_tex(
    path: Path,
    unit: dict[Point3D, tuple[float, float]],
    voxel: nx.Graph,
    junctions: list[Point3D],
    segments: list[list[Point3D]],
    heat: np.ndarray,
) -> None:
    """Emit the one shared graph as TikZ foreach lists.

    Every panel of the figure draws from these macros, which is what makes the
    graph the *same* graph in each panel rather than five similar sketches.
    """
    index = {node: i for i, node in enumerate(sorted(unit))}
    junction_set = set(junctions)

    lines = [
        "% Generated by analysis/poster_architecture_data.py -- do not edit.",
        "% One real vessel subgraph, shared by every panel of the architecture figure.",
        "",
    ]

    voxel_pts = [unit[node] for node in sorted(unit)]
    lines.append(f"\\def\\voxelnodes{{{_tex_points(voxel_pts)}}}")

    # Per-node enhancement fields, rescaled to 0-100 so TikZ can shade each node
    # by one. Row -1 is the final forecast target; rows 0..T-1 are the individual
    # frames, which is what lets the encoder's per-frame plates differ in node
    # value while sharing one topology -- the actual content of message passing.
    # These are measured fields, not model output: no forecaster is run here, so
    # the figure shows the target the decoder is trained on, never a prediction.
    for tag, row in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", heat):
        lines.append(
            f"\\def\\voxelheat{tag}{{{', '.join(str(int(round(v))) for v in row)}}}"
        )
    lines.append(
        f"\\def\\voxelheat{{{', '.join(str(int(round(v))) for v in heat[-1])}}}"
    )

    voxel_edges = [
        f"{index[a]}/{index[b]}"
        for a, b in sorted(voxel.edges(), key=lambda e: index[e[0]])
    ]
    lines.append(f"\\def\\voxeledges{{{', '.join(voxel_edges)}}}")

    junction_pts = [unit[node] for node in sorted(junction_set)]
    lines.append(f"\\def\\junctionnodes{{{_tex_points(junction_pts)}}}")

    junction_index = {node: i for i, node in enumerate(sorted(junction_set))}
    junction_edges = []
    segment_paths = []
    segment_pts = []
    for path_nodes in segments:
        ends = (path_nodes[0], path_nodes[-1])
        if all(end in junction_index for end in ends):
            junction_edges.append(
                f"{junction_index[ends[0]]}/{junction_index[ends[1]]}"
            )
        segment_paths.append(
            " ".join(f"({unit[n][0]:.4f},{unit[n][1]:.4f})" for n in path_nodes)
        )
        midpoint = path_nodes[len(path_nodes) // 2]
        segment_pts.append(unit[midpoint])
    lines.append(f"\\def\\junctionedges{{{', '.join(junction_edges)}}}")
    lines.append(f"\\def\\segmentnodes{{{_tex_points(segment_pts)}}}")

    # Two segments are adjacent when they share an endpoint voxel -- the line
    # graph the segment-as-node representation trains on.
    segment_edges = []
    for i, first in enumerate(segments):
        for j, second in enumerate(segments[i + 1 :], start=i + 1):
            if {first[0], first[-1]} & {second[0], second[-1]}:
                segment_edges.append(f"{i}/{j}")
    lines.append(f"\\def\\segmentedges{{{', '.join(segment_edges)}}}")

    for i, drawn in enumerate(segment_paths):
        lines.append(f"\\def\\segmentpath{chr(ord('A') + i)}{{{drawn}}}")
    lines.append(f"\\def\\segmentcount{{{len(segments)}}}")

    path.write_text("\n".join(lines) + "\n")


def _write_curves_tex(
    path: Path,
    curves: np.ndarray,
    times: list[int],
    highlight: list[int],
    input_len: int,
) -> None:
    """Emit real enhancement curves for the highlighted nodes as pgfplots coords.

    Split at ``input_len`` into the horizon the encoder sees and the horizon it
    forecasts, so the figure can draw the pretext task on measured signal.
    """
    lines = [
        "% Generated by analysis/poster_architecture_data.py -- do not edit.",
        "% Real baseline-referenced DCE enhancement at selected graph nodes.",
        "",
    ]
    scale = float(np.abs(curves[highlight]).max())
    for slot, node in enumerate(highlight):
        tag = chr(ord("A") + slot)
        normalized = curves[node] / scale
        observed = " ".join(
            f"({t},{v:.4f})" for t, v in zip(times[:input_len], normalized[:input_len])
        )
        future = " ".join(
            f"({t},{v:.4f})"
            for t, v in zip(times[input_len - 1 :], normalized[input_len - 1 :])
        )
        lines.append(f"\\def\\curveobs{tag}{{{observed}}}")
        lines.append(f"\\def\\curvefut{tag}{{{future}}}")
    lines.append(f"\\def\\curveinputlen{{{input_len}}}")
    lines.append(f"\\def\\curvetotallen{{{len(times)}}}")
    lines.append(f"\\def\\curvelastinput{{{input_len - 1}}}")
    # Frame count varies by case, so the MRI strip iterates this rather than
    # hard-coding a length that silently drops frames on the next case.
    lines.append(
        f"\\def\\framelist{{{','.join(f'{t:02d}' for t in range(len(times)))}}}"
    )
    path.write_text("\n".join(lines) + "\n")


def _write_readme(
    out_dir: Path, args: argparse.Namespace, provenance: dict[str, object]
) -> None:
    """Write the audit README next to the generated figure inputs."""
    command = (
        "sbatch gnn/slurm/submit_poster_architecture_data.slurm\n"
        f"  (CASE={args.case_id} TARGET_NODES={args.target_nodes} "
        f"SLAB_HALF_WIDTH={args.slab_half_width})"
    )
    out_dir.joinpath("README.md").write_text(
        f"""# Architecture-figure inputs

Real-data inputs for `../architecture.tex`, the poster's two-stage architecture
figure. Generated -- never hand-edited. Regenerating overwrites everything here.

## Regenerate

```
{command}
```

Then rebuild the figure with `../build_architecture.sh`.

Source: `analysis/poster_architecture_data.py`, git commit
`{provenance["git_commit"]}`.

## What is here

| File | Contents |
| --- | --- |
| `graph.tex` | The one shared subgraph: voxel nodes/edges, junction and segment derivations, and per-frame node shading, as TikZ `\\foreach` lists in unit coordinates. |
| `curves.tex` | Real baseline-referenced enhancement at three highlighted nodes, split into the input and forecast horizons. |
| `mri_frame_NN.png` | One real DCE-MRI frame each, slab max-projection, all on one shared intensity window. |
| `mri_centerline.png` | The last frame with the real centerline painted on. |
| `provenance.json` | Every parameter and count below, machine-readable. |

## Inputs

- Case `{provenance["case_id"]}` from the **public MAMA-MIA Duke cohort**, chosen
  because the poster is public and UChicago slices must not be printed on it.
- Skeleton: `{provenance["skeleton_path"]}`
- DCE root: `{provenance["dce_root"]}`
- Frames: {len(provenance["timepoints"])}; input horizon
  {provenance["input_len"]}, the rest are forecast targets.

## Numbers

- Whole-case centerline: {provenance["skeleton_voxels_total"]:,} voxels.
- Drawn subgraph: {provenance["subgraph_nodes"]} nodes,
  {provenance["subgraph_edges"]} edges, {provenance["junction_nodes"]} junction/end
  nodes, {provenance["segments"]} segments.
- Layout: {provenance["layout"]}.
- MRI slab z {provenance["mri_slab_z"]}, crop (x0,x1,y0,y1)
  {provenance["mri_crop_xxyy"]}, shared window {provenance["mri_intensity_window"]}.

## Things a reader should know

- The node shading in the figure is **measured** enhancement, on one scale shared
  across frames. No forecaster is run here, so the "forecast" panel shows the
  target field the decoder is trained on, not a prediction.
- Duke exams carry {len(provenance["timepoints"])} DCE phases. The method itself
  is applied to 13--24-frame ultrafast UChicago exams, so this figure understates
  the length of a real input sequence.
"""
    )


def main() -> None:
    """Extract one case's subgraph, curves, and MRI panels for the TikZ figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", type=str, required=True)
    parser.add_argument("--centerline-root", type=Path, required=True)
    parser.add_argument("--dce-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--slab-half-width",
        type=int,
        default=4,
        help="Slices above/below the subgraph's centre projected into the MRI panels.",
    )
    parser.add_argument(
        "--target-nodes",
        type=int,
        default=60,
        help="Approximate voxel-node count of the drawn subgraph.",
    )
    parser.add_argument(
        "--min-separation-frac",
        type=float,
        default=0.5,
        help="Refuse layouts whose closest node pair is below this x median spacing.",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=3,
        help="Frames in the forecasting input horizon; the rest are targets.",
    )
    parser.add_argument("--pad", type=int, default=6, help="MRI crop padding, voxels.")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mask_path, support_path = _resolve_case(args.centerline_root, args.case_id)
    skeleton = np.load(mask_path).astype(bool)
    support = np.load(support_path).astype(bool)
    timepoints, baseline_frame_count, relative_enhancement = _load_study_metadata(
        args.case_id, mask_path.parent
    )
    dce_paths = discover_raw_dce_paths(args.dce_root, args.case_id, timepoints)
    dce_4d = load_raw_dce_series(dce_paths, expected_shape_zyx=support.shape)

    voxel = _pick_subgraph(_voxel_graph(skeleton), args.target_nodes)
    ordered = sorted(voxel.nodes())

    unit = _principal_plane_layout(ordered, args.min_separation_frac)
    junctions, segments = _junction_graph(voxel)

    radius_map = obtain_radius_map(support, voxel)
    curves = baseline_relative_curve(
        np.stack([dce_4d[:, z, y, x] for x, y, z in ordered]),
        baseline_frame_count=baseline_frame_count,
        relative_enhancement=relative_enhancement,
    )

    # Highlight the three nodes with the largest positive enhancement area. Peak
    # -to-trough swing was tried first and selects for single-frame motion
    # spikes by construction; integrated enhancement is a real kinetic quantity
    # (the pipeline's own ``auc_positive``) and picks vessels that actually fill.
    enhancement_area = np.clip(curves, 0.0, None).sum(axis=1)
    highlight = sorted(np.argsort(enhancement_area)[-3:].tolist())

    # Node shading, one row per frame. One shared scale across all frames, for
    # the same reason the MRI panels share one intensity window: per-frame
    # rescaling would erase the enhancement change the figure exists to show.
    heat_lo, heat_hi = float(curves.min()), float(curves.max())
    heat = 100.0 * (curves.T - heat_lo) / (heat_hi - heat_lo)

    _write_graph_tex(out_dir / "graph.tex", unit, voxel, junctions, segments, heat)
    _write_curves_tex(
        out_dir / "curves.tex",
        curves,
        list(range(len(timepoints))),
        highlight,
        args.input_len,
    )

    # The MRI panels frame the same anatomy the graph was taken from: a slab
    # centred on the subgraph, cropped to its footprint plus a margin.
    voxels = np.array(ordered)  # columns are (x, y, z)
    centre_z = int(round(voxels[:, 2].mean()))
    slab = (
        max(centre_z - args.slab_half_width, 0),
        min(centre_z + args.slab_half_width + 1, skeleton.shape[0]),
    )
    crop = (
        max(int(voxels[:, 0].min()) - args.pad, 0),
        min(int(voxels[:, 0].max()) + args.pad, skeleton.shape[2] - 1),
        max(int(voxels[:, 1].min()) - args.pad, 0),
        min(int(voxels[:, 1].max()) + args.pad, skeleton.shape[1] - 1),
    )
    window = _write_mri_panels(dce_4d, skeleton, slab, crop, out_dir)

    commit = subprocess.run(  # noqa: S603
        ["git", "rev-parse", "HEAD"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    highlight_coords = [ordered[i] for i in highlight]
    provenance = {
        "case_id": args.case_id,
        "git_commit": commit,
        "skeleton_path": str(mask_path),
        "dce_root": str(args.dce_root),
        "timepoints": timepoints,
        "baseline_frame_count": baseline_frame_count,
        "relative_enhancement": relative_enhancement,
        "input_len": args.input_len,
        "layout": "principal-plane projection of the subgraph's 3D voxel coords",
        "mri_slab_z": list(slab),
        "mri_intensity_window": list(window),
        "node_shading_range": [heat_lo, heat_hi],
        "skeleton_voxels_total": int(skeleton.sum()),
        "subgraph_nodes": voxel.number_of_nodes(),
        "subgraph_edges": voxel.number_of_edges(),
        "junction_nodes": len(junctions),
        "segments": len(segments),
        "mri_crop_xxyy": list(crop),
        "highlight_nodes_xyz": [[int(v) for v in n] for n in highlight_coords],
        "highlight_radii": [radius_map[n] for n in highlight_coords],
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    _write_readme(out_dir, args, provenance)

    print(json.dumps(provenance, indent=2))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
