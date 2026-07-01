# GNN track — raw vessel-graph datasets

This package delivers the **raw** vessel graph (nodes, edges, node features) as
[`torch_geometric.data.Data`](https://pytorch-geometric.readthedocs.io/) objects,
in contrast to the tabular / Deep Sets pipelines, which consume *summarized*
morphometry + kinematic features. It is the data layer for the GNN modeling
track.

`gnn/data_loader.py` provides `VanguardCenterlineDataset`, an
`InMemoryDataset` that builds one graph per case from saved centerline outputs.

## What it builds

- **Nodes:** one per skeleton voxel (voxel-level granularity), keyed by
  `(x, y, z)`. A `segment` mode (one node per bifurcation/endpoint) is an
  explicit, not-yet-implemented extension point (`node_mode="segment"` raises).
- **Edges:** between 26-connected skeleton voxels (undirected → symmetric
  `edge_index`).
- **`data.pos`:** `(num_nodes, 3)` voxel coordinates `(x, y, z)`.
- **`data.x`:** node features stacked in the order given by `node_features`
  (default `("peak_time", "radius")`):
  - `peak_time` → normalized peak-contrast time,
    `argmax_t(signal_4d[:, z, y, x] − signal_4d[0, z, y, x]) / (T − 1)`.
    The raw integer index is also kept on `data.peak_time`.
  - `radius` → local vessel radius from the support mask's distance transform.
- **`data.y`:** the binary label (**required** — see below).
- **Metadata:** `data.case_id`, `data.dataset`, `data.site`, `data.num_timepoints`.

The mask → graph conversion reuses the existing `graph_extraction` primitives
(`mask_to_edges_bitmask`, `edges_to_segments`, `segments_to_graph`,
`obtain_radius_map`) so there is a single source of truth. Peak-contrast time is
computed from the same `*_vessel_segmentation.npz` timepoints that
`features/kinematic.py` samples via `signal_4d[:, z, y, x]`.

## Input tree

`root` is the centerline `studies/` tree. Each case directory contains:

```
<studies>/<dataset>/<case_id>/
  <case_id>_skeleton_4d_exam_mask.npy          # 3D (z,y,x) uint8 centerline
  <case_id>_skeleton_4d_exam_support_mask.npy  # 3D (z,y,x) uint8 vessel support
  run_summary.json                             # study_files -> per-timepoint npz
```

The loader reads `run_summary.json["study_files"]` (absolute `*_vessel_segmentation.npz`
paths) to load the 4D signal without re-globbing. If that key is absent, pass
`timeseries_root=<vessel-segmentation root>` and it falls back to
`graph_extraction.core4d.discover_study_timepoints`.

## Labels are required

Every graph must carry a label. Labels are loaded with
`tabular.cohort.load_labels(labels_path, id_column, label_column)` (normalizes to
`{0, 1}`, handles CSV/JSON and true/false). **Cases with no matching label are
skipped** with a logged reason, so a built dataset is always training-ready and
never carries unlabeled graphs. Matched/skipped counts are logged at the end of
`process()`.

## Usage

```python
from gnn.data_loader import VanguardCenterlineDataset

dataset = VanguardCenterlineDataset(
    root="/path/to/centerlines/studies",
    labels_path="/path/to/labels.csv",     # required
    id_column="case_id",
    label_column="pcr",
    node_features=("peak_time", "radius"),
    cache_dir="/path/to/gnn_cache",         # defaults to <root>/gnn_cache
    cases=["NACT_62", "NACT_63"],           # optional whitelist
    profile=True,                           # log per-stage timings
)
data = dataset[0]                            # torch_geometric.data.Data
```

The collated cache is written to `<cache_dir>/processed/data.pt`, plus a
per-case `<case_id>_graph.pt` for debugging. Delete `<cache_dir>/processed/` to
force a rebuild.

## Profiling

With `profile=True` the loader accumulates wall time for each build stage
(`mask_load`, `graph_build`, `timeseries_load`, `peak_time`, `from_networkx`) and
logs mean / median / max across cases. The **4D `timeseries_load`** stage is the
runtime watch item as we move from the small MAMA-MIA cohort to UChicago, where
each case has many timepoints.

> Record the observed per-stage numbers here after the first real-data run on the
> cluster (login-node smoke set of 2–3 NACT cases, then the full cohort via
> Slurm).

## Verification status

- **Synthetic smoke test** (`tests/test_gnn_data_loader.py`): fabricates a tiny
  centerline tree and asserts `num_nodes > 0`, symmetric `edge_index`,
  `x.shape[1] == len(node_features)`, `pos.shape == (num_nodes, 3)`, `peak_time`
  within `[0, T−1]`, `data.y` present, and that an unlabeled case is skipped.
  Runs in CI (installs a CPU `torch` + `torch_geometric` from PyPI).
- **Real data:** the public MAMA-MIA centerlines and private UChicago data live
  on `/net/projects2/...` cluster paths, so real-data builds run in the cluster
  conda env (pinned `torch` 1.11 / PyG 2.3.1), not in CI. The `from_networkx`,
  `InMemoryDataset`, and `Data` APIs used here are stable across PyG 2.x.

## Resolved design decisions

Captured here because `$SV_DATA_DIR/gnn_data_loader_design_doc.md` was not
reachable from this environment (`SV_DATA_DIR` unset); fold these back into that
doc when it is available.

- Node granularity: **voxel-level now**, `node_mode` flag reserved for `segment`.
- Build path: reuse the NetworkX primitives → `from_networkx` (single source of
  truth); profiling instrumentation built in.
- Storage: `InMemoryDataset` (cohort n≈200 fits in memory).
- Node feature v1: peak-contrast time (+ radius), computed from the raw 4D
  timepoints.
- Labels: **required**; unlabeled cases skipped. 
