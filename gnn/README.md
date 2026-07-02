# GNN modeling track

This package is the GNN counterpart to the tabular / Deep Sets pipelines: instead
of consuming *summarized* morphometry + kinematic features, it operates on the
**raw** vessel graph (nodes, edges, node features) as
[`torch_geometric.data.Data`](https://pytorch-geometric.readthedocs.io/) objects.
It covers the full track end to end:

- **`gnn/data_loader.py`** — `VanguardCenterlineDataset`, an `InMemoryDataset`
  that builds one graph per case from saved centerline outputs.
- **`gnn/build_dataset.py`** — CLI wrapper to build/cache the full dataset on
  the cluster.
- **`gnn/model.py`** — `GCNClassifier`, the current MVP model.
- **`gnn/train.py`** — minimal training script (single stratified split,
  standardization, `BCEWithLogitsLoss`, shared metrics).
- **`gnn/slurm/`** — Slurm submission scripts for both build and train.

## Data pipeline

### What it builds

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

### Input tree

`root` is the centerline `studies/` tree. Each case directory contains:

```
<studies>/<dataset>/<case_id>/
  <case_id>_skeleton_4d_exam_mask.npy          # 3D (z,y,x) uint8 centerline
  <case_id>_skeleton_4d_exam_support_mask.npy  # 3D (z,y,x) uint8 vessel support
  run_summary.json                             # study_files -> per-timepoint npz
```

The loader reads `run_summary.json["study_files"]` (absolute `*_vessel_segmentation.npz`
paths) to load the 4D signal. `run_summary.json` and its `study_files` key are
**required** — the loader raises immediately if either is absent or malformed.

### Labels are required

Every graph must carry a label. Labels are loaded with
`tabular.cohort.load_labels(labels_path, id_column, label_column)` (normalizes to
`{0, 1}`, handles CSV/JSON and true/false). **Cases with no matching label are
dropped** (not built) with a loud `logging.warning` and recorded in
`dataset.dropped_case_ids` / `<cache_dir>/processed/dropped_cases.json`, so a
built dataset is always training-ready and never carries unlabeled graphs. If
the dropped fraction exceeds `max_missing_label_frac` (default `0.1`) the
whole build raises `RuntimeError` instead of silently training on a shrunken
cohort -- this guards against a labels file that's stale, mismatched, or
pointed at the wrong cohort. On a cache hit (no rebuild), the manifest is
reloaded and the drop warning is re-logged so missing labels stay visible
without needing a fresh build. Degenerate geometry (empty skeleton, zero
segments, shape mismatch) still raises immediately -- that's a data problem,
not an expected missing-label case.

### Usage

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

### Building the full dataset on the cluster

`gnn/build_dataset.py` is a thin CLI wrapper: constructing
`VanguardCenterlineDataset` triggers the build, so the script just parses args
and instantiates it. Defaults point at the real cluster paths (see the
project `CLAUDE.md`), with the cache written to Spencer's own workspace
(`/gpfs/data/karczmar-lab/workspaces/spencervenancio/gnn_cache`) rather than
into `saritbose`'s centerline tree.

```bash
sbatch gnn/slurm/submit_gnn_build.slurm
```

Override any path via environment variables (`ROOT`, `LABELS_PATH`,
`CACHE_DIR`, `ID_COLUMN`, `LABEL_COLUMN`, `NODE_FEATURES`, `CASES`,
`NO_CACHE=1`) -- see the script header for usage. `CASES` (comma-separated
case IDs) is handy for a smoke-test submission before committing to the full
~1500-case build. See `docs/slurm-site.md` for why the job targets `tier1q`
rather than the `general` partition used by older scripts in this repo.

### Profiling

With `profile=True` the loader accumulates wall time for each build stage
(`mask_load`, `graph_build`, `timeseries_load`, `peak_time`, `from_networkx`) and
logs mean / median / max across cases. The **4D `timeseries_load`** stage is the
runtime watch item as we move from the small MAMA-MIA cohort to UChicago, where
each case has many timepoints.

> Record the observed per-stage numbers here after the first real-data run on the
> cluster (login-node smoke set of 2–3 NACT cases, then the full cohort via
> Slurm).

### Data verification status

- **Synthetic smoke test** (`tests/test_gnn_data_loader.py`): fabricates a tiny
  centerline tree and asserts `num_nodes > 0`, symmetric `edge_index`,
  `x.shape[1] == len(node_features)`, `pos.shape == (num_nodes, 3)`, `peak_time`
  within `[0, T−1]`, `data.y` present, and that an unlabeled case is skipped.
  Runs in CI (installs a CPU `torch` + `torch_geometric` from PyPI).
- **Real data:** the public MAMA-MIA centerlines and private UChicago data live
  on `/net/projects2/...` cluster paths, so real-data builds run in the cluster
  conda env (pinned `torch` 1.11 / PyG 2.3.1), not in CI. The `from_networkx`,
  `InMemoryDataset`, and `Data` APIs used here are stable across PyG 2.x.

### Resolved data-loader design decisions

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

## Model + training (MVP)

`gnn/model.py` provides `GCNClassifier`: a stack of `GCNConv` layers, a global
mean-pool readout to one embedding per graph, and a linear head producing one
logit per graph. No edge features, attention, or segment-level pooling yet --
those are deliberate next steps once this MVP is validated end-to-end, not
omissions. `gnn/train.py` is a minimal training script: a single stratified
train/val split over case indices (not k-fold CV), node-feature
standardization fit on the train split only, plain `BCEWithLogitsLoss`, and
metrics via the shared `evaluation.metrics.compute_binary_metrics` (so numbers
stay comparable once this grows into a pipeline that mirrors
`deepsets/train.py` -- YAML config, `evaluation/build_splits.py` k-fold CV,
results-dir README convention -- which it intentionally does not attempt yet).

```bash
python -m gnn.train --cache-dir /path/to/gnn_cache_smoke --cases NACT_01,NACT_02
```

Every run writes `README.md` + `metrics.json` + `config_used.json` to
`experiments/gnn_mvp_<timestamp>/` (or `--outdir`) per the project's auditing
convention. `gnn/slurm/submit_gnn_train.slurm` mirrors
`submit_gnn_build.slurm` and defaults to the 8-case `gnn_cache_smoke`.