# Design: segment-level vessel graph for the GNN track

**Status:** proposal · **Branch:** `feature/segment-as-node` · **Author:** design doc for
review before implementation.

## 1. Problem

Today `gnn/data_loader.py` builds a **voxel-as-node** graph: one node per skeleton
voxel (`VanguardCenterlineDataset`, `node_mode="voxel"`), edges between 26-connected
voxels, per-voxel DCE kinetics + radius in `data.x`, and a global mean-pool readout
(`GCNClassifier`). This is a faithful, lossless encoding of the skeleton, but it has
structural problems for modeling:

- **Graphs are huge and mostly trivial.** The overwhelming majority of nodes have
  degree 2 — they carry no topological information, they are just points along a
  vessel. A tree with ~100 real branches can be tens of thousands of nodes.
- **Weak receptive field / over-smoothing.** A 2-layer GCN propagates information two
  voxels along a chain. To let two ends of a segment "see" each other the model needs
  as many layers as the segment is long; stacking that many `GCNConv` layers
  over-smooths. Segment-level structure is effectively invisible to the MVP model.
- **Size is a confound.** `num_nodes` scales with vessel/tumor extent. Mean-pool over
  voxels makes it easy to learn "big vessel tree ⇒ label" instead of morphology or
  kinetics. `graph_qc.csv` exists specifically because we already worried about this.
- **The morphometric unit is the segment, not the voxel.** The tabular / Deep Sets
  pipelines already model *segments* (length, tortuosity, curvature, radius, kinetics).
  The voxel graph is the odd one out.

## 2. Proposed structure — segment-as-edge

Contract the skeleton graph to its **topology**:

- **Nodes = junctions and endpoints**, i.e. every voxel with `degree != 2`
  (`degree >= 3` bifurcations + `degree == 1` tips). Degree-2 voxels are **dropped as
  nodes**.
- **Edges = segments.** Each maximal degree-2 chain between two kept nodes collapses to
  a single edge. The dropped voxels don't vanish — the polyline they traced is
  *summarized* onto the edge (§4).

This is the standard skeleton→topological-graph reduction, and we already have the
traversal for it: `extract_segments()` walks exactly these branch-to-branch polylines,
and `detect_bifurcations()` already characterizes the `degree >= 3` nodes. So the graph
reduction is not new code so much as a re-wiring of primitives we trust.

### 2.1 Degenerate topology — decide explicitly, fail loud otherwise

The voxel graph never had to care about these; the segment graph must. Per the repo's
fail-fast philosophy, each gets one committed policy, not a fallback chain:

| Case | Policy |
|---|---|
| **Pure cycle** (a loop of all-degree-2 voxels, no junction) | No natural node. Keep one deterministic representative voxel (min `(x,y,z)`) as a degree-2 node so the loop becomes a self-adjacent edge; record count in QC. |
| **Two junctions joined by ≥2 distinct segments** | Parallel edges ⇒ the graph is a **multigraph**. PyG `edge_index` already supports parallel edges, so keep both; do **not** dedupe (they are different vessels). |
| **Self-loop segment** (junction back to itself) | Keep as a self-loop edge with its summarized features. |
| **Isolated single voxel / two-voxel stub** | Endpoint node(s) with a zero-length edge or a lone node; count in QC, don't silently drop. |

These are surfaced in `graph_qc.csv` (new columns: `num_junctions`, `num_segments`,
`num_cycles`, `num_self_loops`, `num_parallel_edges`) so we can see how often they fire
before trusting a metric.

## 3. What we keep vs. summarize

The voxel graph put **everything on nodes** (`data.x`). The segment graph splits
information across three carriers:

- **Node features** — a junction/endpoint is a single voxel, so its features are the
  same per-voxel quantities we already compute: radius (distance transform), the
  voxel's own DCE kinetics, and topological `degree`. Bifurcation opening angles
  (`detect_bifurcations`) are a natural extra node feature.
- **Edge features (`edge_attr`)** — the summary of the segment's dropped voxels. This
  is the crux of the migration (§4).
- **Graph metadata** — unchanged (`case_id`, `dataset`, `site`, `y`, timepoints).

## 4. Summarizing a segment (the crux)

Two families of information live on the dropped degree-2 voxels: **geometry** and
**DCE kinetics**. Both already have summarizers we can reuse.

### 4.1 Geometry — reuse `compute_segment_metrics()`

`compute_segment_metrics(path, radius_map)` already returns, per segment:
`length`, `tortuosity`, `volume`, and mean/sd/median/quartile/min/max of both `radius`
and `curvature`. These map directly to `edge_attr` columns. Nothing new to derive — we
just stop throwing the result away.

### 4.2 Kinetics — new, but constrained by existing conventions

Per-voxel we currently derive `peak_time`, `peak_enhancement`, `time_to_enhancement`,
`washin_slope`, `auc_positive` (`_node_kinetic_features`). For a segment we need to
reduce the per-voxel curves along the polyline into fixed-width edge features. Two
candidate reductions:

- **(A) Summary-of-scalars (recommended):** compute the existing per-voxel kinetic
  scalars for every voxel on the segment, then take `mean` (and optionally `std`) along
  the segment. Cheap, interpretable, matches how `features/kinematic.py` already
  summarizes segments, and keeps `edge_attr` width small and fixed.
- **(B) Curve-first:** average the raw enhancement curves over the segment's voxels,
  then run `_node_kinetic_features` on the mean curve. Arguably more physical (one
  representative curve per segment) but couples every edge feature to one aggregation
  choice and is harder to reconcile with the per-voxel audit.

**Decision: (A)**, mean along the segment, with per-feature `std` as an opt-in column,
because it reuses the exact per-voxel kinetics we already validate in
`feature_summary/` and keeps the audit trail continuous. This is a summarization with
information loss — it must be called out in the module docstring and README per the
project's error-handling rules.

### 4.3 Information loss — stated plainly

Collapsing a segment to fixed edge features **loses within-segment heterogeneity**
(e.g. a curve that enhances at one end and not the other). We accept this: it's the
point of the migration, and the tabular pipeline already accepts the same loss. If a
future result suggests within-segment gradients matter, the mitigation is a small fixed
set of extra features (e.g. endpoint-vs-midpoint kinetic delta), not reverting to
voxels.

### 4.4 `time_to_enhancement` no-arrival sentinel

`time_to_enhancement` (and `seg_time_to_enhancement_mean/std`) is `NaN` for any
voxel/segment/edge with no detected arrival (peak enhancement ≤ 0). A raw `NaN`
can't enter the model, so when the feature is used it is replaced at finalize
time with `TTE_NO_ARRIVAL_SENTINEL = -1.0` (out-of-range in normalized `[0, 1]`
time), applied identically across voxel nodes, segment nodes, and junction
nodes/edges. This is a distinct, learnable "non-enhancing" value, **not**
imputation to a plausible time — `_sentinel_fill_tte` raises on any NaN outside
a TTE column, and the per-graph count is audited via `graph_qc.csv`'s
`tte_no_arrival_count`. See `AUDITING_RESULTS.md`.

## 5. Model changes

`GCNConv` **ignores edge features** — using it here would discard everything in §4.
Options:

- **Edge-aware conv (recommended):** swap `GCNConv` for `GINEConv` / `NNConv` /
  `GENConv`, which condition the message on `edge_attr`. Keep the rest of
  `GCNClassifier` (readout = global mean pool, linear head) and *all* of `train.py`
  (standardization, k-fold, BCE, metrics) unchanged. Standardization must be extended
  to normalize `edge_attr` as well as `x`.
- **Line-graph transform:** convert segment-as-edge → segment-as-node (§6, Option B)
  so segment features become node features and `GCNConv` works unmodified.

**Decision: edge-aware conv.** It keeps the topological graph honest (junctions are
real nodes with angles/degree) and is a small, contained model change.

## 6. The design fork — DECIDED: build all three, B first

**Decision (2026-07-09):** implement **all three** representations as coexisting
`node_mode`s on the same cohort so they can be compared head-to-head, and build them in
the order **B → A** (voxel already exists). B is first because it reuses the existing
model/standardization/QC stack with zero model-code risk; A is the topologically honest
target and comes second. The three modes, named by *what a node is*:

| `node_mode` | node = | status |
|---|---|---|
| `"voxel"` | one skeleton voxel | **done** (current pipeline) |
| `"segment"` | one whole segment (line graph, **B**) | **done** — `gnn/segment_graph.py`, wired through dataset + `build_dataset.py --node-mode` + `train.py` + `configs/gnn_segment*.yaml`; validated on the full 1491-case cohort |
| `"junction"` | a junction/endpoint voxel; segment = edge (**A**) | **done** — `gnn/junction_graph.py` + `EdgeGNNClassifier`, wired through dataset (`--node-mode junction --edge-features`) + `train.py` + `configs/gnn_junction*.yaml`; edge features standardized per-fold |

All three modes are now selectable end to end and ready for the matched
A-vs-B-vs-voxel experiment.

`"segment"` (B) matches the branch name `feature/segment-as-node`. All three must stay
selectable so a voxel-vs-segment-vs-junction comparison runs on identical folds.

Options A and B below are different graphs — the **line graph** of one is the other —
and they push complexity to different places:

| | **A. Segment-as-edge** (this doc's primary) | **B. Segment-as-node** (line graph) |
|---|---|---|
| Node = | junction / endpoint voxel | a whole segment |
| Edge = | a segment | "these two segments share a junction" |
| Rich segment features live on | `edge_attr` | `data.x` (node features) |
| Junction angles live on | node features (natural) | edge features / awkward |
| Model change | **new edge-aware conv** required | **none** — reuse `GCNConv` + mean-pool + all of `train.py` |
| Topological fidelity | high (junctions are first-class) | segments first-class, junctions implicit |
| Standardization/QC reuse | needs edge-feature path added | drops in almost unchanged |

Both ship (§6 decision). **A** is the topologically honest object — rich segment
summary as first-class `edge_attr`, junctions as real nodes with angles/degree — at the
cost of one new edge-aware conv layer. **B** answers "does segment-level
morphometry+kinetics beat voxel-level?" with **zero model-code risk**: it reuses
`GCNClassifier`, the per-feature standardizer, `feature_summary/`, and `graph_qc`
plumbing almost verbatim, with segment features simply becoming node features. We build
**B first** for exactly that reuse, then **A**.

## 7. Plumbing changes (both options)

- **`node_mode`:** the switch already exists as a stubbed extension point
  (`node_mode="segment"` currently raises `NotImplementedError`). Implement it there;
  keep `"voxel"` working so the two representations can be compared on the same cohort.
- **Cache manifest:** `_manifest_settings()` already records `node_mode` — a segment
  cache and a voxel cache won't be confused. Add the segment kinetics reduction choice
  (§4.2) and edge-feature list to the manifest so caches stay self-describing.
- **`edge_attr` in the cache:** `from_networkx` carries edge attributes through if we
  set them on the `nx` edges, so the collate/cache path needs little change beyond
  populating them.
- **QC:** extend `graph_qc.csv` with the topology counters (§2.1) and per-`edge_attr`
  min/max/mean/std, mirroring the existing per-node-feature QC. The size confound plots
  become *segment-count* vs `pcr`/`dataset` — same code, new column.
- **Feature summary:** add edge-feature histograms + NaN/inf report alongside the
  existing node-feature ones.

## 8. How B (`node_mode="segment"`, line graph) is built

The whole point of B is that segment features become **node** features, so the graph
flows through the *existing* `from_networkx` → `_finalize_data` → `data.x` →
`GCNClassifier` path unchanged. Concretely, per case:

1. **Voxel graph** as today: `segments_to_graph(edges_to_segments(mask_to_edges_bitmask(
   skeleton)))` + `obtain_radius_map(support, graph)`.
2. **Segments:** `extract_segments(voxel_graph)` → list of polylines, each running
   junction/endpoint → junction/endpoint. Each polyline becomes **one node** in the
   line graph.
3. **Node features per segment:** geometry from `compute_segment_metrics(path,
   radius_map)` (length, tortuosity, volume, radius & curvature stats) + kinetics
   summarized along the path per §4.2 (mean/std of the existing per-voxel scalars,
   sampling `dce_4d[:, z, y, x]` for each voxel in `path`).
4. **Line-graph adjacency:** map each junction/endpoint voxel → the segment indices
   touching it (a segment touches `path[0]` and `path[-1]`); connect every pair of
   segments that share a junction. No `edge_attr` for B — plain adjacency, so `GCNConv`
   is used verbatim. (Junction angle → edge feature is an A concern; see §6.)
5. **Degenerate topology** (§2.1): pure cycles produce no `extract_segments` output —
   apply the representative-voxel policy; count self-loops / parallel segments in QC.

This yields an `nx.Graph` whose nodes carry segment attributes; from there the build is
identical to voxel mode. The feature vocabulary (`_FEATURE_ATTR`) becomes **mode-aware**
— voxel mode keeps its per-voxel names, segment mode exposes the segment-level names
above — and the manifest records the kinetics-reduction choice (§4.2).

## 8b. Implementation phases

1. **B builder (new logic):** `gnn/segment_graph.py::build_segment_line_graph(...)`
   returning the line-graph `nx.Graph` with per-segment node attrs (steps 1–5 above).
   Unit-test on tiny hand-built skeletons (a Y, a cross, a loop) for correct node count
   and adjacency.
2. **Wire `node_mode="segment"`** into `_build_case` dispatch; make the feature
   vocabulary mode-aware; extend the cache manifest.
3. **QC + audit:** segment-count confound plots + segment-feature histograms / NaN
   report (reuse the existing writers with the new feature names).
4. **Run B vs voxel** on the same cohort/folds; compare AUC *and* size-confound plots.
5. **A (`node_mode="junction"`):** topological graph + `edge_attr` + edge-aware conv +
   `edge_attr` standardization.
6. **Three-way compare:** voxel vs segment vs junction on identical folds.

## 9. Open questions

- ~~Fork A vs B~~ — **resolved (§6):** build all three, B → A.
- Kinetics reduction: mean only, or mean+std (§4.2)? Default mean+std.
- ~~Bifurcation angles as junction node features~~ — **resolved:** implemented as
  `bifurcation_angle_{mean,min,max}` in `gnn.junction_graph` (§A only, still awkward in B).
  NaN for degree-1 endpoints (no bifurcation), sentinel-filled/audited via
  `gnn.data_loader.NO_BIFURCATION_SENTINEL`/`no_bifurcation_count`.
- Directionality: vessels have flow direction; do we keep the graph undirected (current)
  or attempt a root/flow orientation later? Defer — out of scope for this migration.

## 10. Matched representation experiment (voxel vs. segment vs. junction)

With all three `node_mode`s implemented, the point is a **matched** head-to-head:
identical cohort, identical folds, identical underlying signals, so any AUC
difference is the representation, not the inputs. Machinery (see
`experiments/graph_repr_compare_v1/README.md`):

- **Matched features (v1, 6 shared signals):** `peak_time`, `peak_enhancement`,
  `time_to_enhancement`, `washin_slope`, `auc_positive`, `radius` — on each
  representation's natural carrier: voxel/junction **nodes** carry the per-voxel
  names; segment **nodes** and junction **edges** carry the `seg_*_mean`
  summaries. Configs: `configs/gnn_{voxel,segment,junction}_compare.yaml`.
- **Matched folds:** `gnn/make_fold_assignments.py` freezes the evaluation
  framework's `StratifiedKFold(random_state=42)` split to a shared
  `pcr_labels_folded.csv`; all three arms consume it via `split_mode: predefined`
  (`split_col: fold`), so each case keeps the same fold even if an arm drops a
  case. Both build **and** train must point at that folded labels file (the cache
  manifest records `labels_path`).
- **Run it:** `gnn/slurm/submit_gnn_graph_compare_{build,train}.slurm`
  (`--array=0-2`, one arm each), then `analysis/gnn_graph_compare_plot.py` — which
  also **asserts** the three `split_manifest.csv` files agree per case (verifying
  the match rather than assuming it).
- **Extending to v2 (more features):** copy the three configs, add features
  (e.g. `degree`, segment geometry `seg_length/tortuosity/curvature`, `_std`
  summaries) to `gnn_node_features`/`gnn_edge_features`, point them at new
  `..._v2` cache dirs, set `CAMPAIGN=graph_repr_compare_v2`, and rerun the same
  build/train arrays + aggregator. The Slurm drivers read mode/features/paths
  from the configs, so **no script changes** are needed — only new configs.

## Summary of tradeoffs (concise)

- **Voxel-as-node (current):** lossless, trivial to build, `GCNConv` works directly —
  but huge graphs, tiny receptive field per layer, segment structure invisible, and
  graph size is a real confound.
- **Segment graph (proposed):** compact, topology explicit, segment = first-class
  object with reusable summarized geometry+kinetics, far less size-confounded, better
  inductive bias for vasculature — at the cost of within-segment information loss and,
  for the edge version, a new edge-aware conv.
- **Decision:** build the **segment graph**, as **segment-as-edge** (§2) with an
  **edge-aware conv** (§5), reusing `extract_segments` / `compute_segment_metrics` /
  `detect_bifurcations` for summarization (§4) and the existing `node_mode` extension
  point. Flagging the branch-name tension (§6): if the intent is segment-as-node
  (line graph) for maximum model reuse, that's a one-line redirect before I start.
