# Auditing Results — documented data-handling decisions

Per the repo's fail-fast error-handling philosophy, the pipeline avoids silent
fallbacks/imputation. Where a genuine, known data edge case is nonetheless
handled deliberately, it is recorded here (what was handled, why, and what
failure mode it protects against), and flagged at the point of use.

## GNN — time-to-enhancement "no arrival" sentinel

- **What is handled.** In the GNN centerline pipeline, the node/edge feature
  `time_to_enhancement` (and its segment summaries `seg_time_to_enhancement_mean`
  / `seg_time_to_enhancement_std`) is `NaN` for any voxel / segment / edge with
  no detected contrast arrival (peak enhancement ≤ 0, i.e. non-enhancing
  tissue). When such a feature is requested, `gnn/data_loader.py::_finalize_data`
  (via `_sentinel_fill_tte`) replaces that `NaN` with a fixed sentinel
  `TTE_NO_ARRIVAL_SENTINEL = -1.0` in the normalized `[0, 1]` time space, applied
  identically to voxel nodes, segment nodes, and junction nodes/edges.
- **Why.** A raw `NaN` in `data.x` / `data.edge_attr` propagates to a `NaN`
  training loss, so `time_to_enhancement` cannot be used as a feature without a
  policy. The sentinel is *not* imputation to a plausible value: `-1.0` is
  out-of-range for the normalized arrival time, so "no detectable arrival"
  becomes a distinct, learnable value (non-enhancement may itself be prognostic)
  rather than being blended into the mean. This was chosen over mean-fill (which
  erases the no-arrival signal) with the user for the matched
  voxel-vs-segment-vs-junction comparison (`gnn/DESIGN_segment_graph.md`,
  `experiments/graph_repr_compare_v1/`).
- **Failure mode it protects against.** Silent corruption of results from a
  `NaN` loss, or from imputing a fake arrival time. The fill is bounded and
  audited: `_sentinel_fill_tte` **raises** if any `NaN` remains in a non-TTE
  column (so an unexpected NaN surfaces loudly instead of being filled), and the
  per-graph no-arrival count is written to `graph_qc.csv`'s
  `tte_no_arrival_count` (so the fill is never silent). The features that may
  legitimately be `NaN` are pinned in `gnn/data_loader.py::_TTE_FEATURE_NAMES`.
- **Where.** `gnn/data_loader.py` (`TTE_NO_ARRIVAL_SENTINEL`,
  `_TTE_FEATURE_NAMES`, `_sentinel_fill_tte`, `_finalize_data`,
  `_write_graph_qc`); documented in `gnn/README.md` (Graph QC summary) and
  `gnn/DESIGN_segment_graph.md`; tests in `tests/test_gnn_data_loader.py`.
