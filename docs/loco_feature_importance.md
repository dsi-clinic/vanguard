# LOCO (Leave-One-Covariate-Out) Feature Importance

## Overview

LOCO is a **retrain-based** feature-importance method: for each covariate, a model is
retrained with that one covariate excluded, and its cross-validated performance is compared
against a "full" baseline model trained with every covariate. The performance drop when a
covariate is removed is its LOCO importance score.

This is different from **permutation importance**, which perturbs (shuffles/zeroes) a column
on an already-trained fixed model and re-evaluates, without retraining. LOCO answers "how much
does this covariate's *presence during training* help," which can differ from permutation
importance when covariates are correlated or a model can compensate for a missing feature by
leaning on correlated ones during fitting.

## Scope (v1)

- All three model families are covered: **GNN** (priority), DeepSets, and tabular.
- The GNN adapter is generic across all `gnn_node_mode` values (voxel/segment/junction), driven
  entirely by that mode's own feature vocabulary in `gnn/data_loader.py` -- no mode-specific code.
- Stability analysis (bootstrap resampling + Jaccard index over top-K important-covariate sets)
  is **explicitly out of scope for v1** -- see [Future Work](#future-work-stability-via-bootstrap--jaccard-index)
  below for the design sketch.

## Architecture: build the cache once with the full feature superset, select a subset at load time

Naively, LOCO for GNN/DeepSets would mean rebuilding the on-disk graph/tensor cache from
scratch for every left-out covariate (N+1 rebuilds) -- expensive, since the dominant build cost
(raw 4D DCE volume I/O for GNN; the gated dynamic-kinetics computation for DeepSets) is paid per
case regardless of which features are eventually used.

This is avoidable: **build the cache once with the full superset of candidate features, then do
LOCO by column-selecting a subset at load/train time.** Verified per family:

- **GNN**: graph topology (segment merging, junction detection, edge connectivity) is completely
  independent of which node/edge features are requested. The segment and junction builders
  (`gnn/segment_graph.py`, `gnn/junction_graph.py`) already compute their *full* feature
  vocabulary unconditionally today, regardless of what's requested -- so a superset build costs
  exactly the same as today's default build. Only `gnn/data_loader.py`'s cache-manifest check
  needed to change: `_check_cache_manifest` now allows `node_features`/`edge_features` to be any
  *subset* of what the cache was built with (every other setting -- roots, labels, node_mode --
  still requires exact equality), and `_load_processed`/`_reslice_for_requested_features`
  column-slices the loaded tensor down to the requested subset. No rebuild happens between LOCO
  runs; only the very first (superset) build pays the real cost.
- **DeepSets**: the same idea works, but needed one new named regime,
  `deepsets_point_feature_set: loco_superset` (`deepsets/build_dataset.py::DEEPSETS_FEATURE_LOCO_SUPERSET`),
  since none of the six existing regimes was already the union of all of them. Its build cost
  equals the priciest existing regime (`geometry_topology_dynamic`), since dynamic kinetic
  features are the one genuinely gated/expensive per-point computation. Column selection at load
  time is done by `deepsets.data.SavedSetLookup`'s `keep_features` parameter -- the single choke
  point every downstream consumer (splits, standardization, model `input_dim` inference,
  training) reads through, so one change makes the subset visible everywhere uniformly.
- **Tabular**: already free today -- `tabular.cohort.select_features`'s `explicit_model_columns`
  filters an already-built dataframe at train time; there's no cache/rebuild concept here at all.

## Per-family design

### Core (`evaluation/loco.py`)

Family-agnostic orchestrator:

- `LOCOAdapter` -- a `Protocol` each family implements: given `drop_covariate` (or `None` for the
  baseline), train + evaluate on the *same* folds as every other call in the sweep, and return a
  `KFoldResults`. Fold identity is guaranteed by never letting a per-covariate config override
  touch split-relevant `model_params` keys.
- `run_loco_sweep(covariates, adapter, base_config, outdir, family, node_mode=None, feature_group="node")`
  -- runs one baseline + `len(covariates)` leave-one-out training runs, builds the results
  tables, writes `loco_summary.csv`/`loco_fold_deltas.csv`/`loco_config_used.yaml`/`README.md`.
- `build_loco_tables` -- turns two `KFoldResults` (baseline, one covariate's run) into summary +
  fold-level delta rows for *every* metric in `evaluation.metrics.METRIC_REGISTRY` (not
  hardcoded to AUC).

### GNN (`evaluation/loco_gnn.py`)

- `superset_features_for_mode(node_mode)` / `superset_edge_features_for_mode(node_mode)` --
  reads `gnn.data_loader._MODE_FEATURE_ATTR` / `_MODE_EDGE_FEATURE_ATTR` for the mode's full
  vocabulary. Excludes `pcr_dummy` (the leakage-canary feature) from the superset -- it's never
  a real modeling feature.
- `ensure_gnn_superset_cache` -- a pre-flight *check*, not a builder: fails loudly with a
  concrete `gnn/build_dataset.py` command if the cache at `data_paths.gnn_cache_dir` wasn't
  built with the full superset. Not called automatically inside the sweep (the baseline run
  would hit the same failure first anyway); intended for a cheap check between a build job and a
  train array in a Slurm pipeline.
- `gnn_loco_adapter` -- the `LOCOAdapter` implementation: varies `model_params.gnn_node_features`
  (or `gnn_edge_features`, for `feature_group="edge"`, junction mode only) while pointing every
  run at the same superset cache dir.
- `run_gnn_loco(config, outdir)` -- driver. Defaults to sweeping *every* feature in the mode's
  vocabulary (GNN's per-mode vocabularies are small -- single digits to ~20 -- unlike tabular's
  ~800-column kinematic block, so this is tractable by default); `loco.covariates` restricts the
  set. Sweeps both node and edge features separately when the mode has edge features (junction),
  unless `loco.feature_group` pins one.

### DeepSets (`evaluation/loco_deepsets.py`)

- `ensure_deepsets_superset_manifest` -- pre-flight check that the manifest at
  `data_paths.deepsets_manifest_csv` was actually built with the `loco_superset` regime (cheap
  relative to the GNN equivalent, since it's a per-case point-set peek, not a full graph cache
  load -- called eagerly).
- `deepsets_loco_adapter` -- varies `keep_features` passed through
  `deepsets.train.run_deepsets_pipeline` -> `build_deepsets_dataset` -> `SavedSetLookup`.
- `run_deepsets_loco(config, outdir)` -- driver, defaults to sweeping all 28 `loco_superset`
  columns; `loco.covariates` restricts the set.

### Tabular (`evaluation/loco_tabular.py`)

Diverges intentionally from the GNN/DeepSets shape: instead of a per-covariate `LOCOAdapter`
callback, it generates N+1 named ablation arms up front (`build_tabular_loco_arms`) and
delegates the whole sweep to the existing `modeling.ablation.run_ablation_matrix` in one call
(single full-dataset build, per-arm `select_features(..., explicit_model_columns=...)`,
training, baseline deltas already computed there) -- then reshapes `ablation_summary.csv` /
`ablation_fold_auc.csv` into the shared LOCO table schema (`_melt_ablation_summary_to_loco`,
`_melt_ablation_fold_to_loco`).

Covariates must be scoped explicitly: `loco.covariates` (an explicit column list) or
`loco.covariates_from_block` (a single feature block, expanded and capped at
`loco.max_covariates`, default 50) -- tabular LOCO never silently expands to every column in
every selected block, since a single block (e.g. `kinematic`) can have ~800 columns.

`loco_fold_deltas.csv` only carries the `auc` metric for tabular -- a limitation of the reused
`ablation_fold_auc.csv` output, not something this module adds.

## Config schema

```yaml
loco:
  family: gnn                    # gnn | deepsets | tabular
  covariates: null                # explicit list; omit to sweep every candidate feature (gnn/deepsets)
  feature_group: null             # gnn only: "node" | "edge" | omit to sweep both (junction mode)

  # tabular only:
  selected_features: [vessel_kinematic]   # feature blocks eligible
  # covariates: [feat_a, feat_b, ...]     # REQUIRED (or covariates_from_block below)
  # covariates_from_block: vessel_kinematic
  # max_covariates: 50
  # baseline_arm_name: full_baseline
```

Run with:

```
python -m evaluation.loco --config <path-to-config-above> --outdir <output-dir>
```

## Results schema

`loco_summary.csv` -- one row per (covariate, metric):

| column | meaning |
|---|---|
| `family` | `tabular` \| `deepsets` \| `gnn` |
| `node_mode` | GNN only (`voxel`/`segment`/`junction`) |
| `feature_group` | `node` \| `edge` (GNN) or `column` (tabular) or `node` (deepsets) |
| `covariate` | the dropped feature/column name |
| `metric` | e.g. `auc`, `ap` |
| `baseline_mean` / `baseline_std` | full-model run |
| `loco_mean` / `loco_std` | this covariate's leave-one-out run |
| `delta_mean` | `loco_mean - baseline_mean`; **more negative = more important** |
| `n_splits` | fold count (sanity check both runs used the same n) |

`loco_fold_deltas.csv` mirrors this at per-fold granularity, for variance inspection (and as the
natural input to the bootstrap/Jaccard stability extension below).

Worked example row (a covariate whose removal cost 0.05 AUC):

```
family,node_mode,feature_group,covariate,metric,baseline_mean,baseline_std,loco_mean,loco_std,delta_mean,n_splits
gnn,segment,node,seg_radius_mean,auc,0.78,0.03,0.73,0.04,-0.05,5
```

## How to run

- **Tabular**: reuses `slurm/submit_ablation_arm_fold_array.slurm` / `modeling/run_arm_fold.py`
  unmodified -- only the generated `ablation_arms` config content changes.
- **GNN**: `gnn/slurm/submit_gnn_loco.slurm` (design only as of this writing) -- one build job
  for the superset cache, then a dependent array job (one task per covariate + baseline).
- **DeepSets**: `deepsets/slurm/submit_deepsets_loco.slurm` (design only as of this writing) --
  same shape: one sharded build job for the `loco_superset` manifest, then dependent train-only
  jobs.

**Actual `sbatch` submission of any of these requires separate explicit approval** per this
repo's cluster-safety rules -- this doc and the scripts it references only describe the design;
running a real (non-smoke) sweep is a deliberate, separately-approved action.

## Future Work: Stability via Bootstrap + Jaccard Index

**Not implemented in v1.** A point-estimate LOCO importance ranking (one `delta_mean` per
covariate from one train/val split) can be sensitive to which cases happen to land in which
fold. A natural follow-up:

1. Resample the cohort with replacement (bootstrap), `B` times (e.g. `B = 100`-`1000`).
2. For each bootstrap replicate, recompute folds (respecting the same `split_mode`/group
   constraints as the real run) over the resampled cohort, and rerun the full LOCO sweep
   (baseline + every covariate) against that replicate.
3. Rank covariates by `delta_mean` within each replicate (most negative = most important).
4. For each pair of replicates (or each replicate vs. the full-cohort ranking), take the
   top-K important covariates from each and compute the **Jaccard index** of the two sets:
   `|A ∩ B| / |A ∪ B|`.
5. Report the distribution of pairwise (or replicate-vs-full) Jaccard indices as a
   ranking-stability metric, alongside the point-estimate importance table -- not instead of it.
   A low Jaccard index means the "important" covariates are not robust to which cases happen to
   be in the training set, and the point-estimate ranking should be reported with that caveat.

**Cost caveat, and why this is deferred**: this multiplies total training-run count by `B`. Even
with the superset-cache trick eliminating the rebuild cost, each bootstrap replicate still pays
for `B × (N+1)` training runs (the cache build itself is still paid only once, since covariates
-- not cases -- determine the cache). For GNN in particular this could be a meaningful cluster
cost. A cheaper first pass (`B ≈ 20`-`30`, scoped to the single highest-priority
family/`node_mode` rather than the full matrix) is a reasonable v1.5 scope-in once v1's
point-estimate numbers exist to judge whether the extra cost is worth it.

## Verification / testing

- `tests/test_gnn_data_loader.py` -- the manifest-subset-select fix (node and edge features),
  including value-level parity checks against a fresh direct build, plus a regression test
  confirming a truly-uncached feature still fails loud even with `allow_manifest_mismatch=True`.
- `tests/test_deepsets_data.py` -- `SavedSetLookup.keep_features` (shape, order-faithful values,
  `feature_names` update, unknown-feature error).
- `tests/test_deepsets_point_features.py` -- `loco_superset` regime (28 unique columns, and an
  end-to-end `_build_case_set` check that the gated dynamic-kinetics path actually runs).
- `tests/test_loco_tabular.py` -- arm generation and the `ablation_summary.csv` /
  `ablation_fold_auc.csv` -> `loco_summary.csv` / `loco_fold_deltas.csv` reshaping logic.
- End-to-end smoke runs (not committed as pytest fixtures, given the heavier setup): a full
  `run_deepsets_loco` sweep against a tiny synthetic superset-regime manifest was run manually
  to confirm the whole pipeline -- splits, training, `keep_features` slicing, baseline
  comparison, results table, README -- works together, not just in isolation.
