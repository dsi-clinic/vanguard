# GNN modeling track

This package is the GNN counterpart to the tabular / Deep Sets pipelines: instead
of consuming *summarized* morphometry + kinematic features, it operates on the
**raw** vessel graph. The main parts are:

- **`gnn/data_loader.py`** — `VanguardCenterlineDataset`, an `InMemoryDataset`
  that builds one graph per case from saved centerline outputs.
- **`gnn/build_dataset.py`** — CLI wrapper to build/cache the full dataset on
  the cluster.
- **`gnn/model.py`** — `GCNClassifier`, the current MVP model.
- **`gnn/train.py`** — k-fold training script built on the shared
  `evaluation/` framework (standardization, `BCEWithLogitsLoss`, shared
  metrics/split/aggregation code).
- **`gnn/slurm/`** — Slurm submission scripts for both build and train.

## Data pipeline

### What it builds

- **Nodes** (`node_mode`, default `"voxel"`): one per skeleton voxel, keyed by
  `(x, y, z)`.
    - `"segment"` contracts each vessel segment to a single node (line graph);
      see `gnn/segment_graph.py` and `gnn/DESIGN_segment_graph.md`. It uses a
      different node-feature vocabulary (`seg_*`), so build it with matching
      `--node-features` (or omit them for the mode default).
    - `"junction"` keeps junction/endpoint voxels as nodes and each vessel
      segment as an **edge** carrying the segment summary as `edge_attr`
      (segment-as-edge, Option A; `gnn/junction_graph.py`). Node features are
      per-voxel signal + `degree`; edge features are the `seg_*` summary, passed
      via `--edge-features`. Trained with the edge-aware `EdgeGNNClassifier`
      (`gnn/model.py`) since `GCNConv` ignores edge features.
- **Edges:** between 26-connected skeleton voxels (8 corners, 12 *voxel* edges, 6 faces)
- **`data.pos`:** `(num_nodes, 3)` voxel coordinates `(x, y, z)`.
- **`data.x`:** node features stacked in the order given by `node_features`, see [Modeling Features](#modeling-features)

- **`data.y`:** the binary label (**required** — see [below](#labels-are-required)).
- **Metadata:** `data.case_id`, `data.dataset`, `data.site`, `data.num_timepoints`.


### Input tree

`root` is the centerline `studies/` tree. Each case directory contains:

```
<studies>/<dataset>/<case_id>/
  <case_id>_skeleton_4d_exam_mask.npy          # 3D (z,y,x) uint8 centerline
  <case_id>_skeleton_4d_exam_support_mask.npy  # 3D (z,y,x) uint8 vessel support
  run_summary.json                             # study_timepoints -> raw DCE phase indices
```

separately, under `dce_root`:

```
<dce_root>/<case_id>/<case_id>_NNNN.nii.gz   # one raw DCE-MRI phase per timepoint
```

The loader reads `run_summary.json["study_timepoints"]` (the integer `NNNN`
phase indices) and resolves each one to `<dce_root>/<case_id>/<case_id>_NNNN.nii.gz`
to load the raw 4D DCE signal (`gnn.raw_dce.discover_raw_dce_paths` /
`load_raw_dce_series`). `run_summary.json` and its `study_timepoints` key are
**required** — the loader raises immediately if either is absent or malformed,
and raises `FileNotFoundError` immediately if any expected raw DCE phase file
is missing.

### Labels are required

 Labels are loaded with
`tabular.cohort.load_labels(labels_path, id_column, label_column)`. **Cases with no matching label are dropped** (not built) with a loud `logging.warning` and recorded in
`dataset.dropped_case_ids` / `<cache_dir>/processed/dropped_cases.json`. If
the dropped fraction exceeds `max_missing_label_frac` (default `0.1`) the
whole build raises `RuntimeError` instead of silently training on a shrunken
cohort. When using the cache the manifest is
reloaded and the drop warning is re-logged so missing labels stay visible.

### Usage

Most users should build the dataset via Slurm (`gnn/build_dataset.py`, see
[Building the full dataset on the cluster](#building-the-full-dataset-on-the-cluster) below) rather than instantiating
the dataset directly. The CLI is a thin wrapper around this:

```python
VanguardCenterlineDataset(
    root="/path/to/centerlines/studies",
    labels_path="/path/to/labels.csv",     # required
    dce_root="/path/to/MAMA-MIA-syn60868042/images",  # required, raw DCE NIfTI tree
    id_column="case_id",
    label_column="pcr",
    node_features=("peak_time", "radius"),
    cache_dir="/path/to/gnn_cache",         # defaults to <root>/gnn_cache
    cases=["NACT_62", "NACT_63"],           # optional whitelist
    profile=True,                           # log per-stage timings
)
```
### Caching

The cache is written to `<cache_dir>/processed/data.pt`, plus a
per-case `<case_id>_graph.pt` for debugging. Constructing the dataset only
rebuilds when `<cache_dir>/processed/data.pt` is missing, so to force a
rebuild either delete `<cache_dir>/processed/` yourself, or pass
`--force-rebuild` to `gnn/build_dataset.py` (`FORCE_REBUILD=1` for the Slurm
job) to have it renamed aside to `processed_archive_<timestamp>/` first --
keeping the old cache (including its `feature_summary/`) around for the
record instead of overwriting it. `--force-rebuild` is incompatible with
`--no-cache` (which never persists a cache to archive in the first place).

Every fresh build also writes `<cache_dir>/processed/cache_manifest.json`,
recording everything that determines what the cached graphs actually
contain: `centerline_root`, `dce_root`, `labels_path`, `id_column`,
`label_column`, `cases` (whitelist, if any), `node_mode`, `node_features`,
`feature_source` (currently
`"raw_dce_protocol_baseline_physical_time_all_modes_v4"`), plus provenance-only fields not used
for comparison: `code_commit`, `num_graphs`, `label_counts`, `built_at`.

Every later load of that cache (constructing `VanguardCenterlineDataset`
against the same `cache_dir`, including via `gnn/train.py`) re-derives the
same settings from what's requested and compares them against the manifest.
A cache with no manifest (built before this check existed) or one whose
manifest doesn't match raises `RuntimeError` immediately, rather than
silently training on graphs built under a different root/label/feature
config than the one you asked for. Pass `allow_manifest_mismatch=True`
(`--allow-manifest-mismatch` / `ALLOW_MANIFEST_MISMATCH=1` for
`gnn/build_dataset.py`) to explicitly bypass this once you've confirmed a
mismatch is benign; otherwise rebuild the cache (`--force-rebuild`) so a
fresh, matching manifest is written.

### Modeling Features 
 - `peak_time` → elapsed time to peak enhancement divided by the full
    acquisition duration. The raw integer index is also kept on `data.peak_time`.
  - `peak_enhancement` → `max(enhancement)`.
  - `time_to_enhancement` → normalized arrival time, using
    `graph_extraction.feature_stats._arrival_index_from_enhancement` (first
    timepoint the curve reaches 20% of its own peak) -- the same helper
    `features/kinematic.py` uses. `NaN` for nodes with no detected arrival (no
    meaningful enhancement); the raw index (`-1` sentinel when undetected) is
    kept on `data.time_to_enhancement`. `NaN`s are caught per-feature by the
    `feature_summary/feature_na_report.json` audit rather than silently
    defaulted -- see `_write_feature_summary`.
  UFAST cases use the protocol-defined precontrast-frame mean, relative signal
  change, and physical acquisition seconds in voxel, segment, and junction
  modes. Legacy cases retain the single-frame absolute-enhancement convention.
  - `washin_slope` → `(enhancement[peak] − enhancement[arrival_or_0]) /
    (time[peak] − time[arrival_or_0])`.
  - `auc_positive` → `trapz(max(enhancement, 0))` over the study's time axis.
  - `radius` → local vessel radius from the support mask's distance transform.
  - `pcr_dummy` (opt-in only) → the graph's own `pcr` label, broadcast onto
    every node. A leakage-canary feature: it perfectly predicts `data.y` by
    construction, so it exists purely to sanity-check that the training
    pipeline (`data.x` → `GCNConv` stack → pooled logit → loss) can learn an
    end-to-end trivial signal. It's computed only when `"pcr_dummy"` is
    explicitly listed in `node_features` — never a hardcoded default, and
    never valid for real modeling results. `gnn/train.py` can optionally
    redraw this column at train time as class-conditional Gaussian noise
    instead of the clean 0/1 broadcast — see
    [pCR-as-Gaussian learnability sweep](#pcr-as-gaussian-learnability-sweep)
    below — without ever rebuilding this cache.
#### Feature summary

Every time a new cache is actually built (not on a cache hit, and not with
`no_cache=True`), the loader also writes
`<cache_dir>/processed/feature_summary/`:

- `<feature>_hist.png` — a histogram of that node feature over every node in
  every graph in the build (e.g. `peak_time_hist.png`, `radius_hist.png`).
- `feature_na_report.json` — per-feature `num_values` / `num_nan` / `num_inf`
  / `min` / `max` / `mean`.
- `README.md` — short auditing summary linking the histograms and embedding
  the NaN/inf report.

This is generated once per build (it's not cheap to recompute on every
training run) and is meant to catch upstream data issues (missing values,
degenerate ranges) before they reach modeling.

#### Graph QC summary

Every time a new cache is built (not on a cache hit, and not with
`no_cache=True`), the loader also writes `<cache_dir>/processed/graph_qc.csv`,
one row per graph. Graph size and feature ranges are possible confounders for
this GNN, not side details -- e.g. a model could be learning tumor size or a
site effect instead of enhancement kinetics -- so this file has what's needed
to plot `num_nodes` vs `pcr`, `num_nodes` vs `dataset`, prediction vs
`num_nodes` (join on `case_id` against `predictions.csv`), and feature
distributions by `dataset` or `pcr`, without re-deriving anything from the
cached graphs:

- `case_id`, `dataset`, `pcr`, `num_nodes`.
- `num_edges` -- matches the `data.num_edges` convention used in
  `split_manifest.csv` (edges are stored in both directions, so this is 2x
  the true undirected edge count).
- `num_connected_components` -- from the pre-`from_networkx` `nx.Graph`;
  disconnected skeleton branches show up as > 1.
- `mean_degree` -- `num_edges / num_nodes`, the correct average node degree
  under the doubled `num_edges` convention above.
- `missing_feature_count` / `nan_feature_count` -- total non-finite
  (NaN-or-inf) and NaN-only entries in `data.x` for that graph. These are
  normally `0`: `time_to_enhancement` (the only feature that can be "missing",
  i.e. no detected arrival) is sentinel-filled before `data.x` is finalized, so
  a non-zero value here signals an *unexpected* NaN/inf, not a missing arrival.
- `tte_no_arrival_count` -- number of no-arrival cells that were replaced with
  `TTE_NO_ARRIVAL_SENTINEL` (`-1.0`, normalized space) across `data.x` and, in
  junction mode, `data.edge_attr`. A raw NaN cannot enter the model, so
  no-arrival time-to-enhancement is encoded as a distinct out-of-range sentinel
  (a learnable "non-enhancing" value) rather than imputed to a plausible time;
  this column keeps that fill audited per graph. See `AUDITING_RESULTS.md`.
- `<feature>_min` / `_max` / `_mean` / `_std` for every name in
  `node_features`, computed over that graph's nodes only (NaN-aware). For a
  `time_to_enhancement` feature these now include the `-1.0` sentinel.

The same build also renders the confound plots directly, into
`<cache_dir>/processed/graph_qc_plots/` (`gnn/graph_qc_plots.py`, called from
`_write_graph_qc`):

- `num_nodes_vs_pcr.png` / `num_nodes_vs_dataset.png` -- jittered scatter of
  graph size against the other categorical variable, colored by
  dataset/pcr respectively; legend and title state the exact `n` (graph
  count) behind each category.
- `feature_distributions_by_dataset.png` / `feature_distributions_by_pcr.png`
  -- node-level density histograms (unfilled step outlines, independently
  normalized per group) for every `node_features` entry; legend states both
  the graph count and node count behind each group, since these pool nodes
  across graphs and a "distribution" can otherwise silently mean n=1 graph.

`prediction_vs_num_nodes.png` is **not** written at build time -- it needs a
trained model's predictions, which don't exist yet. `gnn/train.py` writes it
after every run instead, into both that run's own output directory (alongside
`predictions.csv`) and back into this same `<cache_dir>/processed/graph_qc_plots/`
directory, so the cache's plot folder always has all 5 plots, with the
prediction one reflecting whichever training run against that cache is most
recent (it's overwritten each run, not versioned per-run -- check the run's
own `<outdir>/<experiment_setup.name>/prediction_vs_num_nodes.png` copy if you
need the one from a specific past run).

To regenerate these plots by hand (e.g. against an older cache, or a
`predictions.csv` from a run that hasn't been retrained), use
`gnn/plot_graph_qc.py`:

```bash
python -m gnn.plot_graph_qc --cache-dir <cache_dir> --out-dir <dir> \
    [--predictions-csv path/to/predictions.csv]
```

### Building the full dataset on the cluster


```bash
sbatch gnn/slurm/submit_gnn_build.slurm
```

Override any path via environment variables (`ROOT`, `DCE_ROOT`, `LABELS_PATH`,
`CACHE_DIR`, `ID_COLUMN`, `LABEL_COLUMN`, `NODE_FEATURES`, `CASES`,
`NO_CACHE=1`) -- see the script header for usage. `CASES` (comma-separated
case IDs) is handy for a smoke-test submission before committing to the full
~1500-case build. 

#### Parallel build (`num_workers`)

Because each case's graph is built from its own files only, it is possible to fan out
across a process pool via `--num-workers` / `NUM_WORKERS` (or the
`GNN_BUILD_WORKERS` env var read by `build_dataset.py`'s CLI default).

### Profiling

With `profile=True` the loader accumulates wall time for each build stage
(`mask_load`, `graph_build`, `timeseries_load`, `peak_time`, `from_networkx`) and
logs mean / median / max across cases. The **4D `timeseries_load`** stage is the
runtime watch item as we move from the small MAMA-MIA cohort to UChicago, where
each case has many timepoints.

> Record the observed per-stage numbers here after the first real-data run on the
> cluster (login-node smoke set of 2–3 NACT cases, then the full cohort via
> Slurm).

## Model + training

`gnn/model.py` provides `GCNClassifier`: a stack of `GCNConv` layers, a
configurable graph readout, and a linear head producing one logit per graph.
The readout is set by `model_params.gnn_pooling` (Tier 1, decision D2.1):
`"mean"` (default, the original single `global_mean_pool` -- backward
compatible), `"mean_max"`, or `"mean_max_sum"`, which concatenate global
max/add pools so the graph embedding keeps the distributional tails mean-pool
discards (`gnn/model.py` `POOLING_WIDTHS` / `_graph_readout`). Junction mode
(`EdgeGNNClassifier`) still mean-pools only. Attention readout and stronger
conv operators are the next Tier-1 steps (see `gnn/PLAN_advanced_modeling.md`).

Outline of `gnn/train.py`:

1. Load the cached `VanguardCenterlineDataset` (built ahead of time by
   `gnn/build_dataset.py`).
2. Build a one-row-per-graph cohort table (`build_graph_cohort`): `case_id`,
   `y`, `dataset`, `site`, `graph_index` (the graph's position in the
   dataset), plus a fold column (default name `"fold"`, from
   `model_params.split_col`) merged in from the labels file *if* it has one.
3. Hand that table to `evaluation.build_splits.create_splits_for_dataframe`,
   which returns an `Evaluator` and a list of `FoldSplit`s -- standard
   stratified/group k-fold by default (`model_params.n_splits`,
   `use_group_split`, `group_col`, `stratum_col`), or, with
   `model_params.split_mode: "predefined"`, leave-one-fold-out CV over the
   fold column above. Predefined-fold support lives centrally in
   `evaluation/kfold.py` (`create_predefined_splits`), so any model family can
   opt into the same fold definition.
4. Train a fresh `GCNClassifier` per fold (`fit_predict_one_fold`):
   node-feature standardization fit on that fold's train split only, a
   config-selected loss (`build_loss_fn`), metrics via
   `evaluation.metrics.compute_binary_metrics`.
5. Aggregate fold predictions with `evaluator.aggregate_kfold_results` and
   save with `evaluator.save_results` -- the same `metrics.json` /
   `predictions.csv` / ROC-PR plots convention every other model family uses.

### Training knobs (Tier 0)

`fit_predict_one_fold` honors the same `model_params` training knobs as
`deepsets/train.py` (their implementations are deliberately parallel; see
`gnn/PLAN_advanced_modeling.md`, Tier 0). All default to reproducing the
historical plain-Adam / final-epoch behavior *except* the two shared config
defaults noted below, so **`configs/gnn.yaml` pins them explicitly** to stay a
faithful baseline reference (decision D0.6):

- `loss` -- `weighted_bce` (default; `pos_weight = n_neg/n_pos` from the fold's
  train split, up-weighting the minority pCR class), `unweighted_bce` (the
  historical plain `BCEWithLogitsLoss`), or `focal` (`focal_alpha`/`focal_gamma`).
  A fold with zero training positives falls back to `pos_weight=1.0` and logs
  its class balance -- a degenerate split, not a silent default.
- `weight_decay` -- Adam L2 penalty (shared default `1e-4`).
- `max_grad_norm` -- global grad-norm clip before each step (`0.0` disables).
- `lr_scheduler` -- `none` (default), `cosine` (`CosineAnnealingLR`), or
  `plateau` (`ReduceLROnPlateau` on val loss, with `lr_scheduler_factor` /
  `lr_scheduler_patience`).
- `early_stopping_patience` -- stop after N epochs with no val-loss improvement
  (`0` disables). Raise `epochs` so patience, not the ceiling, ends most folds.
- `restore_best_epoch` -- report the best-**val-loss** epoch's weights instead of
  the final epoch's. (Val loss, not val AUC, matches `deepsets/train.py` and is
  smoother on these small imbalanced folds.) `loss_history.csv` gains an `lr`
  column so the schedule is auditable.

The Tier-0 arm that turns these on lives in `configs/gnn_tier0.yaml` (identical
to `gnn.yaml` except the training knobs, so a paired ΔAUC is attributable to
optimization, not representation).

Config-driven, like `tabular/train.py` and `deepsets/train.py`:

```bash
python -m gnn.train --config configs/gnn_smoke.yaml
```

See `configs/gnn.yaml` (full cohort) and `configs/gnn_smoke.yaml` (8-case
smoke test) for the `data_paths.gnn_*` / `model_params.gnn_node_features`
schema. Every run writes `config_used.yaml` + `config_used.json` (the
resolved config, for programmatic loading) directly under
`experiments/<name>_<timestamp>/` (or `--outdir`), plus the evaluator's
`<outdir>/<experiment_setup.name>/{split_manifest.csv,predictions.csv,metrics.json,plots/}`.
`gnn/train.py` additionally writes `loss_history.csv`, `loss_by_epoch.png`,
and `auc_by_epoch.png` (train/val curves per fold) alongside it.
`gnn/slurm/submit_gnn_train.slurm` mirrors `submit_gnn_build.slurm` and
defaults to `configs/gnn_smoke.yaml`.

### pCR-as-Gaussian learnability sweep

Extends the `gnn_pcr_dummy_check` leakage-canary experiment (perfect 0/1
signal, AUC=1.0 by construction) into a graded-difficulty sweep: instead of a
clean binary broadcast, `pcr_dummy` is redrawn at train time as
class-conditional Gaussian noise — label 0 → `N(gnn_pcr_dummy_class0_mean,
sigma^2)`, label 1 → `N(gnn_pcr_dummy_class1_mean, sigma^2)` — so val AUC can
be traced from ~1.0 (clean signal) down to ~0.5 (pure noise) as a function of
signal-to-noise ratio. Implemented in `gnn.train._apply_pcr_dummy_noise`,
called from `fit_predict_one_fold` **after** cloning graphs out of the cached
dataset and **before** node standardization — the noise is a train-time
transform layered on top of the existing cache, never a reason to rebuild it
(see `configs/gnn_pcr_gaussian.yaml`, which points at the same
`gnn_cache_pcr_dummy` cache the original dummy-check run built).

Key facts:

- Defaults (`class0_mean=0.0`, `class1_mean=1.0`, `noise_std=0.0`) exactly
  reproduce the original deterministic 0/1 broadcast — this is a no-op unless
  at least one is explicitly overridden, and raises if overridden while
  `"pcr_dummy"` isn't in `gnn_node_features` (no silent no-op).
- `gnn/train.py` standardizes every node feature per-fold as
  `(x - train_mean) / train_std`, which exactly cancels a constant offset —
  so **only the ratio of class separation to noise_std matters** (a
  Cohen's-d / SNR quantity), not the raw magnitude of either mean.
- Noise is drawn once per **graph** (not per node), broadcast across all of
  that graph's nodes — matching how the clean dummy is already broadcast —
  and keyed by `graph_index` (the graph's position in the cached dataset) so
  the same case gets the same draw across folds/seeds/reruns that share
  `gnn_pcr_dummy_noise_seed`.

Sweep a grid without regenerating YAML files, via CLI overrides:

```bash
python -m gnn.train --config configs/gnn_pcr_gaussian.yaml \
    --pcr-dummy-class1-mean 1.0 --pcr-dummy-noise-std 1.0 --pcr-dummy-noise-seed 0 \
    --outdir experiments/gnn_pcr_gaussian_grid/d1.0_seed0
```

### Auditing outputs

Every run writes enough to inspect what happened without reading the code:

- **`split_manifest.csv`** (`evaluation.build_splits.build_split_manifest`,
  the same central split-building module `create_splits_for_dataframe` lives
  in, so other model families can reuse it): one row per `(case, fold)` --
  `case_id`, `graph_index`, `dataset`, `site`, `pcr`, `fold`, `train_or_val`,
  `num_nodes`, `num_edges`. Each case appears once per fold (`"val"` in the
  fold it's held out on, `"train"` in every other fold), so you can check
  whether folds are balanced by site/dataset/graph size before trusting a
  metric.
- **`predictions.csv`**: the evaluator's standard `case_id`, `y_true`,
  `y_pred`, `y_prob`, `fold` columns, plus `dataset`, `site`, and `pcr`
  (identical to `y_true`, named after the label itself) merged in from the
  cohort table -- one row per validation case. Cross-referencing `y_prob`
  against `dataset`/`site`/`num_nodes` (via `split_manifest.csv`) is how you
  tell whether the model is learning biology, a site effect, or graph size.
- **`prediction_vs_num_nodes.png`**: that same size-vs-prediction check,
  plotted directly (see "Graph QC summary" above) -- also copied back into
  the cache's `processed/graph_qc_plots/`.
- **`metrics.json`** / **`metrics_per_fold.json`**: aggregated + per-fold
  metrics, written by `evaluator.save_results`.
- **`config_used.yaml`** / **`config_used.json`**: the exact resolved config
  (including defaults) for the run, written by `load_cohort.write_config_snapshot`.
