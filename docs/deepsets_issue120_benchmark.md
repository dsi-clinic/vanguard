# Deep Sets point-feature benchmark (issue #120)

## Training hyperparameters

Use the same `model_params` block in all three configs (already matched in the
pinned YAMLs under `configs/deepsets_ispy2_pointfeat_*.yaml`): batch size, epochs,
hidden width, layers, dropout, learning rate, weight decay, pooling, and
tumor-local radius settings.

## Building datasets

For each arm, point `CONFIG` at the matching YAML and use a **separate**
`OUT_ROOT` so manifests and `.pt` sets never mix feature regimes.

| Arm | Config path | Suggested `OUT_ROOT` |
|-----|-------------|----------------------|
| Baseline (curvature only) | `configs/deepsets_ispy2_pointfeat_baseline.yaml` | `experiments/deepsets_ispy2_pointfeat_baseline` |
| Geometry + topology | `configs/deepsets_ispy2_pointfeat_geom_topo.yaml` | `experiments/deepsets_ispy2_pointfeat_geom_topo` |
| Geometry + topology + dynamic | `configs/deepsets_ispy2_pointfeat_geom_topo_dynamic.yaml` | `experiments/deepsets_ispy2_pointfeat_geom_topo_dynamic` |

Run `build_deepsets_dataset.py` (and `merge_deepsets_manifest.py` if sharded),
then inject `data_paths.deepsets_manifest_csv` into a runtime YAML the same way
`slurm/submit_deepsets_pipeline.sh` does, and run `train_deepsets.py`.

Optional: `scripts/benchmark_deepsets_feature_sets.sh` submits or documents the
three `(CONFIG, OUT_ROOT)` pairs in one loop.

## Point features per config

Column order is defined by `deepsets_point_feature_names()` in
[`build_deepsets_dataset.py`](../build_deepsets_dataset.py) and is serialized into
each case `.pt` as `feature_names` (same order as columns of `x`).

### `configs/deepsets_ispy2_pointfeat_baseline.yaml`

`model_params.deepsets_point_feature_set: baseline` — **1** feature:

| # | Name | Description |
|---|------|-------------|
| 1 | `curvature_rad` | Local skeleton curvature (radians). |

### `configs/deepsets_ispy2_pointfeat_geom_topo.yaml`

`model_params.deepsets_point_feature_set: geometry_topology` — **16** features:

| # | Name | Description |
|---|------|-------------|
| 1 | `signed_distance_mm` | Signed distance to tumor boundary (mm); negative inside tumor. |
| 2 | `abs_signed_distance_mm` | Absolute distance to tumor boundary (mm). |
| 3 | `inside_tumor` | Binary: point inside tumor mask. |
| 4 | `shell_0_2mm` | One-hot: outside tumor, 0–2 mm shell. |
| 5 | `shell_2_5mm` | One-hot: 2–5 mm shell. |
| 6 | `shell_5_10mm` | One-hot: 5–10 mm shell. |
| 7 | `shell_ge_10mm` | One-hot: ≥10 mm outside (coarse outer bin). |
| 8 | `degree` | Graph degree on the skeleton adjacency used in the builder. |
| 9 | `is_endpoint` | Binary: degree 1. |
| 10 | `is_chain` | Binary: degree 2. |
| 11 | `is_bifurcation` | Binary: degree ≥ bifurcation threshold. |
| 12 | `offset_x_mm` | Offset from tumor centroid along x (mm). |
| 13 | `offset_y_mm` | Offset from tumor centroid along y (mm). |
| 14 | `offset_z_mm` | Offset from tumor centroid along z (mm). |
| 15 | `support_radius_mm` | Local vessel “support” radius from EDT on support mask (mm), or zero if unavailable. |
| 16 | `support_radius_available` | Binary: support mask present and shape-matched for EDT. |

### `configs/deepsets_ispy2_pointfeat_geom_topo_dynamic.yaml`

`model_params.deepsets_point_feature_set: geometry_topology_dynamic` — **27**
features: the **16** geometry/topology features above, then **11** dynamic
(kinetic) features sampled from the aligned vessel 4D time series at each
skeleton voxel:

| # | Name | Description |
|---|------|-------------|
| 17 | `arrival_index_norm` | Normalized index of first enhancement above baseline (0 if none). |
| 18 | `has_arrival` | Binary: a clear arrival time was detected. |
| 19 | `peak_index_norm` | Normalized time index of peak enhancement. |
| 20 | `peak_enhancement` | Peak signal change vs pre-contrast at the voxel. |
| 21 | `washin_slope` | Simple wash-in slope from arrival to peak. |
| 22 | `washout_slope` | Simple wash-out slope from peak to last timepoint. |
| 23 | `positive_enhancement_auc` | Area under positive enhancement vs time (trapezoid). |
| 24 | `peak_rel_reference` | Peak enhancement relative to a reference tissue curve. |
| 25 | `auc_rel_reference` | Positive AUC relative to reference. |
| 26 | `kinetic_signal_ok` | Binary: enough finite timepoints and valid 4D for dynamics. |
| 27 | `reference_ok` | Binary: reference curve for ratios was usable. |

## Results

| Config path | Mean validation AUC (or primary metric) | Notes |
|-------------|------------------------------------------|-------|
| `configs/deepsets_ispy2_pointfeat_baseline.yaml` | `0.5323` | From `experiments/deepsets_ispy2_pointfeat_baseline/train/deepsets_ispy2_pointfeat_baseline_20260421_125620/deepsets_ispy2_pointfeat_baseline/metrics.json` (`aggregated_metrics.auc.mean`; std `0.0294`; `n_features=1`). |
| `configs/deepsets_ispy2_pointfeat_geom_topo.yaml` | `0.5138` | From `experiments/deepsets_ispy2_pointfeat_geom_topo/train/deepsets_ispy2_pointfeat_geom_topo_20260421_130516/deepsets_ispy2_pointfeat_geom_topo/metrics.json` (`aggregated_metrics.auc.mean`; std `0.0096`; `n_features=16`). |
| `configs/deepsets_ispy2_pointfeat_geom_topo_dynamic.yaml` | `0.5277` | Recovered by rerunning training against the existing dynamic manifest with Slurm job `840037`: `experiments/deepsets_ispy2_pointfeat_geom_topo_dynamic_existing_manifest_train/deepsets_ispy2_pointfeat_geom_topo_dynamic_20260504_184523/deepsets_ispy2_pointfeat_geom_topo_dynamic/metrics.json` (`aggregated_metrics.auc.mean`; std `0.0248`; `n_features=27`). |

Metrics are written under the training run directory (see
`evaluation/evaluator.py` `save_results` output for
`experiment_setup.name`).

![Deep Sets feature-set validation AUC](deepsets_issue120_figures/feature_set_auc_comparison.png)

The chart source table is
`docs/deepsets_issue120_figures/feature_set_benchmark_summary.csv`.

## Interpretation

In this fixed-training comparison, geometry+topology underperformed the baseline
(`0.5138` vs `0.5323` mean validation AUC), so these added point features did
not improve the baseline by themselves. Adding dynamic features recovered most
of that drop (`0.5277` mean validation AUC) and improved over geometry+topology,
but it still did not beat the curvature-only baseline in this run.

## Smoke build/merge path confirmation

- Slurm logs indicate successful build shards and merge for the original fixed
  arm runs:
  - dynamic build shards: `logs/deepsets-build-816757-*.err` (all complete);
  - merge: `logs/deepsets-merge-816755.err` (empty error log; success path).
- Local smoke merge also succeeded on current artifacts:
  - `python merge_deepsets_manifest.py --output-dir /home/lunad/vanguard/experiments/deepsets_ispy2_pointfeat_geom_topo_dynamic`
  - output present: `experiments/deepsets_ispy2_pointfeat_geom_topo_dynamic/deepsets_manifest.csv`.
- The dynamic training metric above was recovered without rebuilding the full
  dataset by submitting train-only Slurm job `840037` against
  `/home/lunad/vanguard/experiments/deepsets_ispy2_pointfeat_geom_topo_dynamic/deepsets_runtime_config.yaml`.
- A payload contract spot check on
  `experiments/deepsets_ispy2_pointfeat_geom_topo_dynamic/sets/ISPY2_100899.pt`
  confirmed `x.shape == (846, 27)`, `len(feature_names) == 27`,
  `point_feature_set == "geometry_topology_dynamic"`, and
  `kinetic_timepoint_count == 6`.
- A full dynamic rebuild was also started in
  `experiments/deepsets_ispy2_pointfeat_geom_topo_dynamic_issue120_rerun_20260504`.
  Initial Slurm build array `839719` completed six shards but shards 0 and 1 hit
  the 2-hour build limit; retry array `839943` resubmitted only those two shards
  with a 4-hour limit. After train-only job `840037` recovered the required
  metric, the superseded retry chain was stopped: retry shard 1 completed,
  retry shard 0 was canceled while still running, and dependent merge `839944`
  and train `839945` were canceled before starting.
