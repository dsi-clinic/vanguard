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

## Results (fill after training)

| Config path | Mean validation AUC (or primary metric) | Notes |
|-------------|------------------------------------------|-------|
| `configs/deepsets_ispy2_pointfeat_baseline.yaml` | `0.5323` | From `experiments/deepsets_ispy2_pointfeat_baseline/train/deepsets_ispy2_pointfeat_baseline_20260421_125620/deepsets_ispy2_pointfeat_baseline/metrics.json` (`aggregated_metrics.auc.mean`). |
| `configs/deepsets_ispy2_pointfeat_geom_topo.yaml` | `0.5138` | From `experiments/deepsets_ispy2_pointfeat_geom_topo/train/deepsets_ispy2_pointfeat_geom_topo_20260421_130516/deepsets_ispy2_pointfeat_geom_topo/metrics.json` (`aggregated_metrics.auc.mean`). |
| `configs/deepsets_ispy2_pointfeat_geom_topo_dynamic.yaml` | _Artifact missing_ | `logs/deepsets-train-816756.err` shows the full 5-fold/40-epoch run completed, but the corresponding `metrics.json` was not present under `experiments/deepsets_ispy2_pointfeat_geom_topo_dynamic/train` at documentation time. |

Metrics are written under the training run directory (see
`evaluation/evaluator.py` `save_results` output for
`experiment_setup.name`).

## Interpretation (fill after training)

In the available on-disk metrics, geometry+topology underperformed the baseline
(`0.5138` vs `0.5323` mean validation AUC), so these added point features did
not improve this fixed-training setup. Dynamic-feature directionality is still
undetermined from artifacts currently present in the repository because the
dynamic-arm training logs exist but its `metrics.json` output was missing.

## Smoke build/merge path confirmation

- Slurm logs indicate successful build shards and merge for the fixed arm runs:
  - dynamic build shards: `logs/deepsets-build-816757-*.err` (all complete);
  - merge: `logs/deepsets-merge-816755.err` (empty error log; success path).
- Local smoke merge also succeeded on current artifacts:
  - `python merge_deepsets_manifest.py --output-dir /home/lunad/vanguard/experiments/deepsets_ispy2_pointfeat_geom_topo_dynamic`
  - output present: `experiments/deepsets_ispy2_pointfeat_geom_topo_dynamic/deepsets_manifest.csv`.
