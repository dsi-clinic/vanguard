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
| `configs/deepsets_ispy2_pointfeat_baseline.yaml` | _TBD_ | _TBD_ |
| `configs/deepsets_ispy2_pointfeat_geom_topo.yaml` | _TBD_ | _TBD_ |
| `configs/deepsets_ispy2_pointfeat_geom_topo_dynamic.yaml` | _TBD_ | _TBD_ |

Metrics are written under the training run directory (see
`evaluation/evaluator.py` `save_results` output for
`experiment_setup.name`).

## Interpretation (fill after training)

_Short note on whether geometry/topology and dynamic features helped versus the
baseline, based on the table above._
