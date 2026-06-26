# Issue #120 — Deep Sets point-feature benchmark (ISPY2)

## Three core arms (validation AUC, 5-fold)

| Arm | AUC mean ± std | Status |
|-----|----------------|--------|
| Baseline (`curvature_rad` only) | 0.532 ± 0.029 | complete |
| Geometry + topology (16 features) | 0.514 ± 0.010 | complete |
| Geometry + topology + dynamic (27) | — | **pending** (Slurm rebuild submitted 2026-05-21) |
| Curvature + dynamic (12, secondary) | — | **pending** (Slurm rebuild submitted 2026-05-21) |

Source: `analysis/figures/deepsets_issue120/feature_set_benchmark_summary.csv` (refreshed via `scripts/refresh_issue120_benchmark_summary.py`).

## Did the features help? (interim, completed arms only)

1. **Did geom+topo beat baseline?** No. Geometry+topology (≈0.514) was slightly below curvature-only baseline (≈0.532).
2. **Did geom+topo+dynamic beat geom+topo?** **Cannot answer yet** — dynamic arm training not finished.
3. **Did dynamic beat baseline?** **Cannot answer yet** — same blocker.

With all non-dynamic ISPY2 arms run: geometry+topology+curvature reached ≈0.534 (marginally above baseline); geometry+topology without distance shells ≈0.529.

## Slurm reproduction (dynamic arms)

From repo root, use **absolute** `OUT_ROOT` (relative paths resolve under `slurm/`):

```bash
CONFIG=configs/deepsets_ispy2_pointfeat_geom_topo_dynamic.yaml \
OUT_ROOT=/path/to/repo/experiments/deepsets_ispy2_pointfeat_geom_topo_dynamic \
PARTITION=general FORCE_BUILD=1 FORCE_MERGE=1 FORCE_RETRAIN=1 \
bash slurm/submit_deepsets_pipeline.sh

CONFIG=configs/deepsets_ispy2_pointfeat_curvature_plus_dynamic.yaml \
OUT_ROOT=/path/to/repo/experiments/deepsets_ispy2_pointfeat_curvature_plus_dynamic \
PARTITION=general FORCE_BUILD=1 FORCE_MERGE=1 FORCE_RETRAIN=1 \
bash slurm/submit_deepsets_pipeline.sh
```

Profile expects **12 build shards** and **08:00:00** wall time for dynamic configs (`deepsets/deepsets_pipeline_profile.py`). Prior partial builds (7 and 3 manifest parts) were incomplete; FORCE_* triggers clean rebuild.

Jobs submitted this session: build arrays **869562** (geom_topo_dynamic), **869576** (curvature_plus_dynamic); merge/train depend on build completion.

After `train/**/metrics.json` exists, refresh artifacts:

```bash
python scripts/refresh_issue120_benchmark_summary.py
```

## 4D kinetic alignment (point features)

- Vessel 4D series: `_try_load_vessel_4d` in `build_deepsets_dataset.py` → `discover_study_timepoints` + `load_time_series_from_files` + `align_zyx_4d_to_shape` to skeleton ZYX.
- `vessel_segmentation_root` is set only on dynamic YAML configs (e.g. `configs/deepsets_ispy2_pointfeat_geom_topo_dynamic.yaml`).
- Load/alignment failures → zero dynamic columns and `kinetic_signal_ok=0`.
- Reference enhancement for relative ratios uses support/tumor heuristic (`_reference_enhancement_baseline`), not the full breast reference mask from tabular `features/kinematic.py`.
- Wash-in uses time-normalized slope from arrival (or t=0) to peak, not the issue toy’s peak−baseline definition.

Optional coverage audit on built `.pt` sets:

```bash
python deepsets/deepsets_kinetic_coverage.py \
  --out-root experiments/deepsets_ispy2_pointfeat_geom_topo_dynamic
```

## Caveats

- Same Deep Sets hyperparameters across arms (`configs/deepsets_ispy2_pointfeat_*.yaml`); only `deepsets_point_feature_set` and `vessel_segmentation_root` differ for dynamic arms.
- Default inclusion: `local_radius_with_fallback`.
- Incomplete kinetic coverage on some cases would attenuate dynamic signal; run coverage script after build.

## Final conclusions

**Update this section** after dynamic `metrics.json` files land and `feature_set_benchmark_summary.csv` includes both dynamic arms.
