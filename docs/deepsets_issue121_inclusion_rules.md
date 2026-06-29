# Deep Sets inclusion-rule comparison (issue #121)

## Goal

Compare a small set of tumor-local point-inclusion rules and choose one default
for Deep Sets modeling. Selection is based on **dataset-build** statistics
(coverage, fallback rate, point counts), not validation AUC.

## Rule definitions

Signed distance (mm) is negative inside the tumor mask and positive outside.

| Rule | Primary selection | Fallback |
|------|-------------------|----------|
| `local_radius_with_fallback` | signed distance ≤ `local_radius_mm` (from floor/scale/cap + `tumor_equiv_radius_mm`) | nearest 64 if empty |
| `fixed_radius_30mm_with_fallback` | signed distance ≤ 30 mm | nearest 64 if empty |
| `fixed_radius_50mm_with_fallback` | signed distance ≤ 50 mm | nearest 64 if empty |
| `peritumoral_shells_with_fallback` | inside tumor or outside with signed distance < 5 mm | nearest 64 if empty |
| `local_radius_only` | same cutoff as default primary filter | none (case skipped if empty) |
| `nearest_64_only` | always 64 closest points by signed distance | n/a |

Only `deepsets_inclusion_rule` controls which rule is written into each case
`.pt` file. `deepsets_compare_inclusion_rules` evaluates extra rules in the same
pass for `inclusion_rule_summary.csv` metrics only.

## Cohort summary (ISPY2, pinned)

Built with `configs/deepsets_ispy2.yaml` and
`OUT_ROOT=experiments/issue121_inclusion_compare` (980-case cohort). Hyperparameters:
`deepsets_local_radius_floor_mm=50`, `scale=2`, `cap_mm=60`.

| Rule | cases_written | cases_skipped | fallback_fraction | num_points_median | num_points_range |
|------|---------------|---------------|-------------------|-------------------|------------------|
| `local_radius_with_fallback` | 980 | 0 | 0.045 | 517.5 | 1-3537 |
| `fixed_radius_30mm_with_fallback` | 980 | 0 | 0.101 | 253.5 | 1-2250 |
| `fixed_radius_50mm_with_fallback` | 980 | 0 | 0.045 | 517.5 | 1-3537 |
| `peritumoral_shells_with_fallback` | 980 | 0 | 0.392 | 64.0 | 1-727 |
| `local_radius_only` | 936 | 44 | 0.0 | 547.5 | 1-3537 |
| `nearest_64_only` | 980 | 0 | 0.0 | 64.0 | 13-64 |

`fixed_radius_50mm_with_fallback` matches the default on this cohort because
`local_radius_mm` is capped at 60 mm and is often ≤ 50 mm for these tumors, so
the effective cutoff coincides with a 50 mm fixed rule for most cases.

## Default choice

**Production default:** `local_radius_with_fallback` (set in `config.py` and
`configs/deepsets_ispy2.yaml`).

**Rationale:**

- **Coverage:** 0 skipped cases vs 44 for `local_radius_only` (fallback rescues
  empty strict sets without abandoning tumor-relative locality).
- **Fallback load:** ~4.5% fallback fraction — primary cutoff usually suffices.
- **Locality:** tumor-relative radius adapts to tumor size; avoids a single
  fixed mm cutoff for all cases.
- **Rejected alternatives:**
  - `fixed_radius_30mm_with_fallback` — higher fallback (10%) and smaller
    median sets (253.5 points).
  - `peritumoral_shells_with_fallback` — high fallback (39%) and median 64
    points (often the fallback cap).
  - `nearest_64_only` — always small, fixed-size sets; not tumor-local in the
    intended sense.
  - `local_radius_only` — drops 44 cases; too brittle for routine pipelines.

Do not judge a rule only by higher `num_points_median`; prefer stable coverage,
low fallback, and tumor-local focus.

## Reproducing builds

### Fixture smoke (local)

Requires fixture data under `tmp/issue121_fixture/` (centerlines, tumor masks,
labels). Not committed to the repo.

```bash
python deepsets/build_deepsets_dataset.py \
  --config configs/deepsets_issue121_fixture.yaml \
  --output-dir experiments/issue121_fixture_build \
  --num-shards 1 --shard-index 0
```

Or: `scripts/issue121_fixture_smoke.sh`

Summary: `experiments/issue121_fixture_build/inclusion_rule_summary.csv`

### Cohort comparison (Slurm)

Set `deepsets_compare_inclusion_rules` in the YAML (see
`configs/deepsets_issue121_fixture.yaml` for the full comparison list), then:

```bash
MODE=build CONFIG=configs/deepsets_ispy2.yaml \
  OUT_ROOT=experiments/issue121_inclusion_compare NUM_SHARDS=1 \
  SLURM_ARRAY_TASK_ID=0 sbatch slurm/deepsets_job.slurm
```

For a full shard array, use `slurm/submit_deepsets_pipeline.sh` with the same
`CONFIG` and `OUT_ROOT` (build stage only if you only need the summary CSV).

Summary: `experiments/issue121_inclusion_compare/inclusion_rule_summary.csv`

Routine production ISPY2 builds use `local_radius_with_fallback` only and omit
the compare list (see `slurm/README.md`).

## Notebook

Interactive tables and plots:
`analysis/deepsets_issue121_notebook.ipynb`
