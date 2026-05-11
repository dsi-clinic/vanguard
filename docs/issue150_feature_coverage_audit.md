# Issue #150 — Tabular feature coverage audit (all 4 datasets)

## What this delivers

A reproducible per-case audit of upstream tabular-pipeline artifact coverage
across **DUKE, ISPY1, ISPY2, NACT**, plus a multi-dataset config that exposes
the full labeled cohort to `tabular_cohort.py` without any additional Slurm
reruns.

## Bottom line

- Every one of the **1506 study directories** under
  `/net/projects2/vanguard/centerlines_tc4d/studies/` has all five upstream
  artifacts required by the tabular pipeline: centerline `.npy`, morphometry
  JSON, tumor-graph features JSON with `status == "ok"`, tumor mask, and
  patient info JSON for clinical features.
- Of those, **1491 have a pCR label** (15 cases — 11 DUKE + 4 ISPY1 — have no
  label in `pcr_labels.csv`, so they cannot be used for pCR modeling
  regardless of pipeline state).
- The “only 808 used in #118” gap was **entirely caused by config filters**
  (`dataset_include: ["ISPY2"]` and `bilateral_filter: False`), not by missing
  artifacts. With those filters relaxed, the cohort jumps from 808 to 1491
  cases (+85%) with no additional pipeline work.

| Dataset | Study dirs | Centerline `.npy` | Morph JSON | Tumor-graph `status=ok` | Tumor mask | pCR label | All artifacts + label |
|---|---|---|---|---|---|---|---|
| DUKE | 291 | 291 | 291 | 291 | 291 | 280 | 280 |
| ISPY1 | 171 | 171 | 171 | 171 | 171 | 167 | 167 |
| ISPY2 | 980 | 980 | 980 | 980 | 980 | 980 | 980 |
| NACT | 64 | 64 | 64 | 64 | 64 | 64 | 64 |
| **Total** | **1506** | **1506** | **1506** | **1506** | **1506** | **1491** | **1491** |

Per-case audit table: `results/feature_coverage_audit.csv`
Per-dataset summary: `results/feature_coverage_summary.csv`

## Reproduce the audit

```bash
micromamba run -n vanguard python scripts/audit_feature_coverage.py
```

The script is read-only, walks the centerline tree directly, and takes
~1 minute. All paths can be overridden via CLI flags (see `--help`); the
defaults match `configs/issue118_baseline_arms.yaml`.

## Multi-dataset config

`configs/all_datasets.yaml` mirrors the structure of
`configs/issue118_baseline_arms.yaml`, with two changes:

```yaml
feature_toggles:
  dataset_include: ["DUKE", "ISPY1", "ISPY2", "NACT"]
  bilateral_filter: null
```

End-to-end smoke test (12-minute centerline parse over 1506 studies):

```
Centerline build applied dataset prefilter: ['DUKE', 'ISPY1', 'ISPY2', 'NACT']
Centerline file coverage:    1506 / 1506
Tumor mask coverage:         1506 / 1506
Tumor-graph JSON coverage:   1506 / 1506
Merged feature table shape:  (1506, 1216)
Applied dataset filter ['DUKE', 'ISPY1', 'ISPY2', 'NACT']: 1506 -> 1506 rows
After inner-merging labels:  (1491, 1217)
  DUKE   280
  ISPY1  167
  ISPY2  980
  NACT    64
```

No code changes to `tabular_cohort.py` were required — it already supports
`dataset_include` as a list and skips the bilateral filter when set to
`null`.

## Slurm reruns

**None required.** The pre-audit assumption (that DUKE / ISPY1 / NACT might
need vessel-segmentation, centerline, or tumor-graph reruns) turned out to be
false: every study directory has every artifact with `status == "ok"`.

## What this unblocks

`configs/all_datasets.yaml` is now a drop-in starting point for any future
analysis that wants the wider cohort — including the next-step issues:

- #118 follow-up: rerun the baseline-vs-vessel arm comparison on n = 1491
  instead of n = 808
- #117 follow-up: site-exclusive group CV across all four datasets (much
  stronger out-of-site stress test than within-ISPY2)
- #151: rank top features and evaluate top-K vs. baseline on the wider cohort

## Refs

- Closes #150
- Refs #118 (the ISPY2-only constraint addressed here)
- Refs #117, #151 (downstream analyses that now have access to the wider
  cohort)
