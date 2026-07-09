# `cohorts/` — per-dataset adapters

This package is the one place that knows **how each dataset is shaped**: how to
find its cases, what a case's cohort identity is, how to orient/resample its
volumes, and where its clinical metadata and labels live. Pipeline stages ask an
adapter these questions instead of hardcoding the MAMA-MIA answer, so a new
dataset is added by writing one small class rather than editing stage logic in a
dozen files.

If you just want to *use* or *extend* this, read this file — you don't need the
full design/decision record (kept separately by the team).

---

## The contract: `DatasetAdapter`

`DatasetAdapter` (`base.py`) is the base class every dataset fills in. It doubles
as a checklist — to add a dataset you subclass it and override only the methods
that genuinely differ. The base class itself encodes **current MAMA-MIA
behavior**, so `MamaMiaDataset` needs zero overrides.

Construct an adapter with the dataset's on-disk `root` (injected from run config,
not hardcoded), then call:

| Method / attribute | Answers |
|---|---|
| `discover_cases()` | enumerate case ids (directory glob vs. manifest CSV) |
| `case_dataset_name(case_id)` | which dataset/cohort a case belongs to |
| `group_key(case_id)` | patient grouping for CV splits (site vs. `patient_key`) |
| `load_timepoints(case_id)` | ordered raw phase files for a case |
| `preprocess(volume)` | Stage-0 reorientation; base = the MAMA-MIA axis transform |
| `resample(volume, native_spacing_mm)` | resample to `target_spacing_mm` (no-op when it's `None`) |
| `load_clinical()` | per-case clinical/imaging metadata table |
| `load_labels()` | `(case_id, pcr)` label table |
| `target_spacing_mm` | explicit resample target, or `None` to keep native spacing |
| `default_split_policy` | `"compute"` (build our own folds) vs. `"provided"` (ship folds) |
| `report_by` | optional column to break QC/eval results down by (else `None`) |
| `tumor_mask_filename` / `centerline_filename` / `morphometry_filename` | per-case artifact naming |

---

## The two datasets today

### `MamaMiaDataset(cohort, root)` — `mamamia.py`
One class for all four MAMA-MIA cohorts (`duke`/`ispy1`/`ispy2`/`nact`), because
they are ~95% identical. The `cohort` argument sets a discovery filter (so one
instance represents one cohort) and identifies the DUKE radiologist-annotation
special case. **No pipeline method is overridden** — it is the base class with a
cohort label.

> Note: `case_dataset_name()` reads the cohort off the case-id prefix
> (`"ISPY2_045" -> "ISPY2"`), which is independent of the `cohort` argument. So
> in a stage that processes several cohorts in one table, any `MamaMiaDataset`
> instance resolves identity correctly regardless of its `cohort`.

### `UChicagoDataset(root, manifest_csv=None)` — `uchicago.py`
Genuinely different, so it overrides the handful of methods that differ:
`discover_cases()`, `case_dataset_name()`, `group_key()`, `load_timepoints()`,
`load_labels()`, and `load_folds()` are all **manifest-driven** (read from a CSV,
including a `phase_files` list per row); it sets `default_split_policy =
"provided"` (ships patient-grouped folds) and `report_by = "dataset"`
(sub-source breakdown). `preprocess()` is a **documented pass-through** — the
manifest's phase files are already preprocessed upstream (`policy_name =
hfdp_t1_v1`), so no repo-side transform is applied (and, importantly, the base
MAMA-MIA orientation transform is *not* used, which would be wrong here). The
181-exam student manifest lives at
`/gpfs/data/karczmar-lab/vanguard/dce2d_internal_ultrafast_manifest/`; select it
with `configs/uchicago.yaml`.

---

## Selecting a dataset from run config — `factory.py`

These functions are the only seams that read run config:

- `build_adapter_from_config(config)` reads the `dataset:` block and returns the
  right adapter, or **`None`** when no dataset is configured (so callers fall
  back to today's behavior).
- `resolve_split_policy(config, adapter)` applies the run-config `split_policy`
  knob (`auto`/`compute`/`provided`) on top of the adapter's default.
- `resolve_folds(config, adapter)` ties the two together for the caller: it
  returns the `(case_id, fold)` table to use when the policy resolves to
  `provided`, or `None` when the run should compute its own splits (raising if
  `provided` is asked of a dataset that ships no folds).

The `dataset:` block in run config (`config.py` `DEFAULT_CONFIG`):

```yaml
dataset:
  name: mamamia        # "mamamia" | "uchicago"; null (default) => no adapter
  cohort: ispy2        # mamamia only: duke | ispy1 | ispy2 | nact
  root: /gpfs/data/karczmar-lab/MAMA-MIA-syn60868042
  split_policy: auto   # auto (use adapter default) | compute | provided
```

The dataset **root path** lives in run config (it can move on disk); everything
*under* the root is structurally fixed and lives in the class.

---

## How a pipeline stage uses an adapter

A stage takes an **optional** adapter and falls back to today's behavior when
it's `None`, so adoption is incremental and can't break existing runs:

```python
adapter = build_adapter_from_config(config)   # None for every config w/o a dataset: block
result = some_stage(config, adapter=adapter)   # adapter=None => byte-identical to before
```

**Adopted so far — feature extraction.** `tabular/cohort.py`'s feature build
takes an optional adapter and, when given, resolves a case's `dataset` identity
via `case_dataset_name()` instead of the parent directory name. Both entry
points wire it the same way: `tabular/train.py` (`run_pipeline_from_config`) and
`modeling/ablation.py` (`_prepare_full_dataset`). The equivalence is guarded by
`tests/test_feature_adapter_parity.py` (CI) and the full-data gate
`scripts/validate_adapter_feature_parity.py` (byte-identical MAMA-MIA output with
vs. without the adapter). Other stages still run the pre-adapter way and are
migrated one at a time.

---

## How to add a new dataset

1. Add a subclass of `DatasetAdapter` in a new `cohorts/<name>.py`. Override
   **only** the methods where your dataset genuinely differs from MAMA-MIA — the
   base class handles the rest.
2. If your data needs a different orientation/target spacing, override
   `preprocess()` and/or set `target_spacing_mm` (+ implement `resample`).
3. Register it in `build_adapter_from_config()` (one new `if name == ...` branch)
   and export it from `__init__.py`.
4. Add it to the `dataset.name` options and document any dataset-specific config.
5. Add a unit test mirroring `tests/test_cohort_adapters.py`.

The base class is your checklist: anything you don't override keeps MAMA-MIA
behavior, so start minimal and override as you hit real differences.

---

## Design decisions (condensed)

The short "why", for context while reading the code. (The full rationale and the
alternatives we weighed are kept in the team's design record, not in this repo.)

1. **Class hierarchy, not a config/registry of profiles.** More explicit and
   more extensible long-term; the base class is a compile-time checklist.
2. **One `MamaMiaDataset` parameterized by `cohort`**, not four cohort
   subclasses — the four are ~95% identical, so four classes would be empty
   ceremony.
3. **Split policy is a run-config knob** (`auto`/`compute`/`provided`), not baked
   into the dataset — UChicago defaults to its shipped folds, but any run can
   force recomputed CV for cross-dataset comparability.
4. **UChicago preprocessing is a frozen copy ported into the repo** (versioned,
   team-editable), the deliberate opposite of the pinned vessel-seg submodule.
5. **Dataset facts live in the class; the root path lives in run config** and is
   injected into the adapter, because the root can move on disk.
6. **The package is `cohorts/`, not `datasets/`**, to avoid colliding with the
   Hugging Face `datasets` PyPI package.
