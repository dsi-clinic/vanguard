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
| `report_by` | optional column to break QC/eval results down by (else `None`); **must exist in the predictions frame — a missing one raises rather than falling back to another column** |
| `tumor_mask_filename` / `centerline_filename` / `morphometry_filename` | per-case artifact naming |

---

## The two datasets today

### `MamaMiaDataset(cohort, root)` — `mamamia.py`
One class for all four MAMA-MIA cohorts (`duke`/`ispy1`/`ispy2`/`nact`), because
they are ~95% identical. The `cohort` argument sets a discovery filter (so one
instance represents one cohort) and identifies the DUKE radiologist-annotation
special case; `cohort=None` means no filter (`discover_cases()` returns all
four cohorts combined, matching `find_nii_files`' old unfiltered behavior —
needed by imaging Slurm jobs that process the whole MAMA-MIA tree in one run).
**No pipeline method is overridden** — it is the base class with a cohort label.

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
(sub-source breakdown). `preprocess()` is a **pass-through** — see below. The
181-exam student manifest lives at
`/gpfs/data/karczmar-lab/vanguard/dce2d_internal_ultrafast_manifest/`; select it
with `configs/uchicago.yaml`.

> **UChicago imaging does not run through this repo's NIfTI imaging stages.**
> Vessel segmentation and skeletonization for UChicago come from the paired
> raw-DICOM HR/UFAST pipeline in `preprocessing/`: it segments the
> *high-resolution* phases, builds the skeleton from HR vessel probabilities,
> maps that static skeleton onto the motion-corrected UFAST grid, and computes
> kinetics from raw UFAST signal using **physical timestamps** (not filename
> indices). Accordingly, `segmentation/batch_segmentation.py` and
> `graph_extraction/run_skeleton_processing.py` **reject** `--dataset-name
> uchicago` with a pointer to that pipeline (see
> `cohorts.factory.IMAGING_ROUTE_SUPERSEDED`), so there aren't two competing
> implementations that look interchangeable from the CLI.
>
> This adapter is still fully supported for every *non-imaging* consumer:
> discovery, labels, provided folds, patient grouping, and QC `report_by`.
>
> **History.** An earlier revision of this branch made `preprocess()` reorient
> UChicago volumes so they'd satisfy the vessel model's tiling-coverage
> assertion. That transform served the now-retired NIfTI route and has been
> removed along with it, so this adapter asserts no orientation convention at
> all — the raw-DICOM pipeline owns its own spatial handling. (One of those
> earlier attempts was a header-derived flip that turned out to silently
> left-right mirror every patient; near-symmetric breast anatomy hides a mirror
> in MIPs, which is why nothing here should be trusted from headers or visual
> checks alone.)

> Note: `case_dataset_name()` here returns a manifest **sub-source**
> (`simbiosys`/`uch_nac`/`her2_naclike`), a finer granularity than
> `MamaMiaDataset`'s cohort (`"ISPY2"`). The `dataset` column means different
> things across adapters — fine while runs stay separate, but flag it before
> building a combined MAMA-MIA+UChicago table keyed on `dataset`.

`load_timepoints()` rebases each `phase_files` path from the manifest's recorded
`preproc_root` onto the injected `root / "images"`, so moving the manifest
directory (decision 5) doesn't leave it pointing at a stale location.

---

## Selecting a dataset from run config — `factory.py`

These functions are the only seams that read run config:

- `build_adapter_from_config(config)` reads the `dataset:` block and returns the
  right adapter, or **`None`** when no dataset is configured.
- `require_adapter_from_config(config)` / `require_imaging_adapter_from_config(config)`
  wrap the above and raise a clear error instead of returning `None` — every
  stage uses one of these (Step 5).
- `resolve_split_policy(config, adapter)` applies the run-config `split_policy`
  knob (`auto`/`compute`/`provided`) on top of the adapter's default.
- `resolve_folds(config, adapter)` ties the two together for the caller: it
  returns the `(case_id, fold)` table to use when the policy resolves to
  `provided`, or `None` when the run should compute its own splits (raising if
  `provided` is asked of a dataset that ships no folds). This is parsing and
  validation only — see below for how (and whether) a run actually consumes it.

The `dataset:` block in run config (`config.py` `DEFAULT_CONFIG`), now required
on every run:

```yaml
dataset:
  name: mamamia        # "mamamia" | "uchicago"
  cohort: ispy2         # mamamia only: duke | ispy1 | ispy2 | nact | null (all four)
  root: /gpfs/data/karczmar-lab/MAMA-MIA-syn60868042
  split_policy: auto   # auto (use adapter default) | compute | provided
```

The dataset **root path** lives in run config (it can move on disk); everything
*under* the root is structurally fixed and lives in the class.

---

## How a pipeline stage uses an adapter

Every stage requires an adapter (Step 5 of the multi-dataset migration retired
the `adapter=None` fallback that early steps used for incremental adoption):

```python
adapter = require_adapter_from_config(config)   # raises if no dataset: block
result = some_stage(config, adapter=adapter)
```

`require_adapter_from_config` / `require_imaging_adapter_from_config` (in
`cohorts/factory.py`) wrap `build_adapter_from_config` /
`build_imaging_adapter_from_config` with a clear error when the config has no
`dataset:` block. `MamaMiaDataset(cohort=None, root=...)` selects all four
MAMA-MIA cohorts combined (no discovery filter) for runs that aren't scoped to
one cohort.

**Feature extraction.** `tabular/cohort.py`'s feature build takes a required
adapter and resolves a case's `dataset` identity via `case_dataset_name()`
instead of the parent directory name. Both entry points wire it the same way:
`tabular/train.py` (`run_pipeline_from_config`) and `modeling/ablation.py`
(`_prepare_full_dataset`, which also mirrors `tabular/train.py`'s
folds/grouping/`report_by` wiring). `tests/test_cohort_adapters.py` and
`tests/test_qc_report_by.py` cover the identity/reporting seams CI-safely.

**Provided CV folds.** `tabular/train.py`'s
`run_pipeline_from_config` calls `resolve_folds(config, adapter)` and, when it
returns a fold table, merges it onto the feature table as the
`model_params.split_col` column (`_apply_provided_folds`). This makes the folds
*available*; it does not by itself change which splits a run trains on. A run
opts in separately by setting `model_params.split_mode: predefined` (existing
infra in `evaluation/build_splits.py`, previously used for hardcoded fold
columns) with `split_col` matching the merged name — see `configs/uchicago.yaml`
for the pairing. Split *policy* (does the dataset have folds to offer) and split
*mode* (does this run use them) are deliberately separate knobs, per decision 3.
`prepare_evaluation_context` excludes `split_col` from the model's input
features unconditionally, so the fold assignment itself can never leak in as a
predictor.

Fold attachment is **fail-closed**: `_apply_provided_folds` rejects a `split_col`
that collides with `case_id` or the label column, rejects a provided-fold table
that maps a case more than once, merges `validate="one_to_one"` (so a duplicate
`case_id` on either side is an error, not a case fanned across folds), and
requires every modeled case to receive exactly one non-null fold. Any of these
raises rather than silently corrupting cross-validation.

**Adopted so far — patient grouping for computed folds.** When a run instead
*computes* folds (`split_policy: compute`), `_apply_group_keys` fills
`model_params.group_col` from `adapter.group_key(case_id)` (unless that column is
already present), so `create_splits_for_dataframe` does grouped CV that keeps a
case's group together — for UChicago, all exams of one patient (`patient_key`;
181 exams from 143 patients). When a dataset adapter is configured,
`prepare_evaluation_context` drops `group_col` from the model features, so an
identity-like grouping key can never leak in as a predictor. See
`configs/uchicago.yaml`.

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
