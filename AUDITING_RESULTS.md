# Auditing Results — documented data-handling decisions

Per the repo's fail-fast error-handling philosophy, the pipeline avoids silent
fallbacks/imputation. Where a genuine, known data edge case is nonetheless
handled deliberately, it is recorded here (what was handled, why, and what
failure mode it protects against), and flagged at the point of use.

## GNN — "no signal" sentinels (time-to-enhancement, bifurcation angle)

- **What is handled.** In the GNN centerline pipeline, two features are `NaN`
  for a real, expected reason rather than a bug: `time_to_enhancement` (and its
  segment summaries `seg_time_to_enhancement_mean` /
  `seg_time_to_enhancement_std`) is `NaN` for any voxel / segment / edge with no
  detected contrast arrival (peak enhancement ≤ 0, i.e. non-enhancing tissue);
  `bifurcation_angle_{mean,min,max}` (junction mode only) is `NaN` for a
  degree-1 endpoint node, which has no neighbor pair to measure an opening
  angle from. When such a feature is requested,
  `gnn/data_loader.py::_finalize_data` (via `_sentinel_fill`, called once per
  registered category) replaces each `NaN` with its own fixed sentinel --
  `TTE_NO_ARRIVAL_SENTINEL = -1.0` in the normalized `[0, 1]` time space, or
  `NO_BIFURCATION_SENTINEL = -1.0` (angles are always ≥ 0 degrees, so `-1.0` is
  equally out-of-range there) -- applied identically to voxel nodes, segment
  nodes, and junction nodes/edges wherever each feature applies.
- **Why.** A raw `NaN` in `data.x` / `data.edge_attr` propagates to a `NaN`
  training loss, so neither feature can be used without a policy. Each sentinel
  is *not* imputation to a plausible value: both are out-of-range for their
  feature's real domain, so "no detectable arrival" / "not a bifurcation"
  becomes a distinct, learnable value (both may themselves be prognostic --
  non-enhancement, or being a vessel terminus) rather than being blended into
  the mean. The TTE sentinel was chosen over mean-fill (which erases the
  no-arrival signal) with the user for the matched
  voxel-vs-segment-vs-junction comparison (`gnn/DESIGN_segment_graph.md`,
  `experiments/graph_repr_compare_v1/`); the bifurcation sentinel mirrors that
  same reasoning and was added when bifurcation angles were wired in as a
  junction node feature.
- **Failure mode it protects against.** Silent corruption of results from a
  `NaN` loss, or from imputing a fake arrival time / angle. Each fill is
  bounded and audited: `_raise_on_unexpected_nan` **raises** if any `NaN`
  remains after both sentinel fills run (so an unexpected NaN surfaces loudly
  instead of being filled), and the per-graph no-arrival / no-bifurcation
  counts are written to `graph_qc.csv`'s `tte_no_arrival_count` /
  `no_bifurcation_count` (so neither fill is ever silent). The features that
  may legitimately be `NaN` are pinned in `gnn/data_loader.py`'s
  `_TTE_FEATURE_NAMES` / `_BIFURCATION_FEATURE_NAMES`.
- **Where.** `gnn/data_loader.py` (`TTE_NO_ARRIVAL_SENTINEL`,
  `NO_BIFURCATION_SENTINEL`, `_TTE_FEATURE_NAMES`, `_BIFURCATION_FEATURE_NAMES`,
  `_sentinel_fill`, `_raise_on_unexpected_nan`, `_finalize_data`,
  `_write_graph_qc`); documented in `gnn/README.md` (Graph QC summary) and
  `gnn/DESIGN_segment_graph.md`; tests in `tests/test_gnn_data_loader.py`.

## GNN — clinical covariate imputation and category normalization

- **What is handled.** Graph-level clinical covariates (`gnn.clinical`, used
  when `gnn_graph_features` is set) go through a `sklearn.compose.ColumnTransformer`:
  numeric columns (`age`) are mean-imputed, categorical columns
  (`menopausal_status`, `breast_density`, `tumor_subtype`, and the opt-in
  site/scanner columns) are most-frequent imputed then one-hot encoded
  (`handle_unknown="ignore"`). Additionally, `menopausal_status`'s raw values
  are normalized through an exhaustive, hand-built map (`_MENOPAUSAL_STATUS_MAP`)
  before encoding, and `breast_density` is excluded from
  `DEFAULT_CLINICAL_COLUMNS`.
- **Fit boundary (cross-validation leakage).** The imputer and one-hot encoder
  are fit **per cross-validation fold on the training split only**, never once
  over the whole cohort. A whole-cohort fit would let validation cases
  contribute to the imputer means/modes and to the one-hot category vocabulary,
  so the training representation would depend on the validation distribution --
  optimistic CV and inconsistent with deployment on a genuinely unseen cohort.
  To make that split enforceable, the raw (normalized-but-un-imputed,
  un-encoded) inputs are cached once at build time in
  `processed/graph_feature_inputs.csv` (`_GRAPH_FEATURE_INPUTS_NAME`), and
  `gnn/train.py::_fit_transform_graph_features` fits the transformer on each
  fold's training cases and applies it to that fold's validation cases (a
  category unseen in training encodes as all-zero, i.e. treated as unknown).
  `menopausal_status` normalization is deterministic per case (no cross-case
  fit), so it is still applied once at build time. A cache built before this
  fix (which baked whole-cohort-fit features into the graphs) is detected on
  load and refuses to serve silently; regenerate the sidecar without a full
  rebuild via `python -m gnn.build_dataset --regenerate-graph-feature-inputs`.
- **Why.** A case with one incomplete clinical field (e.g. `age` recorded but
  `menopausal_status` blank) shouldn't be dropped from the cohort entirely --
  the case still has a real vessel graph and label, and the missing field is a
  data-entry gap, not evidence the case is unusable. This mirrors the
  established pattern in `tabular/models.py` (`SimpleImputer` +
  `OneHotEncoder(handle_unknown="ignore")` via `ColumnTransformer`) rather than
  inventing a new convention. The two extra steps are grounded in the real
  MAMA-MIA `patient_info_files` cohort (1506/1506 cases inspected
  exhaustively, 2026-07): `menopausal_status` contains wording/whitespace
  variants of the same category (e.g. `"pre"` vs. `"pre (<6 months since
  LMP...)"` vs. `"pre (< 6 months since LMP...)"`, note the differing space
  after `<`) that would otherwise one-hot into spuriously distinct columns if
  encoded raw; `breast_density` is missing in 1446/1506 (96%) of real cases, so
  including it by default would mostly encode "was density recorded" (a
  data-provenance artifact) rather than density itself.
- **Failure mode it protects against.** Dropping cases wholesale over a single
  missing clinical field (losing real vessel-graph/label data unnecessarily);
  silently splitting one true clinical category into several near-duplicate
  one-hot columns (diluting that category's signal and making per-category
  coefficients uninterpretable); treating a 96%-missing column as if it carried
  real signal. `_normalize_menopausal_status` also **raises** on any raw value
  outside the exhaustive map, so a genuinely new category (data added after
  this map was written) is triaged explicitly rather than silently bucketed
  into "missing" or left to blow up the one-hot vocabulary uncontrolled.
- **Where.** `gnn/clinical.py` (`normalize_clinical_frame`,
  `fit_clinical_transformer`, `transform_clinical_frame`, and the
  `build_clinical_feature_matrix` one-shot wrapper; `_MENOPAUSAL_STATUS_MAP`,
  `_normalize_menopausal_status`, `DEFAULT_CLINICAL_COLUMNS`);
  `gnn/data_loader.py` (`_load_clinical_df`, `_build_graph_feature_inputs`,
  `load_graph_feature_inputs`); the per-fold fit in
  `gnn/train.py::_fit_transform_graph_features`; tests in
  `tests/test_gnn_clinical.py` and `tests/test_gnn_graph_feature_folds.py`.

## GNN — morphometry graph-level feature imputation

- **What is handled.** `gnn/morphometry.py` extracts ~40 `morph_*` scalars per
  case (segment length/tortuosity/volume/curvature/radius summary stats,
  bifurcation angle stats, counts) from each case's `<case_id>_morphometry.json`
  via `features/morph.py`'s existing `extract_morphometry_features`, then
  mean-imputes via `sklearn.impute.SimpleImputer(strategy="mean")`.
- **Fit boundary (cross-validation leakage).** Like the clinical imputer above,
  the mean-imputer is fit **per cross-validation fold on the training split
  only** (`gnn/train.py::_fit_transform_graph_features`), not once over the
  whole cohort -- otherwise validation cases would contribute to the column
  means used to fill training values. The raw (un-imputed) scalars are cached
  once at build time in `processed/graph_feature_inputs.csv`; see the clinical
  section above for the same sidecar/regeneration mechanism.
- **Why.** `features/morph.py::array_stats` (the helper that produces every
  `_sum`/`_mean`/`_std`/`_max` column) returns `NaN` when the underlying group
  has zero valid entries -- e.g. `morph_seg_dup_fraction` is `NaN` whenever a
  case has zero raw segments at all, and any of the six stat groups
  (`seg_length`, `seg_tortuosity`, `seg_volume`, `curvature_mean`,
  `radius_mean`, `bif_angle`) can independently be empty for a sparse/small
  vessel graph. This is a real, observed-in-the-wild case (not hypothetical):
  `morphometry_path` itself is present for 100% of the real MAMA-MIA cohort
  (1506/1506 cases, 2026-07), but the *contents* of a present file can still
  legitimately yield an empty stat group for a case with few segments or
  bifurcations. A raw `NaN` reaching `data.graph_features` would propagate to
  a `NaN` training loss the same way an unhandled `NaN` in `data.x`/
  `data.edge_attr` would.
- **Failure mode it protects against.** A `NaN` loss silently corrupting
  training, or -- the alternative this avoids -- treating a per-case-empty
  stat group as a reason to drop the case from the cohort entirely (unlike
  clinical data, where per-case missingness is expected and case-level; here
  the file itself is always present, so dropping the whole case over one
  empty stat group inside an otherwise-valid file would be a worse trade-off
  than mean-imputing that one column). Unlike the clinical case, there is no
  categorical-normalization step here (`morph_*` columns are already
  all-numeric), so the only fallback in play is numeric mean-imputation, not
  category bucketing.
- **Where.** `gnn/morphometry.py` (`extract_morphometry_frame`,
  `fit_morphometry_imputer`, `transform_morphometry_frame`, and the
  `build_morphometry_feature_matrix` one-shot wrapper; `MORPHOMETRY_COLUMNS`);
  `features/morph.py` (`array_stats`, `extract_morphometry_features`, the
  actual `NaN` source); `gnn/data_loader.py` (`_resolve_morphometry_paths`,
  `_build_graph_feature_inputs`); the per-fold fit in
  `gnn/train.py::_fit_transform_graph_features`; tests in
  `tests/test_gnn_morphometry.py` and `tests/test_gnn_graph_feature_folds.py`.

## GNN — bilateral -> single-breast splitter exclusion policy

- **What is handled.** `gnn/breast_split.py::split_case` restricts a
  bilateral case's skeleton to its tumor-bearing breast (generalizing the
  inner-wall midline method from `tabular/duke_final_symmetry_lines.py`,
  authored by `aakrithiram`). Rather than always producing a best-guess
  single-breast skeleton, it explicitly **excludes** a case (returns
  `included=False` with a specific `exclude_reason`, no filtered skeleton)
  in four situations: `breast_mask_empty` (no breast mask voxels at all,
  so no midline can be computed), `tumor_mask_empty` (no tumor mask, so the
  tumor-bearing side is unknowable), `midline_detection_failed:<reason>`
  (the inner-wall method can't find a clean wall pair on both sides of the
  image center -- notably, this is the *expected* outcome for a native
  unilateral image, which has no second wall; native unilateral cases must
  be routed around this module entirely rather than passed through it),
  `tumor_mostly_outside_breast_mask` (fewer than 50% of the tumor mask's raw
  voxels overlap either breast mask -- signals the tumor/breast masks
  aren't actually co-registered for this case), and
  `tumor_overlap_both_sides` (fewer than 90% of the tumor voxels that do
  overlap a breast mask are on one side of the midline -- covers both a
  near-midline/ambiguous tumor and genuinely bilateral disease, which this
  module has no way to tell apart from geometry alone).
- **Why.** The harmonized single-breast dataset this feeds only makes sense
  if "the tumor-bearing side" is actually known with reasonable confidence
  for every retained case -- silently guessing a side for an ambiguous case
  would inject a labeling error into the cohort exactly where the model is
  supposed to become *more* accurate, not less. The `tumor_overlap_both_sides`
  threshold (90%) is a judgment call, not derived from data, chosen so a
  tumor that's overwhelmingly on one side (small amount of mask-boundary
  noise near the midline) still passes, while a tumor split roughly evenly
  across both breasts does not. It has not yet been validated against
  Aakrithi's `bilateral_breast_cancer` field in the shared side-assignment
  manifest (`mama_mia_bilateral_unilateral_cancer_manifest.csv`, not yet
  readable from this environment -- see `LAB_NOTEBOOK.md`, 2026-07-14); that
  field may be a more authoritative "genuinely bilateral disease" signal
  than the geometric heuristic here and should be cross-checked once
  accessible, per the visual-QC gate before any training run consumes this
  cohort.
- **Failure mode it protects against.** Silently mislabeling which breast is
  "the" tumor-bearing side for an ambiguous or genuinely bilateral case,
  which would corrupt the very comparison (mixed vs. single-breast graphs)
  this cohort exists to support. One bilateral case is never split into two
  independently labeled samples -- only the retained side is produced, or
  the case is excluded outright.
- **Where.** `gnn/breast_split.py` (`split_case`,
  `MIN_TUMOR_SIDE_VOXEL_FRACTION`, `MIN_TUMOR_IN_BREAST_FRACTION`,
  `MIN_INNER_WALL_ROW_FRACTION`); reuses
  `tabular.duke_final_symmetry_lines._inner_wall_midline_from_projection`
  and related geometric helpers; tests in `tests/test_gnn_breast_split.py`
  (synthetic arrays) plus a real-data smoke check on `DUKE_001` (see
  `LAB_NOTEBOOK.md`, 2026-07-14).

## GNN — tumor-mask axis-0 orientation ambiguity

- **What is handled.** `gnn/breast_split.py::resolve_tumor_mask_orientation`
  resolves a real orientation ambiguity in the expert tumor-segmentation
  NIfTI files (`MAMA-MIA-syn60868042/segmentations/expert/*.nii.gz`)
  relative to the skeleton and breast-mask `.npy` arrays.
  `tabular.duke_final_symmetry_lines._load_mask_matching_shape` (reused by
  `gnn/breast_split.py`) only searches axis *permutations* to match a
  target shape -- it never tries axis *flips*, so it can't detect or fix a
  same-shape-but-mirrored orientation. For a real, non-trivial fraction of
  cases, the tumor mask loads with array axis 0 reversed relative to the
  breast mask it should sit inside. The fix: load the tumor mask normally,
  then compare its overlap with the breast mask flipped vs. unflipped on
  axis 0, and keep whichever orientation has higher overlap.
- **Why.** Discovered while running Phase 3 QC of the harmonized
  single-breast dataset plan (`gnn/breast_split.py::split_case`'s
  `tumor_mostly_outside_breast_mask` exclusion, 2026-07-14): a first
  30-case real-data QC batch (18 DUKE + 12 ISPY2, `qc/breast_split_v1/`)
  excluded 23/30 (77%) cases for `tumor_mostly_outside_breast_mask`, all
  for the identical reason, with every included case landing at exactly
  `tumor_side_voxel_fraction=1.0` and every excluded case's raw
  tumor-in-breast overlap fraction clustering near 0 (median 0.018 across
  all 30, `qc/breast_split_v1/overlap_diagnostic.csv`) -- too uniform to be
  organic registration noise. Flipping tumor mask array axis 0 and
  recomputing overlap against the breast mask confirmed it directly: on
  11 real cases spanning both datasets, flipping took several from ~0%
  tumor-in-breast overlap to 100% (e.g. `DUKE_034`: 0/31765 -> 31765/31765),
  while 3 already-good cases were left effectively unchanged (one,
  `DUKE_057`, got very slightly *worse* under the flip -- 99.4% ->
  94.9% -- which is why the fix compares both orientations rather than
  always flipping). A real tumor is, by definition, inside breast tissue,
  so "which orientation puts it inside the breast mask" is a physically
  grounded disambiguator, not an arbitrary tie-break. After the fix, the
  same 30-case sample went to 30/30 included, all with
  `tumor_side_voxel_fraction=1.0`, and a plausible single-breast retention
  fraction (mean 51.5%, range 25.5-73.6% of the original skeleton) --
  visually spot-checked (4 cases spanning both datasets, both retained
  sides, flipped and unflipped, and the extremes of the retention-fraction
  range) with no contralateral/midline leakage observed.
- **Failure mode it protects against.** Without this fix, the harmonized
  single-breast cohort would have silently and overwhelmingly excluded
  correctly-registered cases as `tumor_mostly_outside_breast_mask` --
  not a data-quality problem but a code bug in how a specific file format
  was interpreted, which would have shrunk the usable cohort by ~75% and,
  worse, biased it toward whatever minority of cases happened to load in
  the "lucky" orientation.
- **Where.** `gnn/breast_split.py` (`resolve_tumor_mask_orientation`);
  called from `analysis/breast_split_qc.py::run_one_case` before
  `split_case`; tests in `tests/test_gnn_breast_split.py`
  (`test_resolve_tumor_mask_orientation_*`); real-data diagnosis in
  `analysis/breast_split_diagnose.py` and `qc/breast_split_v1/`.

## GNN — `dataset`/`site` derived from case-id prefix, not directory structure

- **What is handled.** `gnn/data_loader.py::_finalize_data` sets
  `data.dataset`/`data.site` from `case_id.split("_")[0]` (e.g.
  `"DUKE_001"` -> `"DUKE"`). Previously it derived this from
  `mask_path.parent.relative_to(centerline_root)` -- i.e. from where the
  skeleton file lives on disk, on the assumption that every graph's
  skeleton lives under `<centerline_root>/<dataset>/<case_id>/`.
- **Why.** Wiring `breast_split_mode="single"`
  (`gnn/data_loader.py::_resolve_breast_split_paths`) breaks that
  assumption on purpose: a bilateral case's skeleton is substituted with a
  precomputed single-breast skeleton living under a *different* root
  entirely (`--breast-split-skeleton-root`, a separate workspace directory
  from `gnn.build_single_breast_skeletons` -- deliberately not written back
  into the shared `centerlines_tc4d/studies` tree). `mask_path.parent` for
  a substituted case is not a subpath of `centerline_root`, so
  `relative_to()` would raise. The case-id-prefix convention is already the
  canonical way dataset identity is determined elsewhere in this codebase
  (`cohorts/base.py::case_dataset_name`, `evaluation/selection.py`), so this
  is a simplification, not a special case bolted on for breast-split: one
  fewer path-based inference to keep in sync with the real directory
  layout, and `_build_case`/`_finalize_data` no longer need a
  `centerline_root` parameter at all (removed from both, and from the two
  call sites in `_build_cases`).
- **Failure mode it protects against.** A hard crash
  (`ValueError` from `Path.relative_to`) the first time
  `breast_split_mode="single"` was used on a real cohort, since every
  substituted case's `mask_path` lives outside `centerline_root` by
  design. Verified real cases still resolve to the correct
  `dataset`/`site` after the change (`DUKE_057` -> `"DUKE"`,
  `ISPY2_169536` -> `"ISPY2"`, matching case-id prefixes exactly).
- **Where.** `gnn/data_loader.py` (`_finalize_data`, `_build_case`,
  `_build_cases`); tests in `tests/test_gnn_data_loader.py` (the full
  existing suite exercises `data.dataset`/`data.site` on every synthetic
  case; new `test_breast_split_*` tests exercise the substituted-root case
  specifically).

## GNN — precomputed single-breast skeleton needs its support mask and run_summary.json alongside it

- **What is handled.** `gnn.build_single_breast_skeletons` writes, per
  included bilateral case, three files under
  `<out_root>/<dataset>/<case_id>/`: the single-breast skeleton
  (`gnn.breast_split.save_split_skeleton`), the matching single-breast
  support mask (`gnn.breast_split.save_split_support_mask`, using the
  *original* support-mask filename, not a "single_breast"-suffixed one),
  and an unmodified copy of `run_summary.json`.
- **Why.** `gnn/data_loader.py::_build_case` is unaware of breast-split
  substitution -- it resolves both the support mask (for
  radius/distance-transform features) and `study_timepoints` (from
  `run_summary.json`, for DCE kinetic features) via fixed patterns relative
  to `mask_path.parent`. Saving only the skeleton and pointing
  `breast_split_mode="single"` at it would make `_build_case` look for
  `<case_id>_skeleton_4d_exam_support_mask.npy` and `run_summary.json` in
  the new (skeleton-only) output directory and fail with
  `FileNotFoundError` -- caught during Phase 4 integration testing on a
  synthetic fixture before it could surface on a real cluster build. The
  support mask is genuinely re-split (masked to the same retained side as
  the skeleton, per the user's original request to "apply that mask to the
  skeleton/support graph"); `run_summary.json` is copied verbatim since
  acquisition timing doesn't depend on which breast is retained.
- **Failure mode it protects against.** A `FileNotFoundError` at graph-build
  time for every `breast_split_mode="single"` case, only discoverable once
  training actually ran against a precomputed cohort -- instead caught by a
  synthetic integration test in `tests/test_gnn_data_loader.py` and fixed
  before any real Slurm build was attempted.
- **Where.** `gnn/breast_split.py` (`save_split_support_mask`,
  `SUPPORT_MASK_FILENAME_PATTERN`); `gnn/build_single_breast_skeletons.py`
  (`process_one_case`); tests in `tests/test_gnn_breast_split.py`
  (`test_support_mask_*`, `test_save_split_support_mask_*`) and
  `tests/test_gnn_data_loader.py` (`test_breast_split_single_substitutes_bilateral_case_skeleton`).
  Verified end-to-end on two real cases (`DUKE_057`, `ISPY2_169536`,
  2026-07-14) via a full `VanguardCenterlineDataset` build with
  `breast_split_mode="single"`.
