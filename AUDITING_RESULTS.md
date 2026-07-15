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

- **What is handled.** Graph-level clinical covariates (`gnn.clinical`,
  attached as `data.graph_features` when `gnn_graph_features` is set) are built
  by a fit-once `sklearn.compose.ColumnTransformer`
  (`gnn/clinical.py::build_clinical_feature_matrix`): numeric columns (`age`)
  are mean-imputed, categorical columns (`menopausal_status`, `breast_density`,
  `tumor_subtype`, and the opt-in site/scanner columns) are most-frequent
  imputed then one-hot encoded (`handle_unknown="ignore"`). Additionally,
  `menopausal_status`'s raw values are normalized through an exhaustive,
  hand-built map (`_MENOPAUSAL_STATUS_MAP`) before encoding, and
  `breast_density` is excluded from `DEFAULT_CLINICAL_COLUMNS`.
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
- **Where.** `gnn/clinical.py` (`build_clinical_feature_matrix`,
  `_MENOPAUSAL_STATUS_MAP`, `_normalize_menopausal_status`,
  `DEFAULT_CLINICAL_COLUMNS`); `gnn/data_loader.py`
  (`_load_clinical_df`/`_attach_graph_features`); tests in
  `tests/test_gnn_clinical.py`.

## GNN — morphometry graph-level feature imputation

- **What is handled.** `gnn/morphometry.py::build_morphometry_feature_matrix`
  extracts ~40 `morph_*` scalars per case (segment length/tortuosity/volume/
  curvature/radius summary stats, bifurcation angle stats, counts) from each
  case's `<case_id>_morphometry.json` via `features/morph.py`'s existing
  `extract_morphometry_features`, then mean-imputes via a fit-once
  `sklearn.impute.SimpleImputer(strategy="mean")` before the matrix is
  attached as part of `data.graph_features`.
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
- **Where.** `gnn/morphometry.py` (`build_morphometry_feature_matrix`,
  `MORPHOMETRY_COLUMNS`); `features/morph.py` (`array_stats`,
  `extract_morphometry_features`, the actual `NaN` source); `gnn/data_loader.py`
  (`_resolve_morphometry_paths`/`_attach_graph_features`); tests in
  `tests/test_gnn_morphometry.py`.
