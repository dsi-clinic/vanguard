"""Graph-level clinical/demographic covariates for the GNN pCR classifier.

Unlike node/edge features (per-voxel/segment geometry and kinetics), these are
one fixed-width vector per *case*, built once from ``clinical_features.py``'s
loader and concatenated onto the pooled graph embedding in ``gnn.model``
(see ``graph_dim``). This module owns the column vocabulary, the split
between genuine clinical covariates and site/scanner metadata, and the
encoding of raw clinical values into a numeric matrix.

**Column vocabulary, grounded in the real MAMA-MIA cohort (2026-07,
1506/1506 ``patient_info_files`` cases inspected exhaustively):**

- ``DEFAULT_CLINICAL_COLUMNS`` = ``age``, ``menopausal_status``,
  ``tumor_subtype``. ``breast_density`` is deliberately excluded from the
  default set: it is missing in 1446/1506 (96%) of real cases, so as a
  feature it would mostly encode "was density recorded for this case" (a
  data-provenance artifact) rather than density itself. It remains available
  as an explicit opt-in for anyone who wants to test it despite the
  sparsity.
- ``SITE_CLINICAL_COLUMNS`` (``dataset``, ``site``, ``bilateral``,
  ``field_strength``, ``echo_time``, ``repetition_time``,
  ``scanner_manufacturer``, ``scanner_model``) mirror
  ``features.clinical.SITE_COLUMNS`` restricted to what
  ``clinical_features.get_clinical_features`` actually populates. These are
  the codebase's own established site/scanner confound taxonomy (see
  ``tabular/cohort.py``'s ``include_site_features`` toggle, off by default)
  and are opt-in only here too -- never included by
  ``DEFAULT_CLINICAL_COLUMNS``.

**Missing-value handling (flagged per this repo's error-handling rules):**
categorical columns are imputed with the most-frequent observed category and
numeric columns with the column mean, via a ``SimpleImputer`` +
``OneHotEncoder`` ``ColumnTransformer`` -- the same pattern ``tabular/models.py``
already uses for clinical covariates. This is a real
default-value-for-missing-data case; see ``AUDITING_RESULTS.md`` for why it's
needed (a case with one incomplete clinical field shouldn't be dropped from
the cohort) and what it protects against.

**Fit boundary (cross-validation leakage):** the transformer is fit **per
cross-validation fold on the training split only**, never once over the whole
cohort. Fitting on the full cohort would let validation cases contribute to
the imputer means/modes and to the one-hot category vocabulary, making the
training representation depend on the validation distribution -- optimistic
CV and inconsistent with deployment on a genuinely unseen cohort. The three
primitives below make that split explicit: :func:`normalize_clinical_frame`
produces the raw (un-imputed, un-encoded) inputs once at dataset-build time
(cached by ``gnn.data_loader``), then :func:`fit_clinical_transformer` /
:func:`transform_clinical_frame` fit on the training cases and transform the
validation cases per fold in ``gnn.train``. Categories unseen in the training
split are encoded as all-zero (``handle_unknown="ignore"``), i.e. treated as
unknown.

``menopausal_status`` additionally needs *normalization* before encoding: the
real cohort contains wording/whitespace variants of the same clinical
category (e.g. ``"pre"`` vs. ``"pre (<6 months since LMP...)"`` vs. ``"pre (<
6 months since LMP...)"`` -- note the differing space after ``<``) that would
otherwise one-hot into spuriously distinct columns.
``_MENOPAUSAL_STATUS_MAP`` is an *exhaustive* map built from every distinct
raw value observed across the full 1506-case cohort; a value outside it is
treated as a genuinely new category (data added since this map was written)
and raises rather than being silently bucketed.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERIC_CLINICAL_COLUMNS: frozenset[str] = frozenset({"age"})
CATEGORICAL_CLINICAL_COLUMNS: frozenset[str] = frozenset(
    {"menopausal_status", "breast_density", "tumor_subtype"}
)
DEFAULT_CLINICAL_COLUMNS: tuple[str, ...] = (
    "age",
    "menopausal_status",
    "tumor_subtype",
)

# Opt-in only (see module docstring) -- all are categorical except
# field_strength/echo_time/repetition_time, which are numeric scanner params.
SITE_NUMERIC_CLINICAL_COLUMNS: frozenset[str] = frozenset(
    {"field_strength", "echo_time", "repetition_time"}
)
SITE_CATEGORICAL_CLINICAL_COLUMNS: frozenset[str] = frozenset(
    {"dataset", "site", "bilateral", "scanner_manufacturer", "scanner_model"}
)
SITE_CLINICAL_COLUMNS: tuple[str, ...] = (
    "dataset",
    "site",
    "bilateral",
    "field_strength",
    "echo_time",
    "repetition_time",
    "scanner_manufacturer",
    "scanner_model",
)

ALL_NUMERIC_CLINICAL_COLUMNS: frozenset[str] = (
    NUMERIC_CLINICAL_COLUMNS | SITE_NUMERIC_CLINICAL_COLUMNS
)
ALL_CATEGORICAL_CLINICAL_COLUMNS: frozenset[str] = (
    CATEGORICAL_CLINICAL_COLUMNS | SITE_CATEGORICAL_CLINICAL_COLUMNS
)
ALL_CLINICAL_COLUMNS: frozenset[str] = (
    ALL_NUMERIC_CLINICAL_COLUMNS | ALL_CATEGORICAL_CLINICAL_COLUMNS
)

# Exhaustive as of the full 1506-case MAMA-MIA patient_info cohort (2026-07).
_MENOPAUSAL_STATUS_MAP: dict[str, str] = {
    "post": "post",
    "pre": "pre",
    "pre (<6 months since LMP AND no prior bilateral ovariectomy AND not on "
    "estrogen replacement)": "pre",
    "pre (< 6 months since LMP AND no prior bilateral ovariectomy AND not on "
    "estrogen replacement)": "pre",
    "peri (6-12 months since LMP AND no prior bilateral ovariectomy AND not "
    "on estrogen replacement)": "peri",
}


def _is_missing(value: object) -> bool:
    """True for None, NaN, and blank/whitespace-only strings."""
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return isinstance(value, str) and value.strip() == ""


def _normalize_menopausal_status(value: object) -> str | float:
    if _is_missing(value):
        return np.nan
    if value not in _MENOPAUSAL_STATUS_MAP:
        raise ValueError(
            f"Unrecognized menopausal_status value {value!r} -- not in "
            "_MENOPAUSAL_STATUS_MAP. This map was built from every distinct "
            "raw value in the full MAMA-MIA patient_info cohort; a value "
            "outside it is a genuinely new category (data added since the "
            "map was written), not something to silently bucket. Add it "
            "explicitly to _MENOPAUSAL_STATUS_MAP after confirming what "
            "canonical category it belongs to."
        )
    return _MENOPAUSAL_STATUS_MAP[value]


def _normalize_categorical_column(series: pd.Series, column: str) -> pd.Series:
    """Normalize to a canonical category or ``np.nan``.

    Uses ``np.nan`` rather than ``None`` for missing values: scikit-learn's
    ``SimpleImputer`` (default ``missing_values=np.nan``) does not reliably
    treat Python ``None`` as missing in an object-dtype column -- it was
    observed treating ``None`` as its own valid category, producing a
    spurious extra one-hot column instead of imputing it away.
    """
    if column == "menopausal_status":
        return series.map(_normalize_menopausal_status)
    return series.map(lambda v: np.nan if _is_missing(v) else v)


def normalize_clinical_frame(
    clinical_df: pd.DataFrame,
    case_ids: Sequence[str],
    columns: Sequence[str],
) -> pd.DataFrame:
    """Select ``columns`` for ``case_ids`` and normalize to raw, un-imputed inputs.

    Returns a frame indexed by ``case_id`` (in ``case_ids`` order) whose
    categorical columns are canonicalized (``_normalize_categorical_column``)
    and numeric columns coerced to float, with ``np.nan`` for every missing
    value. Normalization is deterministic per case (no cross-case fit), so it
    is done once at dataset-build time; imputation and one-hot encoding are
    deliberately *not* applied here -- they are fit per cross-validation fold
    on the training split by :func:`fit_clinical_transformer` /
    :func:`transform_clinical_frame`, which is what keeps validation cases out
    of the imputer means/modes and the one-hot vocabulary (see module
    docstring).

    ``clinical_df`` must have a ``case_id`` column and a row for every id in
    ``case_ids`` (the caller -- ``gnn.data_loader`` -- resolves which cases
    actually have a clinical record and drops the rest with an audited
    threshold beforehand; a missing case here is a bug, not an expected "no
    clinical data" case, and raises).
    """
    unknown = [c for c in columns if c not in ALL_CLINICAL_COLUMNS]
    if unknown:
        raise ValueError(
            f"Unknown clinical columns {unknown}; supported: "
            f"{sorted(ALL_CLINICAL_COLUMNS)}"
        )

    indexed = clinical_df.set_index("case_id")
    missing_cases = [c for c in case_ids if c not in indexed.index]
    if missing_cases:
        raise KeyError(
            f"No clinical row for case(s) {missing_cases}; the caller must "
            "resolve case coverage against clinical_df before calling "
            "normalize_clinical_frame."
        )
    frame = indexed.loc[list(case_ids), list(columns)].copy()
    frame.index = pd.Index([str(c) for c in case_ids], name="case_id")

    numeric_cols = [c for c in columns if c in ALL_NUMERIC_CLINICAL_COLUMNS]
    categorical_cols = [c for c in columns if c in ALL_CATEGORICAL_CLINICAL_COLUMNS]
    for col in categorical_cols:
        frame[col] = _normalize_categorical_column(frame[col], col)
    for col in numeric_cols:
        frame[col] = pd.to_numeric(
            frame[col].map(lambda v: np.nan if _is_missing(v) else v)
        )
    return frame


def fit_clinical_transformer(
    frame: pd.DataFrame, columns: Sequence[str]
) -> ColumnTransformer:
    """Fit the impute + one-hot ``ColumnTransformer`` on ``frame``.

    ``frame`` must be a :func:`normalize_clinical_frame` output restricted to
    one cross-validation fold's **training** cases. Fitting here -- not over
    the whole cohort -- is what keeps the imputer means/modes and the one-hot
    category vocabulary derived from training data only. Numeric columns are
    mean-imputed; categorical columns are most-frequent-imputed then one-hot
    encoded with ``handle_unknown="ignore"`` so a category unseen in training
    encodes as all-zero (treated as unknown) at transform time.
    """
    numeric_cols = [c for c in columns if c in ALL_NUMERIC_CLINICAL_COLUMNS]
    categorical_cols = [c for c in columns if c in ALL_CATEGORICAL_CLINICAL_COLUMNS]
    transformers: list[tuple[str, object, list[str]]] = []
    if numeric_cols:
        transformers.append(("num", SimpleImputer(strategy="mean"), numeric_cols))
    if categorical_cols:
        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transformers.append(("cat", cat_pipeline, categorical_cols))
    column_transformer = ColumnTransformer(transformers=transformers)
    column_transformer.fit(frame)
    return column_transformer


def transform_clinical_frame(
    column_transformer: ColumnTransformer, frame: pd.DataFrame
) -> tuple[np.ndarray, list[str]]:
    """Apply a fitted clinical transformer to ``frame`` (train or validation rows).

    Returns ``(matrix, output_column_names)``: ``matrix`` has one row per
    ``frame`` row in that order, one column per encoded feature (numeric
    pass-through after mean-imputation; categorical one-hot after
    most-frequent imputation). ``output_column_names`` names each column, for
    provenance.
    """
    matrix = column_transformer.transform(frame)
    output_names = [str(name) for name in column_transformer.get_feature_names_out()]
    return np.asarray(matrix, dtype=np.float64), output_names


def build_clinical_feature_matrix(
    clinical_df: pd.DataFrame,
    case_ids: Sequence[str],
    columns: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    """Normalize, fit, and transform ``columns`` for ``case_ids`` in one shot.

    Convenience wrapper composing :func:`normalize_clinical_frame`,
    :func:`fit_clinical_transformer`, and :func:`transform_clinical_frame` on
    the **same** rows. Because it fits and transforms on the same cases, it is
    only leakage-free when ``case_ids`` is a single fold's training split (or
    the whole cohort when no cross-validation is involved). Cross-validation
    code must instead normalize once and fit per fold on the training split --
    see ``gnn.train`` -- rather than calling this across a train/val boundary.
    """
    frame = normalize_clinical_frame(clinical_df, case_ids, columns)
    column_transformer = fit_clinical_transformer(frame, columns)
    return transform_clinical_frame(column_transformer, frame)
