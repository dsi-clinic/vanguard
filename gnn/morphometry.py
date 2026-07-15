"""Case-level morphometry features (``run_summary.json``'s ``morphometry_path``).

Reuses ``features.morph.extract_morphometry_features`` directly -- the same
function the tabular pipeline already uses to flatten each case's
``<case_id>_morphometry.json`` into a fixed set of ~40 ``morph_*`` scalars
(segment length/tortuosity/volume/curvature/radius summary stats, bifurcation
angle stats, bifurcation/segment counts, and duplicate/invalid-value QC
counters -- see that module for the exact formulas). No new parsing code.

``morphometry_path`` is present for 100% of the real MAMA-MIA cohort
(1506/1506 cases inspected, 2026-07), unlike ``tumor_graph_features_path``
(missing for 10.4%, out of scope -- see the GNN feature-extension plan), so
this module does not need a missing-case drop/threshold policy: a missing or
unreadable morphometry file is a build assumption violated, not an expected
case, and ``gnn.data_loader`` treats it as a hard failure.

**Imputation is still needed within a present file**, though: `array_stats`
(`features/morph.py`) returns ``NaN`` for `_sum`/`_mean`/`_std`/`_max` when a
component has zero valid segments or bifurcations (e.g. `morph_seg_dup_fraction`
is `NaN` when no segments were found at all). See ``AUDITING_RESULTS.md`` for
why this is handled with mean-imputation rather than left to propagate.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from features.morph import extract_morphometry_features

# Pinned from a real MAMA-MIA morphometry.json (DUKE_001, 2026-07) rather than
# derived dynamically -- extract_morphometry_features always emits the same
# key set regardless of input (array_stats fills every stat with NaN even for
# an empty group, see that function's docstring), so this tuple is stable.
MORPHOMETRY_COLUMNS: tuple[str, ...] = (
    "morph_bif_angle_invalid_n",
    "morph_bif_angle_max",
    "morph_bif_angle_mean",
    "morph_bif_angle_n",
    "morph_bif_angle_std",
    "morph_bif_angle_sum",
    "morph_bifurcation_count",
    "morph_curvature_mean_invalid_n",
    "morph_curvature_mean_max",
    "morph_curvature_mean_mean",
    "morph_curvature_mean_n",
    "morph_curvature_mean_std",
    "morph_curvature_mean_sum",
    "morph_radius_mean_invalid_n",
    "morph_radius_mean_max",
    "morph_radius_mean_mean",
    "morph_radius_mean_n",
    "morph_radius_mean_std",
    "morph_radius_mean_sum",
    "morph_seg_dup_fraction",
    "morph_seg_length_invalid_n",
    "morph_seg_length_max",
    "morph_seg_length_mean",
    "morph_seg_length_n",
    "morph_seg_length_std",
    "morph_seg_length_sum",
    "morph_seg_raw_n",
    "morph_seg_tortuosity_invalid_n",
    "morph_seg_tortuosity_max",
    "morph_seg_tortuosity_mean",
    "morph_seg_tortuosity_n",
    "morph_seg_tortuosity_std",
    "morph_seg_tortuosity_sum",
    "morph_seg_unique_n",
    "morph_seg_volume_invalid_n",
    "morph_seg_volume_max",
    "morph_seg_volume_mean",
    "morph_seg_volume_n",
    "morph_seg_volume_std",
    "morph_seg_volume_sum",
)


def build_morphometry_feature_matrix(
    morphometry_paths_by_case: dict[str, Path],
    case_ids: Sequence[str],
    columns: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    """Extract ``columns`` for ``case_ids`` from each case's morphometry JSON.

    ``morphometry_paths_by_case`` must have an entry for every id in
    ``case_ids`` (the caller -- ``gnn.data_loader`` -- resolves this from
    ``run_summary.json``, hard-failing rather than dropping cases, since
    ``morphometry_path`` is present for 100% of the real cohort; see module
    docstring).

    Returns ``(matrix, output_column_names)``: ``matrix`` has one row per
    ``case_ids`` entry in that order and one column per requested feature,
    mean-imputed via a fit-once ``SimpleImputer`` (see module docstring for
    why `NaN`s occur even in a present file). ``output_column_names`` equals
    ``list(columns)`` (no encoding/expansion, unlike ``gnn.clinical`` -- these
    are already all-numeric).
    """
    unknown = [c for c in columns if c not in MORPHOMETRY_COLUMNS]
    if unknown:
        raise ValueError(
            f"Unknown morphometry columns {unknown}; supported: "
            f"{sorted(MORPHOMETRY_COLUMNS)}"
        )
    missing_cases = [c for c in case_ids if c not in morphometry_paths_by_case]
    if missing_cases:
        raise KeyError(
            f"No morphometry_path for case(s) {missing_cases}; the caller "
            "must resolve case coverage before calling "
            "build_morphometry_feature_matrix."
        )

    rows = [
        extract_morphometry_features(morphometry_paths_by_case[case_id])
        for case_id in case_ids
    ]
    frame = pd.DataFrame(rows, columns=list(columns))
    matrix = SimpleImputer(strategy="mean").fit_transform(frame)
    return np.asarray(matrix, dtype=np.float64), list(columns)
