"""Project 1 second-order engineered features.

Each feature is derived from existing first-order columns and added in-place
to a row dict during the tabular pipeline.  A thin CLI wrapper
(``scripts/add_second_order_features.py``) allows retroactive application to
an existing CSV.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from features._common import safe_float, safe_ratio

SECOND_ORDER_COLUMNS: tuple[str, ...] = (
    # tumor_size
    "tumor_size_equiv_radius_vox",
    "tumor_size_shell_0_2_over_tumor",
    "tumor_size_outer_to_inner_shell_ratio",
    # morph
    "morph_seg_length_cv",
    "morph_radius_cv",
    "morph_bifurcation_density_per_length",
    # graph
    "graph_core_to_periphery_length_ratio",
    "graph_core_to_periphery_volume_ratio",
    "graph_near_branching_bias",
    "graph_crossing_fraction_of_near_burden",
    # kinematic
    "kinematic_core_to_periphery_tte_delta",
    "kinematic_core_to_periphery_peak_ratio",
    "kinematic_near_washin_to_washout_ratio",
    "kinematic_crossing_early_fraction",
    "kinematic_arrival_delay_dispersion_near",
)


def _sum_finite(*values: Any) -> float:
    """Sum values, returning NaN if any component is missing or non-finite."""
    total = 0.0
    for v in values:
        f = safe_float(v)
        if f is None or not np.isfinite(f):
            return np.nan
        total += f
    return total


def _mean_finite(*values: Any) -> float:
    """Mean of finite values, NaN if none qualify."""
    finite = []
    for v in values:
        f = safe_float(v)
        if f is not None and np.isfinite(f):
            finite.append(f)
    if not finite:
        return np.nan
    return sum(finite) / len(finite)


def _abs_ratio(numerator: float, denominator: float) -> float:
    """``|numerator| / |denominator|``, NaN-safe."""
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return np.nan
    denom = abs(denominator)
    if denom <= 0.0:
        return np.nan
    return abs(numerator) / denom


def add_second_order_features(row: dict[str, Any]) -> None:
    """Derive all Project 1 second-order features and add them to *row*.

    Operates on the mutable row dict produced during
    ``build_centerline_features`` (which contains both first-order flat
    columns and intermediate payload columns like ``per_shell_topology_*``).
    When called on a DataFrame row converted to a dict the same logic
    applies; features whose source columns are absent will be NaN.
    """

    # ------------------------------------------------------------------
    # tumor_size block
    # ------------------------------------------------------------------
    tumor_voxels = safe_float(row.get("tumor_size_tumor_voxels"))
    row["tumor_size_equiv_radius_vox"] = (
        float((3.0 / (4.0 * math.pi) * tumor_voxels) ** (1.0 / 3.0))
        if tumor_voxels is not None and tumor_voxels > 0.0
        else np.nan
    )

    row["tumor_size_shell_0_2_over_tumor"] = safe_ratio(
        row.get("tumor_size_tumor_shell_r0_2_voxels"),
        row.get("tumor_size_tumor_voxels"),
    )

    row["tumor_size_outer_to_inner_shell_ratio"] = safe_ratio(
        row.get("tumor_size_tumor_shell_r4_8_voxels"),
        row.get("tumor_size_tumor_shell_r0_2_voxels"),
    )

    # ------------------------------------------------------------------
    # morph block
    # ------------------------------------------------------------------
    row["morph_seg_length_cv"] = safe_ratio(
        row.get("morph_seg_length_std"),
        row.get("morph_seg_length_mean"),
    )

    row["morph_radius_cv"] = safe_ratio(
        row.get("morph_radius_mean_std"),
        row.get("morph_radius_mean_mean"),
    )

    row["morph_bifurcation_density_per_length"] = safe_ratio(
        row.get("morph_bifurcation_count"),
        row.get("morph_seg_length_sum"),
    )

    # ------------------------------------------------------------------
    # graph block — core vs periphery structural contrasts
    # ------------------------------------------------------------------
    # "core" = inside_tumor + shell_0_2mm
    # "periphery" = shell_5_10mm + shell_10_20mm
    core_length = _sum_finite(
        row.get("per_shell_topology_inside_tumor_shell_length_mm"),
        row.get("per_shell_topology_shell_0_2mm_shell_length_mm"),
    )
    periphery_length = _sum_finite(
        row.get("per_shell_topology_shell_5_10mm_shell_length_mm"),
        row.get("per_shell_topology_shell_10_20mm_shell_length_mm"),
    )
    row["graph_core_to_periphery_length_ratio"] = safe_ratio(
        core_length, periphery_length
    )

    core_vol = _sum_finite(
        row.get("per_shell_topology_inside_tumor_shell_volume_burden_mm3"),
        row.get("per_shell_topology_shell_0_2mm_shell_volume_burden_mm3"),
    )
    periphery_vol = _sum_finite(
        row.get("per_shell_topology_shell_5_10mm_shell_volume_burden_mm3"),
        row.get("per_shell_topology_shell_10_20mm_shell_volume_burden_mm3"),
    )
    row["graph_core_to_periphery_volume_ratio"] = safe_ratio(
        core_vol, periphery_vol
    )

    # near-branching enrichment: (near bifurcation rate) / (global bifurcation rate)
    near_bif = _sum_finite(
        row.get("per_shell_topology_inside_tumor_bifurcation_count"),
        row.get("per_shell_topology_shell_0_2mm_bifurcation_count"),
    )
    near_nodes = _sum_finite(
        row.get("per_shell_topology_inside_tumor_node_count"),
        row.get("per_shell_topology_shell_0_2mm_node_count"),
    )
    total_nodes = safe_float(row.get("graph_totals_node_count"))
    total_bif = safe_float(row.get("morph_bifurcation_count"))
    near_rate = safe_ratio(near_bif, near_nodes)
    global_rate = safe_ratio(total_bif, total_nodes)
    row["graph_near_branching_bias"] = safe_ratio(near_rate, global_rate)

    row["graph_crossing_fraction_of_near_burden"] = safe_ratio(
        row.get("boundary_crossing_crossing_length_mm"),
        row.get("tumor_burden_inside_or_near_length_mm"),
    )

    # ------------------------------------------------------------------
    # kinematic block
    # ------------------------------------------------------------------
    _tte_prefix = "kinematic_shell_kinetics_{shell}_time_to_enhancement_hurdle_value_given_signal_median"
    _peak_prefix = "kinematic_shell_kinetics_{shell}_peak_enhancement_hurdle_value_given_signal_median"
    _washin_prefix = "kinematic_shell_kinetics_{shell}_washin_slope_hurdle_value_given_signal_median"
    _washout_prefix = "kinematic_shell_kinetics_{shell}_washout_slope_hurdle_value_given_signal_median"

    core_tte = _mean_finite(
        row.get(_tte_prefix.format(shell="inside_tumor")),
        row.get(_tte_prefix.format(shell="shell_0_2mm")),
    )
    periphery_tte = _mean_finite(
        row.get(_tte_prefix.format(shell="shell_5_10mm")),
        row.get(_tte_prefix.format(shell="shell_10_20mm")),
    )
    row["kinematic_core_to_periphery_tte_delta"] = (
        float(core_tte - periphery_tte)
        if np.isfinite(core_tte) and np.isfinite(periphery_tte)
        else np.nan
    )

    core_peak = _mean_finite(
        row.get(_peak_prefix.format(shell="inside_tumor")),
        row.get(_peak_prefix.format(shell="shell_0_2mm")),
    )
    periphery_peak = _mean_finite(
        row.get(_peak_prefix.format(shell="shell_5_10mm")),
        row.get(_peak_prefix.format(shell="shell_10_20mm")),
    )
    row["kinematic_core_to_periphery_peak_ratio"] = safe_ratio(
        core_peak, periphery_peak
    )

    near_washin = _mean_finite(
        row.get(_washin_prefix.format(shell="inside_tumor")),
        row.get(_washin_prefix.format(shell="shell_0_2mm")),
    )
    near_washout = _mean_finite(
        row.get(_washout_prefix.format(shell="inside_tumor")),
        row.get(_washout_prefix.format(shell="shell_0_2mm")),
    )
    row["kinematic_near_washin_to_washout_ratio"] = _abs_ratio(
        near_washin, near_washout
    )

    # crossing-specific early-enhancing fraction (requires updated kinematic.py)
    crossing_frac = safe_float(
        row.get("kinematic_early_enhancing_fraction_count_fraction_crossing")
    )
    row["kinematic_crossing_early_fraction"] = (
        float(crossing_frac)
        if crossing_frac is not None and np.isfinite(crossing_frac)
        else np.nan
    )

    # arrival delay CV for near-tumor segments
    delay_sd = safe_float(
        row.get(
            "kinematic_arrival_delay_vs_reference_near_tumor_segments_hurdle_value_given_signal_sd"
        )
    )
    delay_mean = safe_float(
        row.get(
            "kinematic_arrival_delay_vs_reference_near_tumor_segments_hurdle_value_given_signal_mean"
        )
    )
    row["kinematic_arrival_delay_dispersion_near"] = _abs_ratio(
        delay_sd if delay_sd is not None else np.nan,
        delay_mean if delay_mean is not None else np.nan,
    )
