"""Canonical definitions of the modeling feature blocks."""

from __future__ import annotations

from features import clinical, graph, kinematic, morph, tumor_size

ANNOTATION_COLUMNS = {
    "case_id",
    "dataset",
    "site",
    "bilateral",
    "tumor_subtype",
    "has_centerline_file",
}
FEATURE_BLOCK_ORDER = (
    clinical.BLOCK_NAME,
    tumor_size.BLOCK_NAME,
    morph.BLOCK_NAME,
    graph.BLOCK_NAME,
    kinematic.BLOCK_NAME,
)
FEATURE_BLOCK_DESCRIPTIONS = {
    "clinical": "non-imaging patient and tumor metadata",
    "tumor_size": "tumor size and peritumoral shell-size summaries from the tumor mask",
    "morph": "whole-network morphometry aggregates from the centerline graph",
    "graph": "tumor-centered structural vessel features, including burden, topology, and shape",
    "kinematic": "tumor-centered dynamic vessel features over time",
}


def normalize_selected_features(value: object) -> list[str] | None:
    """Normalize config feature-block selection into canonical ordered names."""
    if value is None:
        return None
    if isinstance(value, str):
        raw_blocks = [value]
    elif isinstance(value, (list, tuple, set)):
        raw_blocks = [str(v) for v in value]
    else:
        raise ValueError(
            "selected_features must be a string, list, tuple, or set."
        )

    normalized = {
        str(block).strip().lower() for block in raw_blocks if str(block).strip()
    }
    if not normalized:
        return None

    invalid = sorted(normalized.difference(FEATURE_BLOCK_ORDER))
    if invalid:
        raise ValueError(
            "Unknown selected_features entries: "
            f"{invalid}. Valid blocks are {list(FEATURE_BLOCK_ORDER)}."
        )

    return [block for block in FEATURE_BLOCK_ORDER if block in normalized]


def feature_block_for_column(column: str) -> str | None:
    """Map a feature column name to one canonical high-level block."""
    if clinical.matches_column(column):
        return clinical.BLOCK_NAME
    if tumor_size.matches_column(column):
        return tumor_size.BLOCK_NAME
    if morph.matches_column(column):
        return morph.BLOCK_NAME
    if kinematic.matches_column(column):
        return kinematic.BLOCK_NAME
    if graph.matches_column(column):
        return graph.BLOCK_NAME
    return None
