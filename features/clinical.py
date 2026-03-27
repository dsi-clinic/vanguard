"""Definitions for the clinical feature block."""

from __future__ import annotations

BLOCK_NAME = "clinical"

FEATURE_COLUMNS = {
    "age",
    "menopausal_status",
    "menopause",
    "breast_density",
    "tumor_subtype",
    "dataset",
    "site",
    "bilateral",
    "field_strength",
    "echo_time",
    "repetition_time",
    "scanner_manufacturer",
    "scanner_model",
    "hr",
    "er",
    "pr",
    "her2",
    "nottingham_grade",
    "bmi_group",
}

SITE_COLUMNS = [
    "dataset",
    "site",
    "scanner_manufacturer",
    "scanner_model",
    "field_strength",
    "echo_time",
    "repetition_time",
    "bilateral",
]


def matches_column(column: str) -> bool:
    """Return whether a column belongs to the clinical block."""
    return column in FEATURE_COLUMNS
