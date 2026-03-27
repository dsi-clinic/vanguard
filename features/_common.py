"""Shared helpers for feature-block extraction."""

from __future__ import annotations

from typing import Any

import numpy as np


def safe_float(value: Any) -> float | None:
    """Convert values to float when possible, otherwise return `None`."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except Exception:  # noqa: BLE001
            return None
    return None


def safe_ratio(numerator: Any, denominator: Any) -> float:
    """Safely compute `numerator / denominator` with missing/zero protection."""
    num = safe_float(numerator)
    den = safe_float(denominator)
    if num is None or den is None or not np.isfinite(num) or not np.isfinite(den):
        return np.nan
    if den <= 0:
        return np.nan
    return float(num / den)


def sanitize_feature_token(token: str) -> str:
    """Normalize nested JSON keys into stable, safe feature tokens."""
    cleaned = []
    for char in str(token):
        if char.isalnum():
            cleaned.append(char.lower())
        else:
            cleaned.append("_")
    out = "".join(cleaned).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out or "x"


def flatten_numeric_payload(value: Any, prefix: str, out: dict[str, float]) -> None:
    """Flatten nested dict/list payloads into numeric scalar features."""
    if isinstance(value, dict):
        for key, child in value.items():
            token = sanitize_feature_token(str(key))
            if token in {
                "status",
                "version",
                "name",
                "path",
                "mask_dir",
                "layout",
                "reference_curve_source",
                "time_series_source",
                "reference_mask_source",
            }:
                continue
            child_prefix = f"{prefix}_{token}" if prefix else token
            flatten_numeric_payload(child, child_prefix, out)
        return

    if isinstance(value, (list, tuple)):
        numeric_values: list[float] = []
        for item in value:
            maybe = safe_float(item)
            if maybe is None or not np.isfinite(maybe):
                continue
            numeric_values.append(float(maybe))

        if numeric_values:
            arr = np.asarray(numeric_values, dtype=float)
            out[f"{prefix}_n"] = float(arr.size)
            out[f"{prefix}_mean"] = float(arr.mean())
            out[f"{prefix}_std"] = float(arr.std())
            out[f"{prefix}_min"] = float(arr.min())
            out[f"{prefix}_max"] = float(arr.max())
            out[f"{prefix}_q25"] = float(np.quantile(arr, 0.25))
            out[f"{prefix}_q50"] = float(np.quantile(arr, 0.5))
            out[f"{prefix}_q75"] = float(np.quantile(arr, 0.75))
        return

    maybe = safe_float(value)
    if maybe is None or not np.isfinite(maybe):
        return
    out[prefix] = float(maybe)
