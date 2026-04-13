"""Helpers for loading serialized Deep Sets case tensors."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pandas as pd
import torch

REQUIRED_DEEPSETS_MANIFEST_COLUMNS = (
    "case_id",
    "set_path",
)
FEATURE_MATRIX_NDIM = 2
MIN_STD = 1e-6


class SavedSetLookup:
    """Lazy loader for per-case Deep Sets inputs stored on disk."""

    def __init__(self, manifest_df: pd.DataFrame) -> None:
        missing = [
            column
            for column in REQUIRED_DEEPSETS_MANIFEST_COLUMNS
            if column not in manifest_df.columns
        ]
        if missing:
            raise ValueError(
                f"Deep Sets manifest is missing required columns: {missing}"
            )
        self._manifest = manifest_df.copy()
        self._manifest["case_id"] = self._manifest["case_id"].astype(str)
        self._path_by_case = {
            str(row.case_id): Path(row.set_path)
            for row in self._manifest.itertuples(index=False)
        }
        self._cache: dict[str, dict[str, Any]] = {}

    def get(self, case_id: str) -> dict[str, Any]:
        """Load one case tensor package by case ID."""
        case_key = str(case_id)
        if case_key not in self._cache:
            _load_kw: dict[str, Any] = {"map_location": "cpu"}
            if "weights_only" in inspect.signature(torch.load).parameters:
                _load_kw["weights_only"] = False
            payload = torch.load(self._path_by_case[case_key], **_load_kw)
            if not isinstance(payload, dict):
                raise TypeError(f"Expected dict payload for case {case_key}")
            self._cache[case_key] = payload
        cached = self._cache[case_key]
        cloned: dict[str, Any] = {}
        for key, value in cached.items():
            cloned[key] = value.clone() if isinstance(value, torch.Tensor) else value
        return cloned

    def subset(self, case_ids: list[str]) -> list[dict[str, Any]]:
        """Return cloned case tensor packages for a list of case IDs."""
        return [self.get(case_id) for case_id in case_ids]


def collate_case_sets(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate variable-length point sets into one batch for Deep Sets."""
    if not items:
        raise ValueError("Cannot collate an empty batch")
    x_chunks: list[torch.Tensor] = []
    batch_index_chunks: list[torch.Tensor] = []
    y_chunks: list[torch.Tensor] = []
    num_points_chunks: list[torch.Tensor] = []
    case_ids: list[str] = []
    for batch_idx, item in enumerate(items):
        x = item["x"]
        if x.ndim != FEATURE_MATRIX_NDIM:
            raise ValueError(
                "Each case tensor must have shape [num_points, num_features]"
            )
        n_points = int(x.shape[0])
        x_chunks.append(x)
        batch_index_chunks.append(torch.full((n_points,), batch_idx, dtype=torch.long))
        y_chunks.append(item["y"].view(1).float())
        num_points_chunks.append(torch.tensor([float(n_points)], dtype=torch.float32))
        case_ids.append(str(item["case_id"]))
    return {
        "x": torch.cat(x_chunks, dim=0),
        "batch_index": torch.cat(batch_index_chunks, dim=0),
        "y": torch.cat(y_chunks, dim=0),
        "num_points": torch.cat(num_points_chunks, dim=0),
        "case_ids": case_ids,
        "feature_names": list(items[0].get("feature_names", [])),
    }


def fit_feature_standardizer(
    items: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-feature mean and std from the training point tensors."""
    if not items:
        raise ValueError("Cannot fit a feature standardizer on an empty item list")
    feature_matrix = torch.cat([item["x"] for item in items], dim=0)
    mean = feature_matrix.mean(dim=0)
    std = feature_matrix.std(dim=0, unbiased=False)
    std = torch.where(std < MIN_STD, torch.ones_like(std), std)
    return mean, std


def apply_feature_standardizer(
    items: list[dict[str, Any]],
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> list[dict[str, Any]]:
    """Return cloned case tensors with standardized point features."""
    standardized_items: list[dict[str, Any]] = []
    for item in items:
        cloned = {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in item.items()
        }
        cloned["x"] = (cloned["x"] - mean) / std
        standardized_items.append(cloned)
    return standardized_items
