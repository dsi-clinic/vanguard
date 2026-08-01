"""Strict encoder-only checkpoints for temporal weight transfer."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import torch
from torch import nn


def _git_commit() -> str:
    result = subprocess.run(  # noqa: S603
        ["git", "rev-parse", "HEAD"],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def save_encoder_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    preprocessing_contract: dict[str, Any],
    data_manifest: str | Path,
    epoch: int,
    validation_mae: float,
) -> None:
    """Save only the reusable encoder and its complete construction contract."""
    encoder = getattr(model, "encoder", None)
    if encoder is None or not hasattr(encoder, "config"):
        raise TypeError("model must expose an encoder with a config() method")
    required_contract = {
        "baseline_frame_count_source",
        "enhancement",
        "baseline_imputation",
        "validity_mask_policy",
        "time_anchor",
    }
    missing = required_contract - set(preprocessing_contract)
    if missing:
        raise ValueError(
            f"preprocessing_contract is missing required fields: {sorted(missing)}"
        )
    payload = {
        "encoder_state_dict": encoder.state_dict(),
        "encoder_config": encoder.config(),
        "preprocessing_contract": dict(preprocessing_contract),
        "git_commit": _git_commit(),
        "data_manifest": str(data_manifest),
        "epoch": int(epoch),
        "validation_mae": float(validation_mae),
    }
    torch.save(payload, Path(path))


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a checkpoint and fail if any weight-transfer field is absent."""
    checkpoint = torch.load(Path(path), map_location="cpu")
    required = {
        "encoder_state_dict",
        "encoder_config",
        "preprocessing_contract",
        "git_commit",
        "data_manifest",
        "epoch",
        "validation_mae",
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"encoder checkpoint is missing fields: {sorted(missing)}")
    return checkpoint


DEFAULT_PREPROCESSING_CONTRACT = {
    "baseline_frame_count_source": "run_summary.kinetic_feature_policy",
    "enhancement": "(S-S0_imputed)/S0_imputed",
    "baseline_imputation": "simultaneous_iterative_neighbor_median",
    "validity_mask_policy": "finite_S0_and_S0_gt_0.05_case_median_S0",
    "time_anchor": "minutes_since_last_baseline_acquisition",
}
