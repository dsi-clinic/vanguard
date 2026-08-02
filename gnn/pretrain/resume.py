"""Exact, atomic epoch-level resume primitives for pretraining jobs."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

_FORMAT_VERSION = 1
_FIRST_RESUMABLE_EPOCH = 2


@dataclass(frozen=True)
class ResumedEpochState:
    """Training metadata restored alongside model, optimizer, and RNG state."""

    next_epoch: int
    history: list[dict[str, float]]
    best_state_dict: dict[str, torch.Tensor]
    best_metric: float
    best_epoch: int
    best_metadata: dict[str, Any]


def sha256(path: str | Path) -> str:
    """Return one artifact's SHA-256 digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_run_fingerprint(resolved_config: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize a resolved run contract and bind it to one digest."""
    try:
        canonical = json.dumps(
            resolved_config,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("resolved resume configuration must be finite JSON") from error
    normalized = json.loads(canonical)
    return {
        "format_version": _FORMAT_VERSION,
        "sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "resolved_config": normalized,
    }


def validate_run_fingerprint(
    observed: dict[str, Any], expected: dict[str, Any]
) -> None:
    """Fail unless a saved fingerprint exactly matches the current run."""
    if observed != expected:
        raise ValueError(
            "resume fingerprint mismatch: saved run configuration does not exactly "
            "match the current run"
        )
    rebuilt = make_run_fingerprint(dict(observed.get("resolved_config", {})))
    if rebuilt != observed:
        raise ValueError("saved resume fingerprint is internally inconsistent")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_torch_save(payload: dict[str, Any], path: str | Path) -> Path:
    """Durably replace a torch artifact without exposing a partial new file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def atomic_json_save(payload: dict[str, Any], path: str | Path) -> Path:
    """Durably replace a JSON artifact without exposing a partial new file."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with temporary.open("w") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def save_epoch_progress(
    path: str | Path,
    *,
    fingerprint: dict[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    next_epoch: int,
    history: list[dict[str, float]],
    best_state_dict: dict[str, torch.Tensor],
    best_metric: float,
    best_epoch: int,
    best_metadata: dict[str, Any],
    sampling_generator: torch.Generator,
    device: torch.device,
) -> Path:
    """Atomically save every state required to repeat the next epoch exactly."""
    if next_epoch < _FIRST_RESUMABLE_EPOCH:
        raise ValueError("epoch progress can only be saved after a complete epoch")
    if best_epoch < 1 or best_epoch >= next_epoch:
        raise ValueError("best_epoch must refer to a completed epoch")
    validate_run_fingerprint(fingerprint, fingerprint)
    cuda_states = (
        torch.cuda.get_rng_state_all() if device.type == "cuda" else []
    )
    payload = {
        "format_version": _FORMAT_VERSION,
        "fingerprint": fingerprint,
        "rng_device_type": device.type,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "next_epoch": int(next_epoch),
        "history": list(history),
        "best_state_dict": best_state_dict,
        "best_metric": float(best_metric),
        "best_epoch": int(best_epoch),
        "best_metadata": dict(best_metadata),
        "sampling_generator_state": sampling_generator.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_states": cuda_states,
    }
    return atomic_torch_save(payload, path)


def load_epoch_progress(
    path: str | Path,
    *,
    expected_fingerprint: dict[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    sampling_generator: torch.Generator,
    device: torch.device,
) -> ResumedEpochState | None:
    """Restore exact epoch-boundary state, or return ``None`` when absent."""
    source = Path(path)
    if not source.exists():
        return None
    payload = torch.load(source, map_location=device)
    required = {
        "format_version",
        "fingerprint",
        "rng_device_type",
        "model_state_dict",
        "optimizer_state_dict",
        "next_epoch",
        "history",
        "best_state_dict",
        "best_metric",
        "best_epoch",
        "best_metadata",
        "sampling_generator_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_states",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"epoch progress is missing fields: {sorted(missing)}")
    if payload["format_version"] != _FORMAT_VERSION:
        raise ValueError("unsupported epoch-progress format version")
    validate_run_fingerprint(payload["fingerprint"], expected_fingerprint)
    if payload["rng_device_type"] != device.type:
        raise ValueError("resume RNG device type does not match current device")

    next_epoch = int(payload["next_epoch"])
    best_epoch = int(payload["best_epoch"])
    history = list(payload["history"])
    if next_epoch < _FIRST_RESUMABLE_EPOCH or len(history) != next_epoch - 1:
        raise ValueError("epoch progress history does not match next_epoch")
    if best_epoch < 1 or best_epoch >= next_epoch:
        raise ValueError("epoch progress best_epoch is invalid")

    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    sampling_generator.set_state(payload["sampling_generator_state"].cpu())
    torch.set_rng_state(payload["torch_cpu_rng_state"].cpu())
    cuda_states = payload["torch_cuda_rng_states"]
    if device.type == "cuda":
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError("saved CUDA RNG state count does not match visible devices")
        torch.cuda.set_rng_state_all(cuda_states)
    elif cuda_states:
        raise ValueError("CPU resume unexpectedly contains CUDA RNG states")

    return ResumedEpochState(
        next_epoch=next_epoch,
        history=history,
        best_state_dict=payload["best_state_dict"],
        best_metric=float(payload["best_metric"]),
        best_epoch=best_epoch,
        best_metadata=dict(payload["best_metadata"]),
    )


def write_arm_completion(
    path: str | Path,
    *,
    fingerprint: dict[str, Any],
    checkpoint_path: str | Path,
    result: dict[str, Any],
) -> Path:
    """Write the last marker for one fully materialized training arm."""
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    validate_run_fingerprint(fingerprint, fingerprint)
    return atomic_json_save(
        {
            "format_version": _FORMAT_VERSION,
            "status": "complete",
            "fingerprint": fingerprint,
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "result": result,
        },
        path,
    )


def load_arm_completion(
    path: str | Path,
    *,
    expected_fingerprint: dict[str, Any],
    checkpoint_path: str | Path,
    checkpoint_validator: Callable[[Path], None] | None = None,
) -> dict[str, Any] | None:
    """Return a completed arm only after validating its exact checkpoint."""
    marker = Path(path)
    if not marker.exists():
        return None
    payload = json.loads(marker.read_text())
    required = {
        "format_version",
        "status",
        "fingerprint",
        "checkpoint_path",
        "checkpoint_sha256",
        "result",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"arm completion is missing fields: {sorted(missing)}")
    if payload["format_version"] != _FORMAT_VERSION or payload["status"] != "complete":
        raise ValueError("arm completion marker is invalid")
    validate_run_fingerprint(payload["fingerprint"], expected_fingerprint)
    checkpoint = Path(checkpoint_path)
    if Path(payload["checkpoint_path"]) != checkpoint:
        raise ValueError("arm completion checkpoint path does not match")
    if not checkpoint.is_file() or sha256(checkpoint) != payload["checkpoint_sha256"]:
        raise ValueError("arm completion checkpoint is missing or changed")
    if checkpoint_validator is not None:
        checkpoint_validator(checkpoint)
    result = payload["result"]
    if not isinstance(result, dict):
        raise ValueError("arm completion result must be a mapping")
    return result


def write_run_completion(
    path: str | Path,
    *,
    fingerprint: dict[str, Any],
    artifacts: list[str | Path],
) -> Path:
    """Write the final run sentinel only after every required artifact exists."""
    validate_run_fingerprint(fingerprint, fingerprint)
    checksums = {}
    for artifact_value in artifacts:
        artifact = Path(artifact_value)
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
        checksums[str(artifact)] = sha256(artifact)
    return atomic_json_save(
        {
            "format_version": _FORMAT_VERSION,
            "status": "complete",
            "fingerprint": fingerprint,
            "artifacts": checksums,
        },
        path,
    )


def validate_run_completion(
    path: str | Path, *, expected_fingerprint: dict[str, Any]
) -> bool:
    """Validate the final sentinel and every bound artifact when present."""
    marker = Path(path)
    if not marker.exists():
        return False
    payload = json.loads(marker.read_text())
    if (
        payload.get("format_version") != _FORMAT_VERSION
        or payload.get("status") != "complete"
    ):
        raise ValueError("run completion marker is invalid")
    validate_run_fingerprint(payload.get("fingerprint", {}), expected_fingerprint)
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("run completion has no bound artifacts")
    for path_text, expected_sha in artifacts.items():
        artifact = Path(path_text)
        if not artifact.is_file() or sha256(artifact) != expected_sha:
            raise ValueError(f"completed run artifact is missing or changed: {artifact}")
    return True
