"""Disk cache for expensive per-case Deep Sets build intermediates."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

CACHE_VERSION = 1
CACHE_META_NAME = "meta.json"
CACHE_ARRAYS_NAME = "arrays.npz"


def resolve_cache_root(
    *,
    output_dir: Path,
    cache_root: Path | None = None,
) -> Path | None:
    """Resolve cache directory from explicit path or environment."""
    if cache_root is not None:
        return cache_root
    env_root = os.environ.get("DEEPSETS_CACHE_ROOT", "").strip()
    if env_root:
        return Path(env_root)
    if os.environ.get("DEEPSETS_CACHE_DISABLE", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        return None
    return output_dir.parent / ".deepsets_build_cache"


def _source_fingerprint(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.resolve()).encode("utf-8"))
        if path.exists():
            stat = path.stat()
            digest.update(str(int(stat.st_mtime_ns)).encode("utf-8"))
            digest.update(str(int(stat.st_size)).encode("utf-8"))
        else:
            digest.update(b"missing")
    return digest.hexdigest()


def cache_dir_for_case(
    cache_root: Path,
    *,
    case_id: str,
    dataset_name: str,
    source_paths: list[Path],
) -> Path:
    """Return the on-disk cache directory for one case."""
    fingerprint = _source_fingerprint(source_paths)
    safe_dataset = dataset_name.replace("/", "_")
    return cache_root / safe_dataset / case_id / fingerprint


def load_case_cache(cache_dir: Path) -> dict[str, Any] | None:
    """Load cached arrays and metadata when present and version-compatible."""
    meta_path = cache_dir / CACHE_META_NAME
    arrays_path = cache_dir / CACHE_ARRAYS_NAME
    if not meta_path.is_file() or not arrays_path.is_file():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if int(meta.get("cache_version", -1)) != CACHE_VERSION:
        return None
    with np.load(arrays_path, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    return {**meta, **arrays}


def save_case_cache(cache_dir: Path, payload: dict[str, Any]) -> None:
    """Persist cached arrays and metadata for one case."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        key: value
        for key, value in payload.items()
        if not isinstance(value, np.ndarray)
    }
    meta["cache_version"] = CACHE_VERSION
    arrays = {
        key: value for key, value in payload.items() if isinstance(value, np.ndarray)
    }
    (cache_dir / CACHE_META_NAME).write_text(
        json.dumps(meta, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    np.savez_compressed(cache_dir / CACHE_ARRAYS_NAME, **arrays)
