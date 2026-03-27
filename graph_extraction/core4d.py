"""Shared 4D study I/O and collapse helpers used by processing + debug scripts."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

DEFAULT_SEGMENTATION_DIR = Path("/net/projects2/vanguard/vessel_segmentations")
NDIM_3D = 3
NDIM_4D = 4


def extract_single_timepoint_volume(arr: np.ndarray) -> np.ndarray:
    """Extract one `(z, y, x)` probability volume from per-timepoint arrays."""
    if arr.ndim != NDIM_3D:
        raise ValueError(
            "Per-timepoint input must be 3D for the current segmentation format, "
            f"got shape {arr.shape}"
        )
    return arr.astype(np.float32, copy=False)


def load_numpy_array_from_path(path: Path) -> np.ndarray:
    """Load one array from current segmentation format (`.npz` with `vessel`)."""
    if path.suffix.lower() != ".npz":
        raise ValueError(
            "Unsupported segmentation file format. Expected `.npz` files only, "
            f"got: {path}"
        )

    loaded = np.load(path, allow_pickle=False)
    if not isinstance(loaded, np.lib.npyio.NpzFile):
        raise ValueError(
            f"Expected NPZ container at {path}, got {type(loaded).__name__}"
        )

    if "vessel" not in loaded.files:
        keys = list(loaded.files)
        loaded.close()
        raise ValueError(
            f"NPZ file must contain `vessel` array in current format: {path} keys={keys}"
        )

    arr = loaded["vessel"]
    loaded.close()
    return arr


def load_time_series_from_files(paths: list[Path]) -> np.ndarray:
    """Load and stack per-timepoint arrays into `(t, z, y, x)`."""
    volumes: list[np.ndarray] = []
    expected_shape: tuple[int, int, int] | None = None

    for path in paths:
        arr = load_numpy_array_from_path(path)
        vol = extract_single_timepoint_volume(arr)
        if expected_shape is None:
            expected_shape = tuple(int(x) for x in vol.shape)
        elif tuple(vol.shape) != expected_shape:
            raise ValueError(
                f"Shape mismatch: expected {expected_shape}, got {vol.shape} from {path}"
            )
        volumes.append(vol)

    if not volumes:
        raise ValueError("No input files provided.")
    return np.stack(volumes, axis=0).astype(np.float32, copy=False)


def discover_study_timepoints(
    input_dir: Path, case_id: str
) -> tuple[list[Path], list[int]]:
    """Discover and sort timepoint files for one case ID."""
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")

    study_dir = input_dir / case_id
    images_dir = study_dir / "images"
    search_dir = images_dir if images_dir.exists() else study_dir
    if not search_dir.exists() or not search_dir.is_dir():
        raise ValueError(
            "Expected study directory layout `<input-dir>/<case-id>/images` "
            f"for case_id='{case_id}', but not found under {input_dir}"
        )

    candidates = sorted(search_dir.glob(f"{case_id}_*_vessel_segmentation.npz"))
    if not candidates:
        raise ValueError(
            "No candidate segmentation files found for case_id "
            f"'{case_id}' in {search_dir}. Expected files like "
            f"`{case_id}_0000_vessel_segmentation.npz`"
        )

    patt = re.compile(
        rf"{re.escape(case_id)}_(\d{{4}})_vessel_segmentation\.npz$",
        flags=re.IGNORECASE,
    )

    timepoint_pairs: list[tuple[int, Path]] = []
    for path in candidates:
        match = patt.search(path.name)
        if match is not None:
            timepoint_pairs.append((int(match.group(1)), path))

    if not timepoint_pairs:
        example_names = ", ".join(p.name for p in candidates[:5])
        raise ValueError(
            "Found candidate files but none matched the expected timepoint pattern "
            f"for case_id='{case_id}'. First candidates: {example_names}"
        )

    seen: dict[int, Path] = {}
    duplicates: list[str] = []
    for tp, path in sorted(timepoint_pairs, key=lambda x: (x[0], x[1].name)):
        if tp in seen:
            duplicates.append(f"{tp:04d}: {seen[tp].name} | {path.name}")
        else:
            seen[tp] = path

    if duplicates:
        dup_msg = "; ".join(duplicates[:5])
        raise ValueError(
            "Duplicate files found for one or more timepoints. "
            f"Please resolve duplicates. Examples: {dup_msg}"
        )

    ordered = sorted(seen.items(), key=lambda kv: kv[0])
    return [p for _, p in ordered], [tp for tp, _ in ordered]
