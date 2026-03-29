"""Definitions and extraction helpers for the morphometry feature block."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from features._common import safe_float

BLOCK_NAME = "morph"
MAX_BIFURCATION_ANGLE_DEGREES = 180.0


def matches_column(column: str) -> bool:
    """Return whether a column belongs to the morphometry block."""
    return column.startswith("morph_")


def array_stats(prefix: str, values: list[float]) -> dict[str, float]:
    """Compute compact summary stats for a list of numeric values."""
    if not values:
        return {
            f"{prefix}_n": 0.0,
            f"{prefix}_sum": np.nan,
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_max": np.nan,
        }

    arr = np.asarray(values, dtype=float)
    return {
        f"{prefix}_n": float(arr.size),
        f"{prefix}_sum": float(arr.sum()),
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_std": float(arr.std()),
        f"{prefix}_max": float(arr.max()),
    }


def extract_morphometry_features(morphometry_path: Path) -> dict[str, float]:
    """Extract per-study aggregate features from the morphometry JSON."""
    morph = json.loads(morphometry_path.read_text())

    seg_length: list[float] = []
    seg_tortuosity: list[float] = []
    seg_volume: list[float] = []
    curvature_mean: list[float] = []
    radius_mean: list[float] = []
    bif_angle: list[float] = []
    bifurcation_count = 0
    raw_segment_count = 0
    unique_segment_count = 0
    seen_segments: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    invalid_counts = {
        "seg_length_invalid_n": 0.0,
        "seg_tortuosity_invalid_n": 0.0,
        "seg_volume_invalid_n": 0.0,
        "curvature_mean_invalid_n": 0.0,
        "radius_mean_invalid_n": 0.0,
        "bif_angle_invalid_n": 0.0,
    }

    for component in morph.values():
        if not isinstance(component, dict):
            continue

        for name, entries in component.items():
            if not isinstance(entries, list):
                continue

            if "bifurcation" in name.lower():
                bifurcation_count += len(entries)
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    angles = entry.get("angles", {})
                    if not isinstance(angles, dict):
                        continue
                    for key in ("pair1", "pair2", "pair3"):
                        maybe = safe_float(angles.get(key))
                        if (
                            maybe is not None
                            and np.isfinite(maybe)
                            and 0.0 < maybe <= MAX_BIFURCATION_ANGLE_DEGREES
                        ):
                            bif_angle.append(maybe)
                        elif maybe is not None:
                            invalid_counts["bif_angle_invalid_n"] += 1.0
            else:
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue

                    raw_segment_count += 1
                    keep_entry = True
                    segment_block = entry.get("segment", {})
                    if isinstance(segment_block, dict):
                        start = segment_block.get("start")
                        end = segment_block.get("end")
                        if isinstance(start, list) and isinstance(end, list):
                            try:
                                start_tuple = tuple(int(v) for v in start)
                                end_tuple = tuple(int(v) for v in end)
                                key = tuple(sorted((start_tuple, end_tuple)))
                                if key in seen_segments:
                                    keep_entry = False
                                else:
                                    seen_segments.add(key)
                            except Exception:  # noqa: BLE001
                                keep_entry = True

                    if not keep_entry:
                        continue

                    unique_segment_count += 1

                    maybe = safe_float(entry.get("length"))
                    if maybe is not None and np.isfinite(maybe) and maybe > 0.0:
                        seg_length.append(maybe)
                    elif maybe is not None:
                        invalid_counts["seg_length_invalid_n"] += 1.0

                    maybe = safe_float(entry.get("tortuosity"))
                    if maybe is not None and np.isfinite(maybe) and maybe >= 1.0:
                        seg_tortuosity.append(maybe)
                    elif maybe is not None:
                        invalid_counts["seg_tortuosity_invalid_n"] += 1.0

                    maybe = safe_float(entry.get("volume"))
                    if maybe is not None and np.isfinite(maybe) and maybe > 0.0:
                        seg_volume.append(maybe)
                    elif maybe is not None:
                        invalid_counts["seg_volume_invalid_n"] += 1.0

                    curvature = entry.get("curvature", {})
                    if isinstance(curvature, dict):
                        maybe = safe_float(curvature.get("mean"))
                        if maybe is not None and np.isfinite(maybe) and maybe >= 0.0:
                            curvature_mean.append(maybe)
                        elif maybe is not None:
                            invalid_counts["curvature_mean_invalid_n"] += 1.0

                    radius = entry.get("radius", {})
                    if isinstance(radius, dict):
                        maybe = safe_float(radius.get("mean"))
                        if maybe is not None and np.isfinite(maybe) and maybe > 0.0:
                            radius_mean.append(maybe)
                        elif maybe is not None:
                            invalid_counts["radius_mean_invalid_n"] += 1.0

    raw_features: dict[str, float] = {
        "bifurcation_count": float(bifurcation_count),
        "seg_raw_n": float(raw_segment_count),
        "seg_unique_n": float(unique_segment_count),
        "seg_dup_fraction": (
            float((raw_segment_count - unique_segment_count) / raw_segment_count)
            if raw_segment_count > 0
            else np.nan
        ),
    }
    raw_features.update(invalid_counts)
    raw_features.update(array_stats("seg_length", seg_length))
    raw_features.update(array_stats("seg_tortuosity", seg_tortuosity))
    raw_features.update(array_stats("seg_volume", seg_volume))
    raw_features.update(array_stats("curvature_mean", curvature_mean))
    raw_features.update(array_stats("radius_mean", radius_mean))
    raw_features.update(array_stats("bif_angle", bif_angle))

    features: dict[str, float] = {}
    for key, value in raw_features.items():
        features[f"morph_{key}"] = float(value)
    return features
