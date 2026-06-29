"""Shared timing helpers for Deep Sets build, merge, and train stages."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def stage_timer(stage: str, **fields: object) -> Iterator[None]:
    """Log wall time for a pipeline stage with optional structured fields."""
    started = time.perf_counter()
    field_suffix = " ".join(f"{key}={value}" for key, value in fields.items())
    if field_suffix:
        logging.info("Deep Sets %s started %s", stage, field_suffix)
    else:
        logging.info("Deep Sets %s started", stage)
    try:
        yield
    finally:
        elapsed_sec = time.perf_counter() - started
        if field_suffix:
            logging.info(
                "Deep Sets %s finished elapsed_sec=%.3f %s",
                stage,
                elapsed_sec,
                field_suffix,
            )
        else:
            logging.info(
                "Deep Sets %s finished elapsed_sec=%.3f",
                stage,
                elapsed_sec,
            )


def log_case_timing(
    *,
    case_id: str,
    elapsed_sec: float,
    shard_index: int,
    num_shards: int,
    point_feature_set: str,
    threshold_sec: float,
) -> None:
    """Log per-case timing when a case exceeds the configured threshold."""
    if elapsed_sec < threshold_sec:
        return
    logging.info(
        "Deep Sets build case timing case_id=%s elapsed_sec=%.3f shard=%d/%d "
        "point_feature_set=%s",
        case_id,
        elapsed_sec,
        shard_index,
        num_shards,
        point_feature_set,
    )
