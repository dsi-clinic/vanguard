"""Per-dataset adapters.

One class per dataset shape: ``MamaMiaDataset`` (parameterized by cohort) and
``UChicagoDataset``, both subclassing :class:`DatasetAdapter`. A factory builds
the right one from run config. Every pipeline stage requires a configured
adapter (multi-dataset migration Step 5) — see ``cohorts/README.md``.
"""

from __future__ import annotations

from cohorts.base import DatasetAdapter
from cohorts.factory import (
    build_adapter_from_config,
    build_imaging_adapter_from_config,
    require_adapter_from_config,
    require_imaging_adapter_from_config,
    resolve_folds,
    resolve_split_policy,
)
from cohorts.mamamia import MamaMiaDataset
from cohorts.uchicago import UChicagoDataset

__all__ = [
    "DatasetAdapter",
    "MamaMiaDataset",
    "UChicagoDataset",
    "build_adapter_from_config",
    "build_imaging_adapter_from_config",
    "require_adapter_from_config",
    "require_imaging_adapter_from_config",
    "resolve_folds",
    "resolve_split_policy",
]
