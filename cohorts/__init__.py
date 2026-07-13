"""Per-dataset adapters (Step 1 scaffold).

One class per dataset shape: ``MamaMiaDataset`` (parameterized by cohort) and
``UChicagoDataset``, both subclassing :class:`DatasetAdapter`. A factory builds
the right one from run config. Nothing in the pipeline stages calls these yet —
see ``cohorts/README.md``.
"""

from __future__ import annotations

from cohorts.base import DatasetAdapter
from cohorts.factory import build_adapter_from_config, resolve_split_policy
from cohorts.mamamia import MamaMiaDataset
from cohorts.uchicago import UChicagoDataset

__all__ = [
    "DatasetAdapter",
    "MamaMiaDataset",
    "UChicagoDataset",
    "build_adapter_from_config",
    "resolve_split_policy",
]
