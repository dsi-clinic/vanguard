"""Per-dataset adapters (Step 1 scaffold).

One class per dataset shape: ``MamaMiaDataset`` (parameterized by cohort) and
``UChicagoDataset``, both subclassing :class:`DatasetAdapter`. A factory builds
the right one from run config. Nothing in the pipeline stages calls these yet —
see ``docs/modularization-design.md``.
"""

from __future__ import annotations

from datasets.base import DatasetAdapter
from datasets.factory import build_adapter_from_config, resolve_split_policy
from datasets.mamamia import MamaMiaDataset
from datasets.uchicago import UChicagoDataset

__all__ = [
    "DatasetAdapter",
    "MamaMiaDataset",
    "UChicagoDataset",
    "build_adapter_from_config",
    "resolve_split_policy",
]
