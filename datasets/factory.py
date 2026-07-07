"""Build the right dataset adapter from run config, and resolve split policy.

These are the only two seams that read run config: the factory turns the
``dataset`` config block into a concrete adapter, and :func:`resolve_split_policy`
applies the run-config override on top of the adapter's default (design doc §10
decisions 3 and 5).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from datasets.base import DatasetAdapter
from datasets.mamamia import MamaMiaDataset
from datasets.uchicago import UChicagoDataset

if TYPE_CHECKING:
    from config import ConfigNode

VALID_SPLIT_POLICIES: frozenset[str] = frozenset({"auto", "compute", "provided"})


def build_adapter_from_config(config: ConfigNode) -> DatasetAdapter | None:
    """Build the dataset adapter selected by run config.

    Reads ``config.dataset.{name, cohort, root}``. Returns ``None`` when no
    dataset is configured, so callers can fall back to today's behavior — Step 1
    wires nothing (design doc §9).

    Args:
        config: A loaded run config (see ``config.DEFAULT_CONFIG``).

    Returns:
        The selected adapter, or ``None`` if ``dataset.name`` is unset.

    Raises:
        ValueError: If ``dataset.name`` is set but not recognized.
    """
    dataset_cfg = config.dataset
    name = dataset_cfg.name
    if not name:
        return None

    root = Path(dataset_cfg.root)
    key = str(name).strip().lower()
    if key == "mamamia":
        return MamaMiaDataset(cohort=dataset_cfg.cohort, root=root)
    if key == "uchicago":
        return UChicagoDataset(root=root)
    raise ValueError(
        f"Unknown dataset name {name!r}; expected 'mamamia' or 'uchicago'."
    )


def resolve_split_policy(config: ConfigNode, adapter: DatasetAdapter) -> str:
    """Resolve the effective split policy: the run-config knob overrides the default.

    ``config.dataset.split_policy`` is ``"auto"``, ``"compute"``, or
    ``"provided"``. ``"auto"`` defers to ``adapter.default_split_policy``; an
    explicit value overrides it (design doc §10 decision 3), so a run can force
    ``"compute"`` even for a dataset that ships folds.

    Args:
        config: A loaded run config.
        adapter: The dataset adapter whose default applies under ``"auto"``.

    Returns:
        Either ``"compute"`` or ``"provided"``.

    Raises:
        ValueError: If the configured policy is not a valid value.
    """
    policy = str(config.dataset.split_policy).strip().lower()
    if policy not in VALID_SPLIT_POLICIES:
        raise ValueError(
            f"Invalid split_policy {policy!r}; expected one of "
            f"{sorted(VALID_SPLIT_POLICIES)}."
        )
    if policy == "auto":
        return adapter.default_split_policy
    return policy
