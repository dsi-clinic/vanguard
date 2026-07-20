"""Build the right dataset adapter from run config, and resolve split policy.

These are the only two seams that read run config: the factory turns the
``dataset`` config block into a concrete adapter, and :func:`resolve_split_policy`
applies the run-config override on top of the adapter's default (see cohorts/README.md).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from cohorts.base import DatasetAdapter
from cohorts.mamamia import MamaMiaDataset
from cohorts.uchicago import UChicagoDataset

if TYPE_CHECKING:
    import pandas as pd

    from config import ConfigNode

VALID_SPLIT_POLICIES: frozenset[str] = frozenset({"auto", "compute", "provided"})


def build_adapter_from_config(config: ConfigNode) -> DatasetAdapter | None:
    """Build the dataset adapter selected by run config.

    Reads ``config.dataset.{name, cohort, root}``. Returns ``None`` when no
    dataset is configured, so callers can fall back to today's behavior — Step 1
    wires nothing (see cohorts/README.md).

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


#: Datasets that must NOT be driven through this repo's NIfTI-based imaging
#: stages (``segmentation/batch_segmentation.py``, ``graph_extraction/
#: run_skeleton_processing.py``), mapped to the route that supersedes them.
#:
#: These stages take an already-converted NIfTI phase series and segment it
#: directly. For UChicago that route is obsolete: the supported pipeline starts
#: from the paired raw HR/UFAST DICOM archives, segments the *high-resolution*
#: phases, builds the skeleton from HR vessel probabilities, maps that static
#: skeleton onto the motion-corrected UFAST grid, and computes kinetics from
#: raw UFAST signal using physical timestamps. Segmenting the legacy UFAST
#: NIfTIs instead silently produces lower-resolution vessels and index-based
#: (not time-based) kinetics, so this fails closed rather than leaving two
#: competing implementations that look interchangeable from the CLI.
IMAGING_ROUTE_SUPERSEDED: dict[str, str] = {
    "uchicago": (
        "UChicago imaging no longer runs through this stage. Segmenting the "
        "legacy ultrafast NIfTI phases has been superseded by the paired "
        "raw-DICOM HR/UFAST pipeline, which segments the high-resolution "
        "phases, builds the skeleton from HR vessel probabilities, maps it "
        "onto the motion-corrected UFAST grid, and derives kinetics from raw "
        "UFAST signal using physical timestamps.\n"
        "Use instead:\n"
        "  python -m preprocessing.pipeline prepare|infer|tc4d|map \\\n"
        "      --exam-id <exam> --output-root <root>\n"
        "See preprocessing/README.md for the full contract and the Slurm "
        "wrappers in preprocessing/slurm/."
    )
}


def build_imaging_adapter_from_config(config: ConfigNode) -> DatasetAdapter | None:
    """Build an adapter for an *imaging* stage, refusing superseded datasets.

    Same as :func:`build_adapter_from_config`, plus the restriction that a
    dataset listed in :data:`IMAGING_ROUTE_SUPERSEDED` cannot be run through
    this repo's NIfTI-based segmentation/skeletonization CLIs. Non-imaging
    consumers (feature extraction, folds/grouping, QC ``report_by``) keep using
    :func:`build_adapter_from_config` and are unaffected -- the adapter itself
    is still fully supported for those, only the imaging *route* is retired.

    Raises:
        ValueError: If the selected dataset's imaging route has been superseded.
    """
    name = config.dataset.name
    if name:
        key = str(name).strip().lower()
        if key in IMAGING_ROUTE_SUPERSEDED:
            raise ValueError(IMAGING_ROUTE_SUPERSEDED[key])
    return build_adapter_from_config(config)


def resolve_split_policy(config: ConfigNode, adapter: DatasetAdapter) -> str:
    """Resolve the effective split policy: the run-config knob overrides the default.

    ``config.dataset.split_policy`` is ``"auto"``, ``"compute"``, or
    ``"provided"``. ``"auto"`` defers to ``adapter.default_split_policy``; an
    explicit value overrides it (see cohorts/README.md — Design decisions), so a run can force
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


def resolve_folds(config: ConfigNode, adapter: DatasetAdapter) -> pd.DataFrame | None:
    """Return the CV folds to use, honoring the resolved split policy.

    Ties :func:`resolve_split_policy` to the adapter's shipped folds so a caller
    gets, in one call, the folds to use — or ``None`` when the run should compute
    its own splits:

    - resolved policy ``"compute"`` -> ``None`` (build splits the usual way);
    - resolved policy ``"provided"`` -> ``adapter.load_folds()`` as
      ``(case_id, fold)``.

    Args:
        config: A loaded run config.
        adapter: The dataset adapter.

    Returns:
        A ``(case_id, fold)`` table, or ``None`` to compute splits.

    Raises:
        ValueError: If ``"provided"`` is resolved but the dataset ships no folds.
    """
    if resolve_split_policy(config, adapter) == "compute":
        return None
    folds = adapter.load_folds()
    if folds is None:
        raise ValueError(
            f"split_policy resolved to 'provided' but {type(adapter).__name__} "
            "ships no folds (load_folds returned None). Use split_policy: compute."
        )
    return folds
