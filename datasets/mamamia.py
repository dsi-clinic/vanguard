"""MAMA-MIA dataset adapter, parameterized by cohort.

One class covers all four MAMA-MIA cohorts (duke/ispy1/ispy2/nact) because they
are ~95% identical (design doc §4/§6.5). Per the design, this requires *no*
method overrides: the base class already encodes MAMA-MIA behavior, and the
``cohort`` argument only sets a discovery filter.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from datasets.base import DatasetAdapter

MAMAMIA_COHORTS: frozenset[str] = frozenset({"duke", "ispy1", "ispy2", "nact"})


class MamaMiaDataset(DatasetAdapter):
    """One of the four MAMA-MIA cohorts (``duke``/``ispy1``/``ispy2``/``nact``).

    The ``cohort`` argument identifies the cohort and filters case discovery to
    it (case-id prefixes are upper-case, e.g. ``ISPY2_045``). No pipeline method
    is overridden.
    """

    #: Accepted cohort keys.
    cohorts: ClassVar[frozenset[str]] = MAMAMIA_COHORTS

    def __init__(self, cohort: str, root: Path) -> None:
        """Build the adapter for one MAMA-MIA cohort.

        Args:
            cohort: One of ``duke``, ``ispy1``, ``ispy2``, ``nact``
                (case-insensitive).
            root: MAMA-MIA data root (e.g. ``.../MAMA-MIA-syn60868042/``).

        Raises:
            ValueError: If ``cohort`` is not a known MAMA-MIA cohort.
        """
        key = str(cohort).strip().lower()
        if key not in MAMAMIA_COHORTS:
            raise ValueError(
                f"Unknown MAMA-MIA cohort {cohort!r}; expected one of "
                f"{sorted(MAMAMIA_COHORTS)}."
            )
        self.cohort = key
        # Filter discovery to this cohort by its upper-case case-id prefix,
        # without overriding any method.
        super().__init__(root, dataset_filter=key.upper())
