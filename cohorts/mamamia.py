"""MAMA-MIA dataset adapter, parameterized by cohort.

One class covers all four MAMA-MIA cohorts (duke/ispy1/ispy2/nact) because they
are ~95% identical (see cohorts/README.md). Per the design, this requires *no*
method overrides: the base class already encodes MAMA-MIA behavior, and the
``cohort`` argument only sets a discovery filter.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from cohorts.base import DatasetAdapter

MAMAMIA_COHORTS: frozenset[str] = frozenset({"duke", "ispy1", "ispy2", "nact"})


class MamaMiaDataset(DatasetAdapter):
    """One of the four MAMA-MIA cohorts, or all of them combined.

    Cohorts are ``duke``/``ispy1``/``ispy2``/``nact``, or all four combined
    when ``cohort`` is ``None``. The ``cohort`` argument identifies the cohort
    and filters case discovery to
    it (case-id prefixes are upper-case, e.g. ``ISPY2_045``). No pipeline method
    is overridden.
    """

    #: Accepted cohort keys.
    cohorts: ClassVar[frozenset[str]] = MAMAMIA_COHORTS

    def __init__(self, cohort: str | None, root: Path) -> None:
        """Build the adapter for one MAMA-MIA cohort, or all four.

        Args:
            cohort: One of ``duke``, ``ispy1``, ``ispy2``, ``nact``
                (case-insensitive), or ``None`` for no discovery filter (all
                four cohorts combined -- matches ``find_nii_files`` walking the
                whole images directory unfiltered).
            root: MAMA-MIA data root (e.g. ``.../MAMA-MIA-syn60868042/``).

        Raises:
            ValueError: If ``cohort`` is set but not a known MAMA-MIA cohort.
        """
        if cohort is None:
            self.cohort = None
            super().__init__(root, dataset_filter=None)
            return
        key = str(cohort).strip().lower()
        if key not in MAMAMIA_COHORTS:
            raise ValueError(
                f"Unknown MAMA-MIA cohort {cohort!r}; expected one of "
                f"{sorted(MAMAMIA_COHORTS)} or None (all cohorts)."
            )
        self.cohort = key
        # Filter discovery to this cohort by its upper-case case-id prefix,
        # without overriding any method.
        super().__init__(root, dataset_filter=key.upper())
