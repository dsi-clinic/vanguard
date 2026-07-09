"""UChicago ultrafast DCE dataset adapter (scaffold).

Overrides the base MAMA-MIA behavior for the differences identified in
``docs/modularization-design.md`` §6.5. The real ultrafast preprocessing is a
frozen-copy port (§10 decision 4) that is **not implemented yet**; ``preprocess``
raises ``NotImplementedError`` so nothing silently applies the wrong transform.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from cohorts.base import DatasetAdapter

if TYPE_CHECKING:
    import pandas as pd

DEFAULT_MANIFEST_NAME = "dce2d_internal_ultrafast_manifest.csv"


class UChicagoDataset(DatasetAdapter):
    """UChicago internal ultrafast DCE cohort (manifest-driven).

    Cases, timepoints, identity, and labels all come from a manifest CSV rather
    than a directory layout, and the dataset ships its own cross-validation
    folds.
    """

    #: UChicago ships patient-grouped folds, so default to using them.
    default_split_policy: ClassVar[str] = "provided"

    #: Report results by manifest ``dataset`` sub-source
    #: (``simbiosys``/``uch_nac``/``her2_naclike``).
    report_by: ClassVar[str | None] = "dataset"

    def __init__(self, root: Path, manifest_csv: Path | None = None) -> None:
        """Build the adapter.

        Args:
            root: UChicago manifest directory root.
            manifest_csv: Path to the manifest CSV; defaults to
                ``root/dce2d_internal_ultrafast_manifest.csv``.
        """
        super().__init__(root)
        self.manifest_csv = (
            Path(manifest_csv)
            if manifest_csv is not None
            else self.root / DEFAULT_MANIFEST_NAME
        )
        self._manifest_cache: pd.DataFrame | None = None

    def discover_cases(self) -> list[str]:
        """Enumerate exam ids from the manifest (not a directory glob)."""
        return [str(exam_id) for exam_id in self._manifest()["exam_id"].tolist()]

    def case_dataset_name(self, case_id: str) -> str:
        """Return a case's sub-source (manifest ``dataset`` column)."""
        return str(self._manifest_row(case_id)["dataset"])

    def group_key(self, case_id: str) -> str:
        """Return a case's patient grouping key (manifest ``patient_key``)."""
        return str(self._manifest_row(case_id)["patient_key"])

    def load_timepoints(self, case_id: str) -> list[Path]:
        """Return ordered DCE phase files from the manifest ``phase_files`` list."""
        phase_files = json.loads(self._manifest_row(case_id)["phase_files"])
        return [Path(p) for p in phase_files]

    def load_labels(self) -> pd.DataFrame:
        """Return pCR labels straight from the manifest (``exam_id`` -> ``pcr``)."""
        labels = self._manifest()[["exam_id", "pcr"]].copy()
        labels = labels.rename(columns={"exam_id": "case_id"})
        labels = labels.dropna(subset=["pcr"])
        labels["pcr"] = labels["pcr"].astype(float).astype(int)
        return labels

    def preprocess(self, volume: np.ndarray) -> np.ndarray:
        """Ultrafast-specific preprocessing — frozen-copy port, not done yet.

        See ``docs/modularization-design.md`` §6 and §10 decision 4. Until the
        frozen copy is brought into the repo this raises, rather than falling
        back to the MAMA-MIA transform, which would be wrong for ultrafast data.
        """
        raise NotImplementedError(
            "UChicago preprocessing is not ported yet (frozen-copy port pending; "
            "see docs/modularization-design.md §6 and §10 decision 4)."
        )

    # -- internal helpers --

    def _manifest(self) -> pd.DataFrame:
        """Load and cache the manifest CSV."""
        if self._manifest_cache is None:
            import pandas as pd

            self._manifest_cache = pd.read_csv(self.manifest_csv)
        return self._manifest_cache

    def _manifest_row(self, case_id: str) -> pd.Series:
        """Return the single manifest row for an exam id."""
        manifest = self._manifest()
        rows = manifest[manifest["exam_id"].astype(str) == str(case_id)]
        if rows.empty:
            raise KeyError(f"exam_id not in manifest: {case_id}")
        return rows.iloc[0]
