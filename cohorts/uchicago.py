"""UChicago ultrafast DCE dataset adapter.

Overrides the base MAMA-MIA behavior for the differences identified in
``cohorts/README.md``: manifest-driven discovery/identity/timepoints/labels,
provided CV folds, and sub-source reporting. ``preprocess`` flips x/z (UChicago
is stored ``RAS`` where MAMA-MIA is ``LAI``) and then applies the base
reorientation; see the method docstring for the evidence and the note on design
decision 4.
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
        self._manifest_indexed_cache: pd.DataFrame | None = None

    def discover_cases(self) -> list[str]:
        """Enumerate exam ids from the manifest (not a directory glob)."""
        return [str(exam_id) for exam_id in self._manifest()["exam_id"].tolist()]

    def case_dataset_name(self, case_id: str) -> str:
        """Return a case's sub-source (manifest ``dataset`` column).

        Note: this is a finer granularity than ``MamaMiaDataset.case_dataset_name``
        (which returns a cohort like ``"ISPY2"``) — here it's a manifest
        sub-source (``simbiosys``/``uch_nac``/``her2_naclike``). The ``dataset``
        column therefore means different things at different granularities across
        adapters. That's fine while UChicago and MAMA-MIA runs stay separate;
        flag it before anyone builds a combined table that groups by ``dataset``
        across both.
        """
        return str(self._manifest_row(case_id)["dataset"])

    def group_key(self, case_id: str) -> str:
        """Return a case's patient grouping key (manifest ``patient_key``)."""
        return str(self._manifest_row(case_id)["patient_key"])

    def load_timepoints(self, case_id: str) -> list[Path]:
        """Return ordered DCE phase files from the manifest ``phase_files`` list.

        The manifest's ``phase_files`` paths are absolute and anchored under its
        own ``preproc_root`` column (today, ``<manifest_root>/images``). To honor
        the injected :attr:`root` (design decision 5 — the pipeline must survive
        the data moving on disk), each path is rebased from the manifest's
        recorded ``preproc_root`` onto ``self.root / "images"``. When a row lacks
        ``preproc_root`` (e.g. an older/synthetic manifest), paths are returned
        as-is rather than guessing.
        """
        import pandas as pd

        row = self._manifest_row(case_id)
        phase_files = [Path(p) for p in json.loads(row["phase_files"])]
        manifest = self._manifest()
        if "preproc_root" not in manifest.columns or pd.isna(row["preproc_root"]):
            return phase_files
        preproc_root = Path(row["preproc_root"])
        images_root = self.root / "images"
        rebased = []
        for phase_file in phase_files:
            try:
                rel = phase_file.relative_to(preproc_root)
            except ValueError:
                rebased.append(phase_file)
            else:
                rebased.append(images_root / rel)
        return rebased

    def load_labels(self) -> pd.DataFrame:
        """Return pCR labels straight from the manifest (``exam_id`` -> ``pcr``)."""
        labels = self._manifest()[["exam_id", "pcr"]].copy()
        labels = labels.rename(columns={"exam_id": "case_id"})
        labels = labels.dropna(subset=["pcr"])
        labels["pcr"] = labels["pcr"].astype(float).astype(int)
        return labels

    def preprocess(self, volume: np.ndarray) -> np.ndarray:
        """Reorient a raw UChicago volume into the pipeline's processing layout.

        UChicago's NIfTI headers are consistently ``RAS`` (direction diag
        ``(-1, -1, 1)``) across all three sub-sources, verified against a
        sample from each. MAMA-MIA's ISPY2/DUKE cases are ``LAI`` (diag
        ``(1, -1, -1)``) -- flipped in x and z relative to UChicago. Since
        ``sitk.GetArrayFromImage`` returns arrays in (z, y, x) index order,
        undoing that x/z sign difference before applying the base class's
        MAMA-MIA swap+flip means array axes 0 (z) and 2 (x) get flipped here
        first.

        This was previously an identity pass-through, on the unverified
        assumption that Anna's HFDP pipeline (``policy_name = hfdp_t1_v1``)
        already wrote UChicago volumes in the model's target layout. It
        didn't: feeding the raw array straight through put the thin slice
        axis where the model expects a large in-plane axis (and vice versa),
        which is why ``segmentation/predict_fast.py``'s tiling coverage
        assertion started failing on wide-matrix UChicago phases.

        Note for Anna: NIfTI header orientation isn't reliable across all of
        MAMA-MIA either -- ISPY1 reports ``PSL`` (a true axis permutation,
        not just sign flips) yet gets the same fixed transform as
        ISPY2/DUKE. This flip was chosen from the header comparison above but
        confirmed by rendering corrected UChicago MIPs and a z-depth
        progression against a MAMA-MIA reference and matching the anatomical
        landmarks -- not trusted from headers alone. Still unverified: a pure
        left-right mirror, which near-symmetric breast anatomy cannot rule out
        visually. See ``cohorts/README.md`` for the full history.
        """
        volume = volume[::-1, :, ::-1]
        return super().preprocess(volume)

    def load_folds(self) -> pd.DataFrame:
        """Return the manifest's patient-grouped CV folds as ``(case_id, fold)``.

        UChicago ships its own folds (``default_split_policy = "provided"``),
        patient-grouped by ``patient_key`` and stratified by ``pcr``. Keyed by
        ``exam_id`` (renamed to ``case_id`` for consistency with the rest of the
        pipeline).
        """
        folds = self._manifest()[["exam_id", "fold"]].copy()
        folds = folds.rename(columns={"exam_id": "case_id"})
        folds = folds.dropna(subset=["fold"])
        folds["fold"] = folds["fold"].astype(int)
        return folds

    # -- internal helpers --

    def _manifest(self) -> pd.DataFrame:
        """Load and cache the manifest CSV."""
        if self._manifest_cache is None:
            import pandas as pd

            self._manifest_cache = pd.read_csv(self.manifest_csv)
        return self._manifest_cache

    def _manifest_row(self, case_id: str) -> pd.Series:
        """Return the single manifest row for an exam id (O(1), indexed once)."""
        indexed = self._manifest_indexed()
        key = str(case_id)
        if key not in indexed.index:
            raise KeyError(f"exam_id not in manifest: {case_id}")
        return indexed.loc[key]

    def _manifest_indexed(self) -> pd.DataFrame:
        """Manifest indexed by (string) ``exam_id``, cached for repeated row lookups.

        Fails loudly on a duplicate ``exam_id`` rather than letting ``.loc``
        silently hand back multiple rows: a duplicate would otherwise make every
        caller of ``_manifest_row`` (which assumes one row per id) either crash
        deep inside pandas or misbehave on a Series-of-rows.
        """
        if self._manifest_indexed_cache is None:
            manifest = self._manifest()
            indexed = manifest.copy()
            indexed.index = manifest["exam_id"].astype(str)
            duplicated = indexed.index[indexed.index.duplicated()].unique()
            if len(duplicated) > 0:
                raise ValueError(
                    f"manifest has duplicate exam_id values: {sorted(duplicated)}"
                )
            self._manifest_indexed_cache = indexed
        return self._manifest_indexed_cache
