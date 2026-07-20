"""Base dataset adapter: the contract every dataset gives the pipeline.

Step 1 scaffolding for the multi-dataset redesign (see
``cohorts/README.md``). Nothing in the pipeline stages calls this
module yet — it is purely additive.

The base class encodes the *current* MAMA-MIA behavior (case-id prefix parsing,
the existing orientation transform, the existing clinical/label loaders, the
current file-naming patterns). That is deliberate: it lets ``MamaMiaDataset``
work with no method overrides (see cohorts/README.md), while other datasets subclass
and override only the handful of methods that genuinely differ.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


class DatasetAdapter:
    """One dataset's answers to "how do I find, orient, and label its cases".

    Construct with the dataset's on-disk ``root``, which is injected from run
    config (see cohorts/README.md — Design decisions) rather than hardcoded in the class, so the
    same class keeps working when the data moves on disk.
    """

    #: Split policy this dataset uses unless run config overrides it. ``"compute"``
    #: means the pipeline builds its own CV folds (MAMA-MIA behavior); ``"provided"``
    #: means use folds shipped with the dataset.
    default_split_policy: ClassVar[str] = "compute"

    #: Optional column to break QC/eval results down by (e.g. a sub-source).
    #: ``None`` means no breakdown (MAMA-MIA behavior).
    report_by: ClassVar[str | None] = None

    #: Target voxel spacing ``(z, y, x)`` in mm to resample to before the shared
    #: pipeline, or ``None`` to keep native spacing. MAMA-MIA currently uses native
    #: spacing (see cohorts/README.md), so the default is ``None``.
    target_spacing_mm: ClassVar[tuple[float, float, float] | None] = None

    def __init__(self, root: Path, *, dataset_filter: str | None = None) -> None:
        """Store the injected dataset root.

        Args:
            root: On-disk base directory for this dataset's data.
            dataset_filter: If set, ``discover_cases`` keeps only cases whose
                :meth:`case_dataset_name` equals this value. ``MamaMiaDataset``
                sets it to the cohort so one class can represent one cohort
                without overriding any method.
        """
        self.root = Path(root)
        self._dataset_filter = dataset_filter
        self._clinical_index: dict[str, pd.Series] | None = None

    # -- standard MAMA-MIA sub-paths (derived from root; subclasses may override) --

    @property
    def images_dir(self) -> Path:
        """Directory of raw per-case DCE images (``root/images/<case_id>/``)."""
        return self.root / "images"

    @property
    def patient_info_dir(self) -> Path:
        """Directory of per-case clinical/imaging JSON files."""
        return self.root / "patient_info_files"

    @property
    def tumor_mask_dir(self) -> Path:
        """Directory of expert tumor-mask segmentations."""
        return self.root / "segmentations" / "expert"

    @property
    def labels_csv(self) -> Path:
        """CSV of pCR labels (``case_id, pcr``)."""
        return self.root / "pcr_labels.csv"

    # -- case discovery + identity --

    def discover_cases(self) -> list[str]:
        """Enumerate case ids from the raw-image directory layout.

        Mirrors ``segmentation/batch_segmentation.py`` ``find_nii_files``: each
        case is a subdirectory of :attr:`images_dir`. When ``dataset_filter`` is
        set, only cases belonging to that dataset are returned.

        Returns:
            Sorted case ids.
        """
        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Images dir not found: {self.images_dir}")
        cases = sorted(d.name for d in self.images_dir.iterdir() if d.is_dir())
        if self._dataset_filter is not None:
            cases = [
                c for c in cases if self.case_dataset_name(c) == self._dataset_filter
            ]
        return cases

    def case_dataset_name(self, case_id: str) -> str:
        """Return the dataset/cohort a case belongs to.

        MAMA-MIA encodes this as the case-id prefix, e.g. ``"ISPY2_045" ->
        "ISPY2"`` (``segmentation/batch_segmentation.py:243``,
        ``case_id.split("_")[0]``). Note this is cohort granularity; some
        subclasses (e.g. ``UChicagoDataset``) return a finer sub-source instead
        — see that override's docstring before combining tables across adapters.
        """
        return case_id.split("_")[0]

    def group_key(self, case_id: str) -> str:
        """Return the grouping key that keeps a case's group together in splits.

        MAMA-MIA groups by the clinical ``site`` column (run-config
        ``group_col: site``). Falls back to the case id when site is unavailable.
        """
        site = self._case_clinical_value(case_id, "site")
        return site if site else case_id

    # -- inputs + geometry --

    def load_timepoints(self, case_id: str) -> list[Path]:
        """Return a case's raw DCE phase files in temporal order.

        MAMA-MIA stores raw phases as ``images_dir/<case_id>/*.nii.gz``; the files
        sort into acquisition order by name.
        """
        case_dir = self.images_dir / case_id
        if not case_dir.is_dir():
            raise FileNotFoundError(f"No image dir for case {case_id}: {case_dir}")
        return sorted(case_dir.glob("*.nii.gz"))

    def load_segmented_timepoints(
        self, input_dir: Path, case_id: str
    ) -> tuple[list[Path], list[int]]:
        """Return a case's per-timepoint vessel-segmentation ``.npz`` files, ordered.

        This is a *different* artifact from :meth:`load_timepoints`: it discovers
        the graph-extraction stage's input (segmented vessel-probability volumes
        produced by the segmentation stage), not raw DCE phases. Delegates to
        ``graph_extraction.core4d.discover_study_timepoints`` for the base/MAMA-MIA
        behavior (``<input_dir>/<case_id>/images/<case_id>_<4-digit>_vessel_segmentation.npz``),
        so this is a no-op wrapper for MAMA-MIA and a documented seam for a future
        dataset whose segmentation output is laid out differently.

        Args:
            input_dir: Root directory of vessel-segmentation output (a pipeline
                artifact location, not this adapter's own ``root`` -- segmentation
                output lives in its own directory, separate from raw data).
            case_id: The case to discover timepoints for.

        Returns:
            ``(ordered file paths, ordered timepoint indices)``.
        """
        from graph_extraction.core4d import discover_study_timepoints

        return discover_study_timepoints(input_dir=input_dir, case_id=case_id)

    #: Axis to flip vessel/tumor/breast masks along for MIP visualization only
    #: (does not affect computed centerline/graph features). MAMA-MIA default.
    viz_flip_spec: ClassVar[str] = "z"

    def preprocess(self, volume: np.ndarray) -> np.ndarray:
        """Reorient a raw volume into the pipeline's processing layout.

        Encodes the MAMA-MIA axis transform from
        ``segmentation/batch_segmentation.py:85`` (swap axes, then flip z).
        Intensity normalization (``normalize_image``/``zscore_image``) is applied
        by the segmentation stage via the vessel-seg submodule and is
        intentionally not duplicated here.
        """
        return np.swapaxes(np.swapaxes(volume, 0, 2), 0, 1)[::-1]

    def resample(
        self, volume: np.ndarray, native_spacing_mm: tuple[float, float, float]
    ) -> np.ndarray:
        """Resample a volume to :attr:`target_spacing_mm`, or return it unchanged.

        MAMA-MIA keeps native spacing (``target_spacing_mm is None``), so this is
        a no-op for it. A dataset that sets a target must supply real resampling;
        it is deliberately not built yet (no dataset needs it in Step 1).

        Args:
            volume: The volume array to resample.
            native_spacing_mm: The volume's current ``(z, y, x)`` spacing in mm.
        """
        if self.target_spacing_mm is None:
            return volume
        raise NotImplementedError(
            "resample-to-target is not implemented yet; add it when a dataset "
            "sets target_spacing_mm (see cohorts/README.md). "
            f"(requested target={self.target_spacing_mm}, native={native_spacing_mm})"
        )

    # -- metadata + labels --

    def load_clinical(self) -> pd.DataFrame:
        """Load per-case clinical/imaging metadata.

        Reuses the existing loader dispatch in
        ``clinical_features.get_clinical_features`` (patient_info JSON directory
        first, then Excel) against this dataset's paths.
        """
        from clinical_features import get_clinical_features
        from config import ConfigNode

        shim = ConfigNode._wrap(
            {
                "data_paths": {
                    "patient_info_dir": str(self.patient_info_dir),
                    "clinical_excel": "",
                }
            }
        )
        return get_clinical_features(shim)

    def load_labels(self) -> pd.DataFrame:
        """Load pCR labels as a ``(case_id, pcr)`` table.

        Reuses ``tabular.cohort.load_labels`` against :attr:`labels_csv` with
        MAMA-MIA's id/label columns.
        """
        from tabular.cohort import load_labels as _load_labels

        return _load_labels(self.labels_csv, id_col="case_id", label_col="pcr")

    def load_folds(self) -> pd.DataFrame | None:
        """Return provided cross-validation folds as ``(case_id, fold)``, or ``None``.

        Base datasets compute their own splits (``default_split_policy ==
        "compute"``), so they ship no folds and this returns ``None``. A dataset
        that ships its own folds (``default_split_policy == "provided"``)
        overrides this to surface them.
        """
        return None

    # -- file-naming rules (current config patterns) --

    def tumor_mask_filename(self, case_id: str) -> str:
        """Filename of a case's tumor mask (config ``tumor_mask_file_pattern``)."""
        return f"{case_id}.nii.gz"

    def centerline_filename(self, case_id: str) -> str:
        """Filename of a case's 4D exam centerline mask."""
        return f"{case_id}_skeleton_4d_exam_mask.npy"

    def morphometry_filename(self, case_id: str) -> str:
        """Filename of a case's morphometry JSON."""
        return f"{case_id}_morphometry.json"

    # -- internal helpers --

    def _case_clinical_value(self, case_id: str, column: str) -> str | None:
        """Look up one clinical column for a case (clinical table cached once)."""
        import pandas as pd

        if self._clinical_index is None:
            clinical_df = self.load_clinical()
            self._clinical_index = {
                str(rec["case_id"]): rec for _, rec in clinical_df.iterrows()
            }
        record = self._clinical_index.get(str(case_id))
        if record is None or column not in record or pd.isna(record[column]):
            return None
        return str(record[column])
