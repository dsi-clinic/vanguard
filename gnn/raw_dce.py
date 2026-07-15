"""Raw DCE-MRI NIfTI time-series loading for GNN kinetic node features.

Vessel-segmentation NPZ timepoints (``*_vessel_segmentation.npz``, loaded via
``graph_extraction.core4d.load_time_series_from_files``) are the segmentation
network's *output*, not measured signal -- sampling kinetic features from them
describes when the segmentation model's response peaks, not when DCE contrast
enhancement peaks. This module loads the underlying raw DCE-MRI series instead
(``<case_id>_<0000..>.nii.gz`` under the MAMA-MIA ``images`` tree), so
``gnn.data_loader`` can compute real enhancement-curve features.

Graph/skeleton coordinates are saved in the orientation of the segmentation
pipeline's outputs, not necessarily the raw SimpleITK array orientation, so
raw volumes are aligned to the skeleton/support mask shape using the same
shape-matching axis-permutation convention as the rest of the repo
(``deepsets.volume_align``) rather than the NIfTI affine/direction cosines.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from deepsets.volume_align import align_zyx_4d_to_shape

_DCE_PHASE_PATTERN = "{case_id}_{timepoint:04d}.nii.gz"
_VANGUARD_TIMES_NAME = "ufast_times_seconds.npy"


def discover_raw_dce_paths(
    dce_root: Path, case_id: str, timepoints: list[int]
) -> list[Path]:
    """Resolve one raw DCE NIfTI path per timepoint index, in ``timepoints`` order.

    Raises ``FileNotFoundError`` immediately if any expected phase file is
    missing -- a partial raw series would silently misalign timepoint indices
    with the vessel-derived ``study_timepoints`` used elsewhere for this case.
    """
    case_dir = Path(dce_root) / case_id
    paths: list[Path] = []
    for t in timepoints:
        path = case_dir / _DCE_PHASE_PATTERN.format(case_id=case_id, timepoint=int(t))
        if not path.exists():
            raise FileNotFoundError(f"case={case_id}: missing raw DCE phase {path}")
        paths.append(path)
    return paths


def load_raw_dce_times(
    dce_root: Path,
    case_id: str,
    timepoints: list[int],
) -> np.ndarray | None:
    """Load Vanguard physical acquisition seconds when the sidecar is present.

    Legacy cohorts do not provide the sidecar and return ``None`` so their
    established index-time behavior remains explicit at the caller.
    """
    path = Path(dce_root) / case_id / _VANGUARD_TIMES_NAME
    if not path.is_file():
        return None
    all_times = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if all_times.ndim != 1:
        raise ValueError(f"case={case_id}: physical time sidecar is not 1D: {path}")
    indices = np.asarray(timepoints, dtype=int)
    if np.any(indices < 0) or np.any(indices >= all_times.size):
        raise IndexError(
            f"case={case_id}: timepoint indices exceed physical time sidecar {path}"
        )
    selected = all_times[indices]
    if not np.all(np.isfinite(selected)) or np.any(np.diff(selected) <= 0.0):
        raise ValueError(
            f"case={case_id}: physical acquisition seconds are not finite and increasing"
        )
    return selected


def load_raw_dce_series(
    paths: list[Path], expected_shape_zyx: tuple[int, int, int]
) -> np.ndarray:
    """Load and stack raw DCE NIfTI phases into ``(t, z, y, x)``, aligned to ``expected_shape_zyx``."""
    volumes = [sitk.GetArrayFromImage(sitk.ReadImage(str(p))) for p in paths]
    stacked = np.stack(volumes, axis=0).astype(np.float32, copy=False)
    return align_zyx_4d_to_shape(stacked, expected_shape_zyx)
