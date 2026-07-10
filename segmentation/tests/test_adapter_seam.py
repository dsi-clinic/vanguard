"""CPU test: the Step 4 dataset-adapter seam in the segmentation stage.

Two seams were added to ``batch_segmentation`` behind a ``None``-fallback
(Step 4 of the multi-dataset migration, see cohorts/README.md):

1. ``preprocess_image`` takes its geometry reorientation from
   ``adapter.preprocess`` instead of the hardcoded MAMA-MIA axis transform.
2. ``build_output_path`` takes the top-level ``source`` directory from
   ``adapter.case_dataset_name`` instead of the ``case_id.split("_")[0]`` prefix.

This checks both invariants Step 4 must hold, on tiny synthetic data, CPU only
(safe for the login node):

* Adapter OFF vs. a MAMA-MIA adapter is byte-identical (the adapter encodes
  exactly today's behavior, so it must change nothing).
* The seam is actually live: an adapter with a pass-through ``preprocess`` and a
  different ``case_dataset_name`` (UChicago-shaped) routes through, producing a
  different preprocessed array and a different output directory.

This lives in ``segmentation/tests/`` (not the CI-collected top-level ``tests/``)
because importing ``batch_segmentation`` pulls in torch and the vessel-seg
submodule, which the CI image does not carry. The adapter methods themselves are
covered CI-safely in ``tests/test_cohort_adapters.py``.

Run:  python segmentation/tests/test_adapter_seam.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk

HERE = Path(__file__).resolve().parent
_SEG_DIR = HERE.parent
_PROJECT_ROOT = _SEG_DIR.parent
for _p in (str(_SEG_DIR), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from batch_segmentation import (  # noqa: E402
    build_output_path,
    preprocess_image,
)

from cohorts.base import DatasetAdapter  # noqa: E402
from cohorts.mamamia import MamaMiaDataset  # noqa: E402


class _PassThroughAdapter(DatasetAdapter):
    """A UChicago-shaped adapter: no geometry transform, sub-source identity.

    Mirrors the two ways UChicago differs from MAMA-MIA at this stage without
    needing a real manifest on disk: ``preprocess`` is a pass-through (data is
    already oriented) and ``case_dataset_name`` returns a fixed sub-source rather
    than a case-id prefix.
    """

    def preprocess(self, volume: np.ndarray) -> np.ndarray:
        return volume

    def case_dataset_name(self, case_id: str) -> str:
        return "uchicago_subsource"


def _write_nii(path: Path, arr: np.ndarray) -> None:
    """Write ``arr`` (z, y, x order) as a .nii.gz at ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(sitk.GetImageFromArray(arr), str(path))


def main() -> int:  # noqa: C901
    """Run the segmentation adapter-seam checks."""
    rng = np.random.default_rng(0)
    ok = True

    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        # Asymmetric shape so the axis transform is observable (a symmetric
        # cube could hide a wrong/absent transform).
        arr = rng.random((6, 8, 10)).astype(np.float32)
        nii = tmp / "DUKE_001" / "DUKE_001_0000.nii.gz"
        _write_nii(nii, arr)

        mamamia = MamaMiaDataset(cohort="duke", root=tmp)
        passthrough = _PassThroughAdapter(root=tmp)

        # 1. preprocess_image: adapter OFF vs. MAMA-MIA adapter is byte-identical.
        off_npy = tmp / "off.npy"
        mm_npy = tmp / "mm.npy"
        pt_npy = tmp / "pt.npy"
        preprocess_image(str(nii), str(off_npy))
        preprocess_image(str(nii), str(mm_npy), adapter=mamamia)
        preprocess_image(str(nii), str(pt_npy), adapter=passthrough)

        off = np.load(off_npy)
        mm = np.load(mm_npy)
        pt = np.load(pt_npy)

        # equal_nan: normalize_image can emit NaN on degenerate tiny inputs; both
        # paths run the identical transform + normalize, so NaNs land in identical
        # positions. We assert element-wise equality treating those as equal.
        noop_identical = np.array_equal(off, mm, equal_nan=True)
        print(f"[preprocess] MAMA-MIA adapter == adapter-off: {noop_identical}")
        ok = ok and noop_identical

        # 2. The seam is live: pass-through orientation differs from the MAMA-MIA
        #    transform (different shape once axes are no longer swapped).
        seam_live = not np.array_equal(off, pt) and off.shape != pt.shape
        print(
            f"[preprocess] pass-through adapter differs from MAMA-MIA transform: "
            f"{seam_live} (off.shape={off.shape}, passthrough.shape={pt.shape})"
        )
        ok = ok and seam_live

        # 3. build_output_path: adapter OFF vs. MAMA-MIA adapter is byte-identical.
        out_root = tmp / "out"
        off_path = build_output_path(out_root, "DUKE_001", "DUKE_001_0000")
        mm_path = build_output_path(
            out_root, "DUKE_001", "DUKE_001_0000", adapter=mamamia
        )
        path_noop_identical = off_path == mm_path
        print(f"[output_path] MAMA-MIA adapter == adapter-off: {path_noop_identical}")
        ok = ok and path_noop_identical

        # 4. The seam is live: the sub-source adapter changes the top-level dir.
        pt_path = build_output_path(
            out_root, "DUKE_001", "DUKE_001_0000", adapter=passthrough
        )
        off_source = off_path.relative_to(out_root).parts[0]
        pt_source = pt_path.relative_to(out_root).parts[0]
        path_seam_live = off_source == "DUKE" and pt_source == "uchicago_subsource"
        print(
            f"[output_path] source dir: adapter-off={off_source!r}, "
            f"pass-through={pt_source!r} -> seam live: {path_seam_live}"
        )
        ok = ok and path_seam_live

    if ok:
        print("\nPASS: segmentation adapter seam is a no-op for MAMA-MIA and live.")
        return 0
    print("\nFAIL: adapter seam did not behave as expected.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
