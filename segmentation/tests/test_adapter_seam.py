"""CPU test: the Step 4 dataset-adapter seam in the segmentation stage.

Seams added to ``batch_segmentation`` behind a ``None``-fallback (Step 4 of the
multi-dataset migration, see cohorts/README.md):

1. ``preprocess_image`` takes its geometry reorientation from
   ``adapter.preprocess`` instead of the hardcoded MAMA-MIA axis transform.
2. ``build_output_path`` takes the top-level ``source`` directory from
   ``adapter.case_dataset_name`` instead of the ``case_id.split("_")[0]`` prefix.
3. ``find_case_files`` routes case/file discovery through
   ``adapter.discover_cases()``/``load_timepoints()`` instead of assuming the
   MAMA-MIA ``<images_dir>/<case_id>/*.nii.gz`` layout.
4. ``_base_name_for`` prefixes the intermediate-file base name with ``case_id``
   when an adapter is given, so cases whose phase files share identical names
   across patients (UChicago's ``phase_0000.nii.gz`` for every exam) don't
   collide in the flat step1/step2/step3 intermediate directories.

This checks the invariants Step 4 must hold, on tiny synthetic data, CPU only
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
    _base_name_for,
    build_output_path,
    find_case_files,
    find_nii_files,
    preprocess_image,
)

from cohorts.base import DatasetAdapter  # noqa: E402
from cohorts.mamamia import MamaMiaDataset  # noqa: E402
from cohorts.uchicago import UChicagoDataset  # noqa: E402


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


def _write_uchicago_manifest(tmp: Path, exam_ids: list[str]) -> Path:
    """Write a manifest where every exam's phase file is named identically.

    Mirrors the real manifest's naming convention (``phase_0000.nii.gz`` for
    every exam) -- exactly the collision risk ``_base_name_for`` exists to
    prevent.
    """
    root = tmp / "uc"
    rows = ["exam_id,dataset,patient_key,fold,pcr,phase_files"]
    for i, exam_id in enumerate(exam_ids):
        phase_path = root / "images" / "simbiosys" / exam_id / "phase_0000.nii.gz"
        phase_path.parent.mkdir(parents=True, exist_ok=True)
        phase_path.touch()
        rows.append(f'{exam_id},simbiosys,p{i},0,1.0,"[""{phase_path.as_posix()}""]"')
    root.mkdir(parents=True, exist_ok=True)
    (root / "dce2d_internal_ultrafast_manifest.csv").write_text("\n".join(rows) + "\n")
    return root


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

        # 5. find_case_files: adapter OFF matches find_nii_files exactly.
        mama_images = tmp / "mama_images"
        (mama_images / "DUKE_001").mkdir(parents=True)
        (mama_images / "DUKE_001" / "DUKE_001_0000.nii.gz").touch()
        none_pairs = sorted((c, str(p)) for c, p in find_case_files(str(mama_images)))
        direct_pairs = sorted((c, str(p)) for c, p in find_nii_files(str(mama_images)))
        discovery_noop_identical = none_pairs == direct_pairs
        print(
            "[discovery] find_case_files(adapter=None) == find_nii_files: "
            f"{discovery_noop_identical}"
        )
        ok = ok and discovery_noop_identical

        # 6. find_case_files with an adapter discovers manifest-driven cases,
        #    not a directory glob (UChicago-shaped: two cases, no <case_id>/
        #    subdirectory of `images_dir` matching find_nii_files' assumption).
        #    NOTE: UChicagoDataset is used here only as a convenient
        #    manifest-shaped fixture for the *generic* discovery and
        #    name-collision logic, which is dataset-independent. UChicago itself
        #    can no longer be run through this stage -- the CLI rejects it (see
        #    cohorts.factory.IMAGING_ROUTE_SUPERSEDED); its imaging route is the
        #    paired raw-DICOM pipeline in preprocessing/. These functions are
        #    called directly, below the factory, so the guard doesn't apply.
        expected_uc_exam_ids = {"e1", "e2"}
        uc_root = _write_uchicago_manifest(tmp, sorted(expected_uc_exam_ids))
        uc_adapter = UChicagoDataset(root=uc_root)
        uc_pairs = find_case_files(str(uc_root), adapter=uc_adapter)
        uc_case_ids = {c for c, _ in uc_pairs}
        manifest_discovery_ok = uc_case_ids == expected_uc_exam_ids and len(
            uc_pairs
        ) == len(expected_uc_exam_ids)
        print(
            f"[discovery] adapter-driven discovery finds manifest cases: "
            f"{manifest_discovery_ok} (found {uc_case_ids})"
        )
        ok = ok and manifest_discovery_ok

        # 7. find_case_files honors case_limit at case granularity.
        uc_limited = find_case_files(str(uc_root), adapter=uc_adapter, case_limit=1)
        case_limit_ok = len({c for c, _ in uc_limited}) == 1
        print(f"[discovery] case_limit=1 yields exactly one case: {case_limit_ok}")
        ok = ok and case_limit_ok

        # 8. _base_name_for: no collision across cases with identically-named
        #    phase files (the real UChicago manifest names every exam's first
        #    phase "phase_0000.nii.gz" -- without the case_id prefix, e1 and
        #    e2 would silently overwrite each other's intermediate .npy file).
        uc_base_names = {
            _base_name_for(case_id, file_path, uc_adapter)
            for case_id, file_path in uc_pairs
        }
        no_collision = len(uc_base_names) == len(uc_pairs)
        print(
            f"[discovery] adapter-driven base names are collision-free: "
            f"{no_collision} ({sorted(uc_base_names)})"
        )
        ok = ok and no_collision

        # 9. _base_name_for is unchanged (bare stem, no prefix) when adapter=None.
        mama_base_name = _base_name_for(
            "DUKE_001", str(mama_images / "DUKE_001" / "DUKE_001_0000.nii.gz"), None
        )
        base_name_noop_identical = mama_base_name == "DUKE_001_0000"
        print(
            "[discovery] _base_name_for(adapter=None) unchanged: "
            f"{base_name_noop_identical} ({mama_base_name!r})"
        )
        ok = ok and base_name_noop_identical

    if ok:
        print("\nPASS: segmentation adapter seam is a no-op for MAMA-MIA and live.")
        return 0
    print("\nFAIL: adapter seam did not behave as expected.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
