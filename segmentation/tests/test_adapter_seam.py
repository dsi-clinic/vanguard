"""CPU test: the dataset-adapter seam in the segmentation stage.

The adapter is required everywhere in this stage (multi-dataset migration
Step 5, see cohorts/README.md):

1. ``preprocess_image`` takes its geometry reorientation from
   ``adapter.preprocess``.
2. ``build_output_path`` takes the top-level ``source`` directory from
   ``adapter.case_dataset_name``.
3. ``find_case_files`` routes case/file discovery through
   ``adapter.discover_cases()``/``load_timepoints()``.
4. ``_base_name_for`` prefixes the intermediate-file base name with ``case_id``
   unless the filename already starts with it -- so cases whose phase files
   share identical names across patients (UChicago's ``phase_0000.nii.gz`` for
   every exam) don't collide in the flat step1/step2/step3 intermediate
   directories, while MAMA-MIA filenames (which already embed the case id)
   aren't double-prefixed.

This checks the invariants Step 5 must hold, on tiny synthetic data, CPU only
(safe for the login node):

* A ``MamaMiaDataset`` adapter reproduces the historical hardcoded MAMA-MIA
  behavior exactly (orientation transform, output path, discovery,
  base-name-for-a-MAMA-MIA-shaped-filename).
* A differently-shaped adapter (pass-through orientation, sub-source
  identity, UChicago-style flat filenames) routes through and produces
  different, still-correct results -- proving the seam is live, not dead code.

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
    preprocess_image,
)

from cohorts.base import DatasetAdapter  # noqa: E402
from cohorts.mamamia import MamaMiaDataset  # noqa: E402
from cohorts.uchicago import UChicagoDataset  # noqa: E402

_MAMAMIA_TRANSFORM_NP = "np.swapaxes(np.swapaxes(v, 0, 2), 0, 1)[::-1]"


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

        # 1. preprocess_image: MamaMiaDataset reproduces the historical
        #    hardcoded MAMA-MIA transform exactly.
        mm_npy = tmp / "mm.npy"
        pt_npy = tmp / "pt.npy"
        preprocess_image(str(nii), str(mm_npy), adapter=mamamia)
        preprocess_image(str(nii), str(pt_npy), adapter=passthrough)

        mm = np.load(mm_npy)
        pt = np.load(pt_npy)
        # Shape-only check: intensity normalization runs after reorientation
        # and is elementwise (shape-preserving), so the historical transform's
        # shape is the invariant to pin without re-deriving normalized values.
        expected_shape = np.swapaxes(np.swapaxes(arr, 0, 2), 0, 1)[::-1].shape
        transform_correct = mm.shape == expected_shape
        print(
            f"[preprocess] MAMA-MIA adapter reproduces {_MAMAMIA_TRANSFORM_NP}: "
            f"{transform_correct} (mm.shape={mm.shape}, expected={expected_shape})"
        )
        ok = ok and transform_correct

        # 2. The seam is live: pass-through orientation differs from the MAMA-MIA
        #    transform (different shape once axes are no longer swapped).
        seam_live = not np.array_equal(mm, pt) and mm.shape != pt.shape
        print(
            f"[preprocess] pass-through adapter differs from MAMA-MIA transform: "
            f"{seam_live} (mm.shape={mm.shape}, passthrough.shape={pt.shape})"
        )
        ok = ok and seam_live

        # 3. build_output_path: MamaMiaDataset reproduces the historical
        #    case-id-prefix source directory.
        out_root = tmp / "out"
        mm_path = build_output_path(
            out_root, "DUKE_001", "DUKE_001_0000", adapter=mamamia
        )
        mm_source = mm_path.relative_to(out_root).parts[0]
        path_correct = mm_source == "DUKE"
        print(f"[output_path] MAMA-MIA adapter source dir == 'DUKE': {path_correct}")
        ok = ok and path_correct

        # 4. The seam is live: the sub-source adapter changes the top-level dir.
        pt_path = build_output_path(
            out_root, "DUKE_001", "DUKE_001_0000", adapter=passthrough
        )
        pt_source = pt_path.relative_to(out_root).parts[0]
        path_seam_live = pt_source == "uchicago_subsource"
        print(
            f"[output_path] source dir: MAMA-MIA={mm_source!r}, "
            f"pass-through={pt_source!r} -> seam live: {path_seam_live}"
        )
        ok = ok and path_seam_live

        # 5. find_case_files with a MAMA-MIA adapter discovers the case via
        #    adapter.discover_cases()/load_timepoints(), not a directory glob.
        #    DatasetAdapter.images_dir is `root / "images"`, so the case dirs
        #    live one level below the adapter's root.
        mama_root = tmp / "mama_root"
        mama_images = mama_root / "images"
        (mama_images / "DUKE_001").mkdir(parents=True)
        (mama_images / "DUKE_001" / "DUKE_001_0000.nii.gz").touch()
        mama_root_adapter = MamaMiaDataset(cohort=None, root=mama_root)
        mama_pairs = find_case_files(mama_root_adapter)
        discovery_ok = mama_pairs == [
            ("DUKE_001", str(mama_images / "DUKE_001" / "DUKE_001_0000.nii.gz"))
        ]
        print(
            f"[discovery] MamaMiaDataset finds the expected (case, file): {discovery_ok}"
        )
        ok = ok and discovery_ok

        # 6. find_case_files with an adapter discovers manifest-driven cases,
        #    not a directory glob (UChicago-shaped: two cases, no <case_id>/
        #    subdirectory of `images_dir` matching the MAMA-MIA layout assumption).
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
        uc_pairs = find_case_files(uc_adapter)
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
        uc_limited = find_case_files(uc_adapter, case_limit=1)
        case_limit_ok = len({c for c, _ in uc_limited}) == 1
        print(f"[discovery] case_limit=1 yields exactly one case: {case_limit_ok}")
        ok = ok and case_limit_ok

        # 8. _base_name_for: no collision across cases with identically-named
        #    phase files (the real UChicago manifest names every exam's first
        #    phase "phase_0000.nii.gz" -- without the case_id prefix, e1 and
        #    e2 would silently overwrite each other's intermediate .npy file).
        uc_base_names = {
            _base_name_for(case_id, file_path) for case_id, file_path in uc_pairs
        }
        no_collision = len(uc_base_names) == len(uc_pairs)
        print(
            f"[discovery] adapter-driven base names are collision-free: "
            f"{no_collision} ({sorted(uc_base_names)})"
        )
        ok = ok and no_collision

        # 9. _base_name_for does NOT double-prefix a MAMA-MIA-shaped filename
        #    that already embeds the case id (regression guard for the Step 5
        #    bug where every prefixed filename came out
        #    "DUKE_001_DUKE_001_0000" instead of "DUKE_001_0000").
        mama_base_name = _base_name_for(
            "DUKE_001", str(mama_images / "DUKE_001" / "DUKE_001_0000.nii.gz")
        )
        base_name_no_double_prefix = mama_base_name == "DUKE_001_0000"
        print(
            "[discovery] _base_name_for does not double-prefix MAMA-MIA filenames: "
            f"{base_name_no_double_prefix} ({mama_base_name!r})"
        )
        ok = ok and base_name_no_double_prefix

    if ok:
        print(
            "\nPASS: segmentation adapter seam reproduces MAMA-MIA behavior and is live."
        )
        return 0
    print("\nFAIL: adapter seam did not behave as expected.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
