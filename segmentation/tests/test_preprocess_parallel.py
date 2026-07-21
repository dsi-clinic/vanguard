"""CPU test: parallel STEP-1 preprocessing == serial preprocessing.

Creates tiny synthetic .nii.gz inputs, runs `batch_segmentation.preprocess_parallel`
across a thread pool, and checks every output .npy is bit-identical to a serial
`preprocess_image` run. Verifies the ThreadPoolExecutor wiring is correct and
order-independent. Tiny data, CPU only — safe for the login node.

Run:  python segmentation/tests/test_preprocess_parallel.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from batch_segmentation import (  # noqa: E402
    find_nii_files,
    preprocess_image,
    preprocess_parallel,
)

from cohorts.mamamia import MamaMiaDataset  # noqa: E402


def main() -> int:
    """Run the parallel-vs-serial preprocessing equivalence check."""
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        images_dir = tmp / "images"
        # 3 cases, varied tiny shapes (sitk array order is z,y,x).
        cases = {
            "CASE_001": (8, 16, 16),
            "CASE_002": (6, 14, 18),
            "CASE_003": (10, 12, 12),
        }
        for cid, shape in cases.items():
            cdir = images_dir / cid
            cdir.mkdir(parents=True)
            arr = rng.random(shape).astype(np.float32)
            sitk.WriteImage(
                sitk.GetImageFromArray(arr), str(cdir / f"{cid}_0000.nii.gz")
            )

        files = sorted(find_nii_files(str(images_dir)), key=lambda x: (x[0], str(x[1])))
        adapter = MamaMiaDataset(cohort=None, root=tmp)

        # Parallel (the feature under test)
        par_dir = tmp / "par"
        par_dir.mkdir()
        base_to_case, failed = preprocess_parallel(
            files, par_dir, workers=3, adapter=adapter
        )

        # Serial reference
        ser_dir = tmp / "ser"
        ser_dir.mkdir()
        for _cid, fp in files:
            base = Path(fp).name.replace(".nii.gz", "")
            preprocess_image(fp, ser_dir / f"{base}.npy", adapter=adapter)

        ok = len(failed) == 0 and len(base_to_case) == len(files)
        for base in base_to_case:
            a = np.load(par_dir / f"{base}.npy")
            b = np.load(ser_dir / f"{base}.npy")
            same = np.array_equal(a, b)
            print(f"{base}: shape={a.shape} identical_to_serial={same}")
            ok = ok and same

        if ok:
            print("\nPASS: parallel preprocessing matches serial, all files produced.")
            return 0
        print("\nFAIL: mismatch or missing outputs.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
