"""One-off diagnostic: raw tumor-in-breast overlap fraction for every sampled case.

Regardless of whether it passes ``MIN_TUMOR_IN_BREAST_FRACTION``.
``analysis/breast_split_qc.py`` only records this fraction for cases that
pass the gate, so a 77% exclusion rate dominated by one reason
(``tumor_mostly_outside_breast_mask``) can't be diagnosed from its summary
CSV alone -- this recomputes it for every case to see whether the failures
cluster near 0 (real systematic issue) or just under the 0.5 threshold
(miscalibrated threshold).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis.breast_split_qc import _detect_axes, _study_dir
from gnn.breast_split import resolve_breast_mask_path
from tabular.duke_final_symmetry_lines import _load_array, _load_mask_matching_shape

TUMOR_MASK_DIR = Path(
    "/gpfs/data/karczmar-lab/MAMA-MIA-syn60868042/segmentations/expert"
)


def diagnose_one(case_id: str, dataset: str) -> dict[str, object]:
    """Recompute tumor/breast mask overlap stats for one case, no gating."""
    study_dir = _study_dir(dataset, case_id)
    skeleton = np.asarray(
        _load_array(study_dir / f"{case_id}_skeleton_4d_exam_mask.npy"), dtype=bool
    )
    z_axis, y_axis, x_axis = _detect_axes(case_id, dataset, skeleton.shape)

    breast_mask_path = resolve_breast_mask_path(dataset, case_id)
    breast_mask, breast_layout = _load_mask_matching_shape(
        breast_mask_path, expected_shape=skeleton.shape, threshold=0.5
    )
    tumor_mask, tumor_layout = _load_mask_matching_shape(
        TUMOR_MASK_DIR / f"{case_id}.nii.gz",
        expected_shape=skeleton.shape,
        threshold=0.5,
    )

    raw_tumor_voxels = int(tumor_mask.sum())
    in_breast_voxels = int((tumor_mask & breast_mask).sum())

    return {
        "case_id": case_id,
        "dataset": dataset,
        "breast_layout": breast_layout,
        "tumor_layout": tumor_layout,
        "skeleton_shape": str(skeleton.shape),
        "breast_voxels": int(breast_mask.sum()),
        "tumor_voxels": raw_tumor_voxels,
        "tumor_in_breast_voxels": in_breast_voxels,
        "tumor_in_breast_fraction": in_breast_voxels / raw_tumor_voxels
        if raw_tumor_voxels
        else float("nan"),
    }


def main() -> None:
    """Recompute overlap stats for every case in the existing QC sample."""
    summary = pd.read_csv("qc/breast_split_v1/breast_split_qc_summary.csv")
    rows = []
    for _, r in summary.iterrows():
        case_id, dataset = r["case_id"], r["dataset"]
        print(f"Diagnosing {case_id} ({dataset})...")
        rows.append(diagnose_one(case_id, dataset))
    out = pd.DataFrame(rows)
    out.to_csv("qc/breast_split_v1/overlap_diagnostic.csv", index=False)
    print(out.to_string(index=False))
    print()
    print(out["tumor_in_breast_fraction"].describe())
    print()
    print("breast_layout counts:", out["breast_layout"].value_counts().to_dict())
    print("tumor_layout counts:", out["tumor_layout"].value_counts().to_dict())


if __name__ == "__main__":
    main()
