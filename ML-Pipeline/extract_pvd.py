"""Module for extracting Perivascular Density (PVD) features from MRI datasets."""

from pathlib import Path
from typing import Any

import pandas as pd
import SimpleITK as sitk
from pvd_feature import get_pvd_feature
from tqdm import tqdm


def pvd_pipeline(dataset_name: str) -> None:
    """Orchestrate the extraction of PVD features for a given dataset.

    Args:
        dataset_name: The name of the dataset (e.g., "DUKE" or "ISPY2").
    """
    config: dict[str, dict[str, Any]] = {
        "DUKE": {
            "data_dir": Path("/net/projects2/vanguard/vessel_segmentations/DUKE"),
            "mask_dir": Path(
                "/net/projects2/vanguard/gt_masks/Segmentation_Masks_NRRD"
            ),
            "output": "pvd_results_duke.csv",
            "mask_ext": ".seg.nrrd",
        },
        "ISPY2": {
            "data_dir": Path("/net/projects2/vanguard/vessel_segmentations/ISPY2"),
            "mask_dir": Path(
                "/net/projects2/vanguard/MAMA-MIA-syn60868042/segmentations/expert"
            ),
            "output": "pvd_results_ispy2.csv",
            "mask_ext": ".nii.gz",
        },
    }

    c = config.get(dataset_name.upper())
    if not c:
        print(f"Dataset {dataset_name} not found in configuration.")
        return

    results: list[dict[str, Any]] = []
    patient_ids = [d.name for d in c["data_dir"].iterdir() if d.is_dir()]

    for p_id in tqdm(patient_ids, desc=f"Calculating {dataset_name} PVD"):
        if dataset_name.upper() == "DUKE":
            num_id = p_id.split("_")[-1]
            mask_folder = c["mask_dir"] / f"Breast_MRI_{num_id}"
            mask_files = list(mask_folder.glob(f"*{c['mask_ext']}"))
            if not mask_files:
                continue
            tumor_path = next(
                (f for f in mask_files if "Dense" in f.name), mask_files[0]
            )
        else:
            tumor_path = c["mask_dir"] / f"{p_id}{c['mask_ext']}"

        if not tumor_path.exists():
            continue

        try:
            cl_path = (
                c["data_dir"] / p_id / "images" / f"{p_id}_0000_vessel_segmentation.npz"
            )
            if not cl_path.exists():
                continue

            pvd_count = get_pvd_feature(cl_path, tumor_path, mm_dilation=10)

            tumor_img = sitk.ReadImage(str(tumor_path))
            tumor_img_int = sitk.Cast(tumor_img, sitk.sitkUInt8)
            stats = sitk.LabelShapeStatisticsImageFilter()
            stats.Execute(tumor_img_int)

            labels = stats.GetLabels()
            tumor_volume = (
                stats.GetNumberOfPixels(1)
                if 1 in labels
                else (stats.GetNumberOfPixels(labels[0]) if labels else 0)
            )

            relative_pvd = pvd_count / tumor_volume if tumor_volume > 0 else 0
            results.append({"Patient_ID": p_id, "PVD": relative_pvd})

        except Exception as e:
            print(f"Skipping {p_id} due to error: {e}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(c["output"], index=False)
    print(
        f"\nFinished {dataset_name} - saved {len(results_df)} results to {c['output']}"
    )


if __name__ == "__main__":
    pvd_pipeline("DUKE")
    pvd_pipeline("ISPY2")
