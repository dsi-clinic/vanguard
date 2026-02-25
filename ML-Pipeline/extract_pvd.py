import os
import glob
import pandas as pd
from tqdm import tqdm
from pvd_feature import get_pvd_feature 

def run_pvd_pipeline(dataset_name):
    """
    Reproducible PVD extraction for DUKE or ISPY2.
    Handles the structural differences in mask storage between datasets.
    """
    
    config = {
        "DUKE": {
            "data_dir": "/net/projects2/vanguard/vessel_segmentations/DUKE",
            "mask_dir": "/net/projects2/vanguard/gt_masks/Segmentation_Masks_NRRD",
            "output": "pvd_results_duke.csv",
            "mask_ext": ".seg.nrrd"
        },
        "ISPY2": {
            "data_dir": "/net/projects2/vanguard/vessel_segmentations/ISPY2",
            "mask_dir": "/net/projects2/vanguard/MAMA-MIA-syn60868042/segmentations/expert",
            "output": "pvd_results_ispy2.csv",
            "mask_ext": ".nii.gz"
        }
    }

    c = config.get(dataset_name.upper())
    if not c:
        print(f"Error: Dataset {dataset_name} not supported.")
        return

    results = []
    patient_ids = [d for d in os.listdir(c['data_dir']) if os.path.isdir(os.path.join(c['data_dir'], d))]

    for p_id in tqdm(patient_ids, desc=f"Calculating {dataset_name} PVD"):
        try:
            cl_path = f"{c['data_dir']}/{p_id}/images/{p_id}_0000_vessel_segmentation.npz"
            
            if dataset_name.upper() == "DUKE":
                num_id = p_id.split('_')[-1]
                mask_search = f"{c['mask_dir']}/Breast_MRI_{num_id}/*{c['mask_ext']}"
                mask_files = glob.glob(mask_search)
                if not mask_files: continue
                tumor_path = next((f for f in mask_files if "Dense" in f), mask_files[0])
            
            else: #ISPY2
                tumor_path = os.path.join(c['mask_dir'], f"{p_id}{c['mask_ext']}")

            if not os.path.exists(cl_path) or not os.path.exists(tumor_path):
                continue

            pvd_value = get_pvd_feature(cl_path, tumor_path, mm_dilation=10)
            results.append({"Patient_ID": p_id, "PVD": pvd_value})
            
        except Exception as e:
            print(f"Skipping {p_id} due to error: {e}")

    df = pd.DataFrame(results)
    df.to_csv(c['output'], index=False)
    print(f"\nFinished {dataset_name} - saved {len(df)} results to {c['output']}")

if __name__ == "__main__":
    run_pvd_pipeline("DUKE")
    run_pvd_pipeline("ISPY2")