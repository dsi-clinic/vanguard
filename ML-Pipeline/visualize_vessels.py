import nrrd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from skimage.transform import resize
from skimage.morphology import skeletonize

P_ID = "041" 
GT_PATH = f'/net/projects2/vanguard/gt_masks/Segmentation_Masks_NRRD/Breast_MRI_{P_ID}/Segmentation_Breast_MRI_{P_ID}_Dense_and_Vessels.seg.nrrd'
DL_PATH = f'/net/projects2/vanguard/vessel_segmentations/DUKE/DUKE_{P_ID}/images/DUKE_{P_ID}_0000_vessel_segmentation.npz'

def compare_vessels(gt_p, dl_p):
    gt_data, _ = nrrd.read(gt_p)
    print(f"Original GT Shape: {gt_data.shape}")

    if gt_data.ndim == 4:
        gt_3d = gt_data[0, :, :, :]
    else:
        gt_3d = gt_data

    v_gt = (gt_3d == 1).astype(np.float32)

    with np.load(dl_p) as data:
        v_dl_prob = data['vessel']
    
    v_dl = v_dl_prob.transpose(1, 0, 2) 
    v_dl = (v_dl > 0.2).astype(np.float32)
    v_dl = resize(v_dl, v_gt.shape, order=0, preserve_range=True, anti_aliasing=False)
    v_dl = np.flipud(v_dl) 

    v_gt_centerline = skeletonize(v_gt.astype(np.uint8))

    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(np.max(v_gt, axis=2), cmap='Blues')
    plt.title(f"Radiologist GT (Patient {P_ID})")

    plt.subplot(1, 3, 2)
    plt.imshow(np.max(v_dl, axis=2), cmap='Reds')
    plt.title("AI Fixed Orientation")

    plt.subplot(1, 3, 3)
    plt.imshow(np.max(v_dl, axis=2), cmap='gray', alpha=0.3)
    plt.imshow(np.max(v_gt_centerline, axis=2), cmap='hot')
    plt.title("Overlay: AI vs GT Centerline")

    output_fn = f'vessel_output_{P_ID}.png'
    plt.savefig(output_fn)
    print(f"Output saved to {output_fn}")

if __name__ == "__main__":
    if os.path.exists(GT_PATH) and os.path.exists(DL_PATH):
        compare_vessels(GT_PATH, DL_PATH)
    else:
        print("Error: Files not found.")