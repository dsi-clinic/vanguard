import nrrd
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.morphology import skeletonize

GT_PATH = '/net/projects2/vanguard/gt_masks/Segmentation_Masks_NRRD/Breast_MRI_002/Segmentation_Breast_MRI_002_Dense_and_Vessels.seg.nrrd'
DL_PATH = '/net/projects2/vanguard/vessel_segmentations/ISPY2/ISPY2_410083/images/ISPY2_410083_0004_vessel_segmentation.npz'

def compare_vessels(gt_p, dl_p, threshold=0.20): 
    gt_data, _ = nrrd.read(gt_p)
    v_gt = (gt_data == 1).astype(np.float32)

    with np.load(dl_p) as data:
        v_dl_prob = data['vessel']
    
    v_dl = v_dl_prob.transpose(1, 0, 2) 
    v_dl = (v_dl > threshold).astype(np.float32)

    v_dl = resize(v_dl, v_gt.shape, order=0, preserve_range=True, anti_aliasing=False)
    v_dl = np.flipud(v_dl)

    print(f"Resized Shapes: GT: {v_gt.shape}, DL: {v_dl.shape}")
    print(f"Unique values in GT NRRD: {np.unique(gt_data)}")

    v_gt_centerline = skeletonize(v_gt.astype(np.uint8))

    intersection = np.sum(v_gt * v_dl)
    dice = (2. * intersection) / (np.sum(v_gt) + np.sum(v_dl))
    print(f"Dice Score: {dice:.4f}")

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(np.max(v_gt, axis=2), cmap='Blues')
    plt.title("Radiologist (GT)")

    plt.subplot(1, 3, 2)
    plt.imshow(np.max(v_dl, axis=2), cmap='Reds')
    plt.title(f"AI (DL) @ T={threshold}")

    plt.subplot(1, 3, 3)
    plt.imshow(np.max(v_dl, axis=2), cmap='gray', alpha=0.3)
    plt.imshow(np.max(v_gt_centerline, axis=2), cmap='hot')
    plt.title("GT Centerlines over AI")

    plt.savefig('vessel_task_output.png')
    print("Output saved to vessel_task_output.png")

if __name__ == "__main__":
    compare_vessels(GT_PATH, DL_PATH)