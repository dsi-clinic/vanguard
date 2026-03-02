import nrrd
import numpy as np
import matplotlib.pyplot as plt

GT_PATH = '/net/projects2/vanguard/gt_masks/Segmentation_Masks_NRRD/Breast_MRI_002/Segmentation_Breast_MRI_002_Dense_and_Vessels.seg.nrrd'
DL_PATH = '/net/projects2/vanguard/vessel_segmentations/DUKE_002_DUKE_002_0000_vessel_segmentation.npy'

def compare_segmentations(gt_p, dl_p):

    gt_data, _ = nrrd.read(gt_p)
    v_gt = (gt_data == 2).astype(np.float32) 

    v_dl_raw = np.load(dl_p) 
    
    v_dl = v_dl_raw[2, :, :, :].astype(np.float32) 

    print(f"Corrected Shapes! GT: {v_gt.shape}, DL: {v_dl.shape}")

    # Calculate Dice
    intersection = np.sum(v_gt * v_dl)
    dice = (2. * intersection) / (np.sum(v_gt) + np.sum(v_dl))
    print(f"\nDice Similarity Coefficient: {dice:.4f}")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(np.max(v_gt, axis=2), cmap='Blues')
    plt.title("Radiologist Vessels (GT)")
    
    plt.subplot(1, 2, 2)
    plt.imshow(np.max(v_dl, axis=2), cmap='Reds')
    plt.title("AI Vessels (DL)")
    
    plt.savefig('vessel_comparison.png')
    print("Plot saved as vessel_comparison.png")

if __name__ == "__main__":
    compare_segmentations(GT_PATH, DL_PATH)