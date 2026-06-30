import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# Load MRI image (baseline)
print("Loading MRI and tumor mask...")
mri = nib.load('mri_images/DUKE_001/DUKE_001_0003.nii.gz')
tumor = nib.load('expert_tumors/DUKE_001.nii.gz')

# Get the data
mri_data = mri.get_fdata()
tumor_data = tumor.get_fdata()

# Get middle slice (z-direction)
mid_slice = mri_data.shape[2] // 2
mri_slice = mri_data[:, :, mid_slice]
tumor_slice = tumor_data[:, :, mid_slice]

print(f"MRI shape: {mri_data.shape}")
print(f"Tumor mask shape: {tumor_data.shape}")
print(f"Showing slice {mid_slice}")

# Create visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('DUKE_001: MRI Baseline with Expert Tumor Segmentation', fontsize=16, fontweight='bold')

# Row 1: Single slices
axes[0, 0].imshow(mri_slice, cmap='gray')
axes[0, 0].set_title('MRI (Baseline)')
axes[0, 0].axis('off')

axes[0, 1].imshow(tumor_slice, cmap='hot')
axes[0, 1].set_title('Expert Tumor Mask')
axes[0, 1].axis('off')

axes[0, 2].imshow(mri_slice, cmap='gray')
axes[0, 2].imshow(tumor_slice, cmap='Reds', alpha=0.5)
axes[0, 2].set_title('Overlay (Tumor in Red)')
axes[0, 2].axis('off')

# Row 2: Other slices
slice_2 = mri_data.shape[2] // 3
slice_3 = (2 * mri_data.shape[2]) // 3

axes[1, 0].imshow(mri_data[:, :, slice_2], cmap='gray')
axes[1, 0].imshow(tumor_data[:, :, slice_2], cmap='Reds', alpha=0.5)
axes[1, 0].set_title(f'Slice {slice_2}')
axes[1, 0].axis('off')

axes[1, 1].imshow(mri_data[:, :, mid_slice], cmap='gray')
axes[1, 1].imshow(tumor_data[:, :, mid_slice], cmap='Reds', alpha=0.5)
axes[1, 1].set_title(f'Slice {mid_slice} (Middle)')
axes[1, 1].axis('off')

axes[1, 2].imshow(mri_data[:, :, slice_3], cmap='gray')
axes[1, 2].imshow(tumor_data[:, :, slice_3], cmap='Reds', alpha=0.5)
axes[1, 2].set_title(f'Slice {slice_3}')
axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('visualization.png', dpi=150, bbox_inches='tight')
print("\nVisualization saved to: visualization.png")
print("You can open it with any image viewer!")

plt.show()
