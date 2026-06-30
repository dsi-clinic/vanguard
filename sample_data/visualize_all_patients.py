import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# Load pCR labels
pcr_data = pd.read_csv('/ess/scratch/scratch1/t-9sbose/pcr_labels.csv')
pcr_dict = dict(zip(pcr_data['case_id'], pcr_data['pcr']))

# List of patients to visualize
patients = ['DUKE_001', 'DUKE_002', 'DUKE_005', 'DUKE_009', 'DUKE_010']

for patient in patients:
    print(f"\nProcessing {patient}...")

    # Load MRI image (baseline - timepoint 0000)
    mri_path = f'mri_images/{patient}/{patient}_0000.nii.gz'
    tumor_path = f'expert_tumors/{patient}.nii.gz'

    if not os.path.exists(mri_path) or not os.path.exists(tumor_path):
        print(f"  Skipping - files not found")
        continue

    mri = nib.load(mri_path)
    tumor = nib.load(tumor_path)

    # Get the data
    mri_data = mri.get_fdata()
    tumor_data = tumor.get_fdata()

    # Find slice with maximum tumor extent
    tumor_slices = np.sum(tumor_data, axis=(0, 1))
    max_tumor_slice = np.argmax(tumor_slices)

    # Get pCR status
    pcr_status = pcr_dict.get(patient, None)
    pcr_label = "pCR Positive (Complete Response)" if pcr_status == 1 else "pCR Negative (No Response)"
    pcr_color = "green" if pcr_status == 1 else "red"

    print(f"  MRI shape: {mri_data.shape}")
    print(f"  Maximum tumor at slice: {max_tumor_slice}")
    print(f"  pCR Status: {pcr_label}")

    # Get slices around the max tumor location
    slice_before = max(0, max_tumor_slice - 3)
    slice_middle = max_tumor_slice
    slice_after = min(mri_data.shape[2] - 1, max_tumor_slice + 3)

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Title with pCR information
    title = f'{patient}: MRI with Expert Tumor Segmentation\n{pcr_label}'
    fig.suptitle(title, fontsize=16, fontweight='bold', color=pcr_color)

    # Row 1: Three individual views
    axes[0, 0].imshow(mri_data[:, :, slice_before], cmap='gray')
    axes[0, 0].imshow(tumor_data[:, :, slice_before], cmap='Reds', alpha=0.5)
    axes[0, 0].set_title(f'Slice {slice_before} (-3)')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(mri_data[:, :, slice_middle], cmap='gray')
    axes[0, 1].imshow(tumor_data[:, :, slice_middle], cmap='Reds', alpha=0.5)
    axes[0, 1].set_title(f'Slice {slice_middle} (MAX TUMOR)')
    axes[0, 1].axis('off')
    # Red border for max tumor slice
    axes[0, 1].spines['top'].set_color('red')
    axes[0, 1].spines['bottom'].set_color('red')
    axes[0, 1].spines['left'].set_color('red')
    axes[0, 1].spines['right'].set_color('red')
    for spine in axes[0, 1].spines.values():
        spine.set_linewidth(3)

    axes[0, 2].imshow(mri_data[:, :, slice_after], cmap='gray')
    axes[0, 2].imshow(tumor_data[:, :, slice_after], cmap='Reds', alpha=0.5)
    axes[0, 2].set_title(f'Slice {slice_after} (+3)')
    axes[0, 2].axis('off')

    # Row 2: Just the key slice in different views
    mri_slice = mri_data[:, :, slice_middle]
    tumor_slice = tumor_data[:, :, slice_middle]

    axes[1, 0].imshow(mri_slice, cmap='gray')
    axes[1, 0].set_title('MRI Only')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(tumor_slice, cmap='hot')
    axes[1, 1].set_title('Tumor Mask Only')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(mri_slice, cmap='gray')
    axes[1, 2].imshow(tumor_slice, cmap='Reds', alpha=0.6)
    axes[1, 2].set_title('Overlay (Tumor in Red)')
    axes[1, 2].axis('off')

    # Add pCR status box at bottom
    pcr_text = f'pCR Status: {pcr_label}'
    fig.text(0.5, 0.02, pcr_text, ha='center', fontsize=14,
             bbox=dict(boxstyle='round', facecolor=pcr_color, alpha=0.3, edgecolor=pcr_color, linewidth=2),
             fontweight='bold', color=pcr_color)

    # Save
    output_file = f'{patient}_visualization.png'
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()

print("\n" + "="*60)
print("SUMMARY:")
print("="*60)
for patient in patients:
    pcr_status = pcr_dict.get(patient, None)
    pcr_label = "pCR POSITIVE (Responded)" if pcr_status == 1 else "pCR NEGATIVE (No Response)"
    print(f"{patient}: {pcr_label}")
print("="*60)
print("\nAll visualizations created!")
print("\nFiles created:")
for patient in patients:
    print(f"  - {patient}_visualization.png")
