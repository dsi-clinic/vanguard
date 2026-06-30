import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# Load pCR labels
pcr_data = pd.read_csv('/ess/scratch/scratch1/t-9sbose/pcr_labels.csv')
pcr_dict = dict(zip(pcr_data['case_id'], pcr_data['pcr']))

# Our 5 patients
patients = ['DUKE_001', 'DUKE_002', 'DUKE_005', 'DUKE_009', 'DUKE_010']

# Separate by pCR status
pcr_negative = []  # No response
pcr_positive = []  # Responded

for patient in patients:
    pcr_status = pcr_dict.get(patient, None)
    if pcr_status == 0:
        pcr_negative.append(patient)
    else:
        pcr_positive.append(patient)

print("pCR Negative (No Response):", pcr_negative)
print("pCR Positive (Responded):", pcr_positive)

# Load tumor and MRI data for each patient
tumor_data_dict = {}
mri_data_dict = {}
max_slices = {}

for patient in patients:
    tumor_path = f'expert_tumors/{patient}.nii.gz'
    mri_path = f'mri_images/{patient}/{patient}_0000.nii.gz'

    if os.path.exists(tumor_path) and os.path.exists(mri_path):
        tumor = nib.load(tumor_path)
        mri = nib.load(mri_path)

        tumor_data = tumor.get_fdata()
        mri_data = mri.get_fdata()

        tumor_data_dict[patient] = tumor_data
        mri_data_dict[patient] = mri_data

        # Find max tumor slice
        tumor_slices = np.sum(tumor_data, axis=(0, 1))
        max_slice = np.argmax(tumor_slices)
        max_slices[patient] = max_slice
        print(f"{patient}: max tumor at slice {max_slice}")

# Total number of patients (rows)
total_patients = len(patients)

# Create figure: 4 columns, 5 rows
fig = plt.figure(figsize=(20, 14))

# Title
fig.suptitle('Tumor Comparison by Treatment Response\n(Maximum Tumor Slice)',
             fontsize=18, fontweight='bold')

subplot_row = 1

# First: pCR Negative
print("\n--- pCR NEGATIVE (No Response) ---")
for patient in pcr_negative:
    tumor_slice = tumor_data_dict[patient][:, :, max_slices[patient]]
    mri_slice = mri_data_dict[patient][:, :, max_slices[patient]]

    print(f"Row {subplot_row}: {patient}")

    # Column 1: Patient name + pCR status (text only)
    ax1 = plt.subplot(total_patients, 4, subplot_row * 4 - 3)
    ax1.axis('off')
    ax1.text(0.5, 0.7, patient, fontsize=16, fontweight='bold',
             ha='center', va='center', transform=ax1.transAxes)
    ax1.text(0.5, 0.3, 'pCR Negative\n(No Response)', fontsize=12,
             ha='center', va='center', transform=ax1.transAxes,
             bbox=dict(boxstyle='round', facecolor='red', alpha=0.3),
             color='darkred', fontweight='bold')

    # Column 2: Tumor mask only
    ax2 = plt.subplot(total_patients, 4, subplot_row * 4 - 2)
    im2 = ax2.imshow(tumor_slice, cmap='hot')
    ax2.set_title('Tumor Mask Only', fontsize=11, fontweight='bold')
    ax2.axis('off')

    # Column 3: MRI only
    ax3 = plt.subplot(total_patients, 4, subplot_row * 4 - 1)
    ax3.imshow(mri_slice, cmap='gray')
    ax3.set_title('MRI (Baseline)', fontsize=11, fontweight='bold')
    ax3.axis('off')

    # Column 4: MRI + Tumor overlay
    ax4 = plt.subplot(total_patients, 4, subplot_row * 4)
    ax4.imshow(mri_slice, cmap='gray')
    ax4.imshow(tumor_slice, cmap='Reds', alpha=0.6)
    ax4.set_title('MRI + Tumor Overlay', fontsize=11, fontweight='bold')
    ax4.axis('off')

    subplot_row += 1

# Then: pCR Positive
print("\n--- pCR POSITIVE (Responded) ---")
for patient in pcr_positive:
    tumor_slice = tumor_data_dict[patient][:, :, max_slices[patient]]
    mri_slice = mri_data_dict[patient][:, :, max_slices[patient]]

    print(f"Row {subplot_row}: {patient}")

    # Column 1: Patient name + pCR status (text only)
    ax1 = plt.subplot(total_patients, 4, subplot_row * 4 - 3)
    ax1.axis('off')
    ax1.text(0.5, 0.7, patient, fontsize=16, fontweight='bold',
             ha='center', va='center', transform=ax1.transAxes)
    ax1.text(0.5, 0.3, 'pCR Positive\n(Responded)', fontsize=12,
             ha='center', va='center', transform=ax1.transAxes,
             bbox=dict(boxstyle='round', facecolor='green', alpha=0.3),
             color='darkgreen', fontweight='bold')

    # Column 2: Tumor mask only
    ax2 = plt.subplot(total_patients, 4, subplot_row * 4 - 2)
    im2 = ax2.imshow(tumor_slice, cmap='hot')
    ax2.set_title('Tumor Mask Only', fontsize=11, fontweight='bold')
    ax2.axis('off')

    # Column 3: MRI only
    ax3 = plt.subplot(total_patients, 4, subplot_row * 4 - 1)
    ax3.imshow(mri_slice, cmap='gray')
    ax3.set_title('MRI (Baseline)', fontsize=11, fontweight='bold')
    ax3.axis('off')

    # Column 4: MRI + Tumor overlay
    ax4 = plt.subplot(total_patients, 4, subplot_row * 4)
    ax4.imshow(mri_slice, cmap='gray')
    ax4.imshow(tumor_slice, cmap='Reds', alpha=0.6)
    ax4.set_title('MRI + Tumor Overlay', fontsize=11, fontweight='bold')
    ax4.axis('off')

    subplot_row += 1

plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.savefig('comparison_pcr_tumor_masks.png', dpi=150, bbox_inches='tight')
print("\nComparison visualization saved: comparison_pcr_tumor_masks.png")
plt.close()

print("\nDone!")
