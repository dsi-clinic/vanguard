import numpy as np
import nibabel as nib
import nrrd
from scipy import ndimage
import os

def get_pvd_feature(cl_path, tumor_mask_path, mm_dilation=10):
    cl_file = np.load(cl_path)
    cl_data = cl_file[cl_file.files[0]]
    
    try:
        if tumor_mask_path.endswith('.nii.gz') or tumor_mask_path.endswith('.nii'):
            img = nib.load(tumor_mask_path)
            tumor_data = img.get_fdata()
            spacing = img.header.get_zooms()[:3]
        else:
            tumor_data, header = nrrd.read(tumor_mask_path)
            if 'space directions' in header and header['space directions'] is not None:
                spacing = [np.linalg.norm(v) for v in header['space directions'] if v is not None]
            else:
                spacing = header.get('spacings', [1, 1, 1])
    except Exception as e:
        print(f"Error reading mask {os.path.basename(tumor_mask_path)}: {e}")
        return np.nan

    tumor_data = np.squeeze(tumor_data)
    if tumor_data.ndim > 3:
        tumor_data = tumor_data[..., 0]
        
    cl_binary = (cl_data > 0).astype(np.uint8)
    tumor_binary = (tumor_data > 0).astype(np.uint8)
    
    if cl_binary.shape != tumor_binary.shape:
        raise ValueError(f"Shape mismatch: Centerline {cl_binary.shape} vs Mask {tumor_binary.shape}")

    its = int(mm_dilation / spacing[0])
    dilated_tumor = ndimage.binary_dilation(tumor_binary, iterations=its)
    shell_mask = np.logical_and(dilated_tumor, np.logical_not(tumor_binary))
    
    pvd_count = np.sum(np.logical_and(cl_binary, shell_mask))
    
    return pvd_count