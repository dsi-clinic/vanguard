# Batch Segmentation Processing Guide

This guide explains how to process all `.nii.gz` files in the `/images` directory and extract STEP-2 breast mask segmentations (`.npy` files).

## Overview

The batch processing pipeline:
1. **Finds all `.nii.gz` files** in patient subdirectories
2. **Preprocesses each file** (STEP-1): normalization, axis rotation, z-scoring
3. **Runs breast segmentation** (STEP-2): generates breast masks using trained model
4. **Collects all `.npy` files** for further processing (e.g., centerline extraction)

## Files Structure

```
/net/projects2/vanguard/MAMA-MIA-syn60868042/images/
├── DUKE_001/
│   ├── DUKE_001_0000.nii.gz
│   ├── DUKE_001_0001.nii.gz
│   └── ...
├── DUKE_002/
│   ├── DUKE_002_0000.nii.gz
│   └── ...
└── ...
```

## Scripts Available

### `batch_segmentation.py` - Batch Processing
Processes all files with parallel processing and progress tracking.

```bash
cd /path/to/vanguard
python batch_processing/batch_segmentation.py --help
```

**Key Options**:
- `--images-dir`: Source directory (default: `/net/projects2/vanguard/MAMA-MIA-syn60868042/images`)
- `--output-dir`: Output directory (default: `vessel_segmentations` relative to project root)
- `--max-workers`: Parallel workers (default: 4)
- `--patient-limit`: Limit for testing (e.g., `--patient-limit 10`)
- `--cleanup`: Clean temporary files

**Example Usage**:
```bash
# Quick test with 3 patients (similar to quick_batch_test.py)
python batch_processing/batch_segmentation.py --patient-limit 3 --output-dir test_breast_masks

# Test with 10 patients
python batch_processing/batch_segmentation.py --patient-limit 10 --output-dir test_masks

# Full processing with 8 parallel workers
python batch_processing/batch_segmentation.py --max-workers 8 --cleanup

# Custom paths
python batch_processing/batch_segmentation.py \
    --images-dir /path/to/images \
    --output-dir /path/to/output \
    --breast-model-path /path/to/breast_model.pth \
    --vessel-model-path /path/to/vessel_model.pth
```

## Output Format

### STEP-2 .npy Files
- **Format**: NumPy arrays with shape `(x, y, z)`
- **Values**: Binary mask (0 = background, 1 = breast tissue)
- **Naming**: `{patient_id}_{filename}_breast_mask.npy`

### Example Output Structure
```
breast_masks/
├── DUKE_001_DUKE_001_0000_breast_mask.npy
├── DUKE_001_DUKE_001_0001_breast_mask.npy
├── DUKE_002_DUKE_002_0000_breast_mask.npy
└── ...
```
## Cluster Submission (SLURM Array)

Use the array submit helper to launch one GPU job per file. This is the recommended way to process the full MAMA-MIA dataset in parallel.

```bash
cd /path/to/vanguard
FILES_PER_TASK=40 START_INDEX=0 END_INDEX=198 ARRAY_THROTTLE=20 \
./slurm_submit_scripts/submit_batch_segmentation_array.sh    
```

# Currently Processing Files

# For checking status of current run
squeue -u "$USER" -t R,PD -o "%.18i %.20j %.2t %.10M %R"     (CHECK NODE STATUS)

ls -t logs/batch-seg-array-*.out | head -n 1 | xargs -r tail -n 30 (tail job)

find /net/projects2/vanguard/vessel_segmentations -type f -newermt "2026-02-12 10:30:00" | wc -l (Round 1)

find /net/projects2/vanguard/vessel_segmentations -type f -newermt "2026-02-15 18:00:00" | wc -l (Round 2)

find /net/projects2/vanguard/vessel_segmentations -type f -newermt "2026-02-16 13:00:00" | wc -l (Round 3)

find /net/projects2/vanguard/vessel_segmentations -type f -mtime -1 | wc -l

Optional overrides via environment variables:

```bash
IMAGES_DIR=/net/projects2/vanguard/MAMA-MIA-syn60868042/images \
OUTPUT_DIR=/net/projects2/vanguard/vessel_segmentations \
BREAST_MODEL=/path/to/breast_model.pth \
VESSEL_MODEL=/path/to/dv_model.pth \
./slurm_submit_scripts/submit_batch_segmentation_array.sh
```

## Performance Estimates

Based on the notebook example:
- **Single file processing**: ~4-5 seconds
- **Total files**: ~1000+ files in the dataset
- **Estimated total time**: 1-2 hours with parallel processing
- **Storage**: ~50-100 MB per `.npy` file

## Next Steps for Centerline Extraction

Once you have all the breast mask `.npy` files:

1. **Convert to 3D Slicer format**: Convert `.npy` to `.nii.gz` or `.vtk`
2. **Load in 3D Slicer**: Import the breast masks
3. **Extract centerlines**: Use VMTK extension for centerline extraction
4. **Automate**: Create Python script to automate the VMTK workflow

## Troubleshooting

### Common Issues:
1. **Memory errors**: Reduce `--max-workers` or process in smaller batches
2. **Model not found**: Check `--model-path` points to `breast_model.pth`
3. **Permission errors**: Ensure write access to output directories
4. **CUDA errors**: The model may require GPU; check PyTorch installation

### Monitoring Progress:
- The script shows real-time progress with ✓/✗ indicators
- Check the summary at the end for success/failure counts
- Failed files are listed for debugging

## Integration with 3D Slicer

The generated `.npy` files can be used as input for your 3D Slicer centerline extraction:

```python
# Example: Convert .npy to .nii.gz for 3D Slicer
import numpy as np
import SimpleITK as sitk

# Load the breast mask
mask = np.load('breast_mask.npy')

# Convert to SimpleITK image
image = sitk.GetImageFromArray(mask.astype(np.uint8))

# Save as .nii.gz
sitk.WriteImage(image, 'breast_mask.nii.gz')
```

This workflow provides you with all the breast mask segmentations needed for the next phase of centerline extraction in 3D Slicer.

