# Segmentation Slurm Scripts

These scripts submit the vessel-segmentation stage to compute nodes.

## Files

- `submit_batch_segmentation_array.sh`
  - preferred wrapper for cohort submission; computes array ranges automatically
- `submit_batch_segmentation_array.slurm`
  - array-task implementation used by the wrapper

## Recommended Entry Point

```bash
cd segmentation/slurm
./submit_batch_segmentation_array.sh
```

Optional overrides:

```bash
IMAGES_DIR=/path/to/images \
OUTPUT_DIR=/path/to/segmentations \
BREAST_MODEL=/path/to/breast_model.pth \
VESSEL_MODEL=/path/to/dv_model.pth \
./submit_batch_segmentation_array.sh
```

Use `START_INDEX` and `END_INDEX` if you only want to process part of the cohort.
