# Segmentation

This directory contains the vessel-segmentation stage that runs before graph extraction.

The goal of this stage is simple: starting from breast MRI volumes, produce vessel segmentation masks that can be passed to the graph-extraction pipeline.

## Contents

- `batch_segmentation.py`
  - batch wrapper around the segmentation models
- `qa_pipeline_status.py`
  - lightweight status utility for checking progress
- `slurm/`
  - colocated Slurm scripts for cohort submission

## Typical Use

Most users should submit the array wrapper:

```bash
cd segmentation/slurm
./submit_batch_segmentation_array.sh
```

The wrapper discovers MRI volumes under `IMAGES_DIR` and submits array chunks that call `batch_segmentation.py` on compute nodes.

## Paths To Review Before Running

The Slurm wrappers default to shared cluster paths. Override these if needed:

- `IMAGES_DIR`
- `OUTPUT_DIR`
- `BREAST_MODEL`
- `VESSEL_MODEL`
- `FILES_PER_TASK`

## Outputs

Outputs are written under `OUTPUT_DIR` as segmentation volumes that feed the graph-extraction pipeline in `graph_extraction/`.
