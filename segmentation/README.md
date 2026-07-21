# Segmentation

This directory contains the vessel-segmentation stage that runs before graph extraction.

The goal of this stage is simple: starting from breast MRI volumes, produce vessel segmentation masks that can be passed to the graph-extraction pipeline.

Inference itself (the model forward passes) lives in the pinned
`vanguard-blood-vessel-segmentation` submodule and is not modified here. This
directory is the wrapper/orchestration layer: file discovery, preprocessing,
batching, output layout, and Slurm submission.

`batch_segmentation.py` runs breast (STEP-2) and vessel (STEP-3) inference
**in-process** (each model loaded once, kept on the GPU) with **batched**,
**AMP** inference and **parallel** STEP-1 preprocessing. It replaced an earlier
subprocess-per-file implementation that shelled out to the submodule's
`predict.py` twice per file; that implementation was validated against this
one (16.15x mean speedup, Dice 0.9998-1.0 on a 7-file sample — see
`validation_results.md`) before being removed.

## Contents

- `batch_segmentation.py`
  - the batch-segmentation driver: discovery, parallel preprocessing,
    in-process batched inference, output layout
- `predict_fast.py`
  - batched + AMP inference functions used by `batch_segmentation.py`
- `qa_pipeline_status.py`
  - lightweight status utility for checking progress
- `validate_outputs.py`
  - ongoing QC: compares pipeline outputs against a ground-truth run (Dice,
    probability diffs)
- `validate_speed_and_accuracy.py`
  - **frozen/historical** — produced the numbers in `validation_results.md` by
    comparing this pipeline against the now-removed old subprocess pipeline;
    kept as a record of methodology, not expected to run
- `validation_results.md`
  - the recorded speedup/accuracy validation (16.15x, Dice 0.9998-1.0)
- `tests/`
  - CPU-safe correctness tests (`test_batching_equiv.py`,
    `test_preprocess_parallel.py`) proving the batched/parallel paths are
    bit-identical to naive reference implementations
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

- `IMAGES_DIR` (used to size the array; discovery itself goes through the
  dataset adapter, see below)
- `OUTPUT_DIR`
- `BREAST_MODEL`
- `VESSEL_MODEL`
- `FILES_PER_TASK`
- `MAMAMIA_ROOT` / `DATASET_COHORT` — passed to `batch_segmentation.py` as
  `--dataset-root`/`--dataset-cohort` (required alongside `--dataset-name`,
  see `cohorts/README.md`); `DATASET_COHORT` unset means all four MAMA-MIA
  cohorts combined.

## Outputs

Outputs are written under `OUTPUT_DIR` as segmentation volumes that feed the graph-extraction pipeline in `graph_extraction/`.
