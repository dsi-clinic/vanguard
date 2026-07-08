# Segmentation Slurm Scripts

These scripts submit the vessel-segmentation stage to compute nodes.

## Files

- `submit_batch_segmentation_array.sh`
  - preferred wrapper for cohort submission; computes array ranges automatically
- `submit_batch_segmentation_array.slurm`
  - array-task implementation used by the wrapper
- `submit_batch_segmentation_smoke.slurm`
  - single-job GPU smoke test on a couple of files, writing to a separate
    output dir; validate afterwards with `../validate_outputs.py`. No `.sh`
    wrapper -- submit directly with `PROJECT_ROOT` set (see below)
- `submit_validate_speed_and_accuracy.slurm`
  - frozen/historical; submitted `../validate_speed_and_accuracy.py`, which
    depended on the now-removed old pipeline's CLI and is kept only as a
    record of how `../validation_results.md` was produced. No `.sh` wrapper --
    submit directly with `PROJECT_ROOT` set (see below)

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

## Submitting the smoke / validation scripts directly

`submit_batch_segmentation_smoke.slurm` and
`submit_validate_speed_and_accuracy.slurm` have no `.sh` wrapper, so
`PROJECT_ROOT` must be set explicitly at submission time -- it is **not**
derived from `$SLURM_SUBMIT_DIR` (that's your shell's cwd when you run
`sbatch`, which is wrong if you've `cd`'d into `segmentation/slurm`) or from
the script's own path (`sbatch` may run a spooled copy, so the path isn't
reliably this checkout). From the repo root:

```bash
PROJECT_ROOT="$(pwd)" sbatch segmentation/slurm/submit_batch_segmentation_smoke.slurm
```

Both scripts fail fast with a clear error if `PROJECT_ROOT` isn't set.
