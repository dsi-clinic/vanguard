# Graph Extraction Slurm Scripts

These scripts submit cohort graph-extraction runs by calling `graph_extraction/run_skeleton_processing.py` once per study.

## Files

- `submit_tc4d_array.sh`
  - discovers studies under the segmentation root and submits the array job
- `submit_tc4d_array.slurm`
  - per-study worker that runs `run_skeleton_processing.py`

## Typical Use

```bash
cd graph_extraction/slurm
./submit_tc4d_array.sh
```

Custom input and output roots:

```bash
./submit_tc4d_array.sh \
  /net/projects2/vanguard/vessel_segmentations \
  /net/projects2/vanguard/centerlines_tc4d/studies
```

Small test run:

```bash
./submit_tc4d_array.sh /net/projects2/vanguard/vessel_segmentations /net/projects2/vanguard/centerlines_tc4d/studies --test
```

Optional environment toggles passed to the worker:

- `FEATURES_ONLY=1`
- `FORCE_FEATURES=1`
- `FORCE_SKELETON=1`
- `STRICT_QC=1`
- `NO_RENDER_MIP=1`
