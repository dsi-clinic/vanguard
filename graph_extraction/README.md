TC4D skeleton extraction and skeleton-to-graph / morphometry pipeline.

## Layout

- `run_skeleton_processing.py`: production entrypoint for tc4d exam-level skeleton extraction + morphometry.
- `processing.py`: shared production helpers for study loading, tc4d extraction, morphometry, and coverage MIP output.
- `tc4d.py`: production tc4d implementation.
- `core4d.py`: shared 4D study I/O utilities.
- `vessel_mip.py`: orthogonal MIP rendering + radiologist hit/miss summary.
- `skeleton3d.py`, `skeleton_to_graph.py`: core skeleton and graph/morphometry modules.
- `batch_process_4d.py`: batch tc4d processing for study-wide morphometry generation.
- `debug_compare_4d_vs_tc4d.py`: debug-only 4d-vs-tc4d comparison entrypoint.

## Production Pipeline

Run tc4d extraction and morphometry for one study:

```bash
python graph_extraction/run_skeleton_processing.py \
  --input-dir /net/projects2/vanguard/vessel_segmentations \
  --study-id ISPY2_202539 \
  --output-dir output
```

Behavior:

- Reuses existing skeleton outputs unless `--force-skeleton` is set.
- Reuses existing morphometry JSON unless `--force-features` is set.
- Writes `*_vessel_coverage_mip.png` by default.
- If matching DUKE/Breast MRI annotations exist, the MIP includes a radiologist row and hit/miss summary.

## Batch Processing

Run tc4d processing over many studies:

```bash
python graph_extraction/batch_process_4d.py \
  --input-dir /net/projects2/vanguard/vessel_segmentations \
  --output-dir report/4d_morphometry
```

Useful flags:

- `--study-ids ...`: restrict to a specific set of studies.
- `--skip-existing`: skip studies that already have morphometry JSON.
- `--study-range START END` and `--chunk-id`: array-task style partitioning.
- `--studies-per-task N`: process multiple studies in parallel within one task.
- `--merge-manifests`: merge `manifest_task_*.json` into one `manifest.json`.

## Debug Comparison

Use this only for regression/debug work comparing the older 4d baseline against tc4d:

```bash
python graph_extraction/debug_compare_4d_vs_tc4d.py \
  --input-dir /net/projects2/vanguard/vessel_segmentations \
  --study-id ISPY2_202539 \
  --output-dir debug_4d_vs_tc4d
```

Useful flags for repeated debug loops:

- `--cache-dir ./.cache/compare_4d`
- `--use-io-cache`
- `--save-intermediate-masks`
