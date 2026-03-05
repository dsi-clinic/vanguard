# Graph Extraction

TC4D skeleton extraction and skeleton-to-graph morphometry for 4D vessel segmentations.

## Layout

- `run_skeleton_processing.py`: production entrypoint (tc4d-only) for exam-level skeleton extraction + morphometry.
- `processing.py`: shared production helpers for study loading, extraction, morphometry, and coverage MIP output.
- `tc4d.py`: production-only tc4d core implementation used by processing.
- `debug/run_compare_4d_vs_tc4d.py`: debug-only 4d-vs-tc4d comparison entrypoint.
- `debug/tc4d_compare_runtime.py`: debug compare helpers + CLI runtime.
- `debug/skeleton4d.py`: 4d baseline skeletonizer kept only for debug comparison.
- `core4d.py`: shared 4D study I/O utilities.
- `vessel_mip.py`: shared orthogonal MIP rendering + radiologist hit/miss summary.
- `skeleton3d.py`, `skeleton_to_graph.py`: core skeleton and graph/morphometry modules.

## Main Pipeline

Run production processing (tc4d):

```bash
python graph_extraction/run_skeleton_processing.py \
  --input-dir /net/projects2/vanguard/vessel_segmentations \
  --study-id ISPY2_202539 \
  --output-dir output
```

Behavior:
- If skeleton outputs already exist, they are reused unless `--force-skeleton` is set.
- If morphometry JSON already exists, it is reused unless `--force-features` is set.
- The pipeline writes `*_vessel_coverage_mip.png` by default (`--render-mip`, `--mip-dpi`).
- If DUKE/Breast annotations exist under `--radiologist-annotations-dir`, the MIP includes a radiologist row and hit/miss summary.

## Debug Comparison

Use this only for regression/debug investigations (`4d` baseline vs `tc4d`):

```bash
python graph_extraction/debug/run_compare_4d_vs_tc4d.py \
  --input-dir /net/projects2/vanguard/vessel_segmentations \
  --study-id ISPY2_202539 \
  --output-dir debug_4d_vs_tc4d
```

Useful runtime flags for repeated debug loops:
- `--cache-dir ./.cache/compare_4d` and `--use-io-cache` (default on): cache aligned raw DCE and annotation mask loads.
- `--save-intermediate-masks`: opt-in write of support/manifold NPYs (off by default to reduce output size).
