# Graph Pruning Centerline Extraction

This folder contains all code and experiments related to the **graph pruning** method for 3D vessel
skeletonization used in the Vanguard project. The goal of this code
is to take segmented 3D volumes of blood vessels and produce a
topology-preserving skeleton and graph representation that can be used
for downstream geometric and network analysis.

> **Note:** This folder was previously named `notebooks/` and then `skeleton3d/`. It was renamed
> to `graph_pruning_centerline_extraction/` to better reflect its purpose and distinguish it from
> the thinning-based method.

## File overview

Fill in the descriptions below to keep this README up to date.

| File | Description |
| ---- | ----------- |
| `skeleton3d_usage` | (Example) Implementation of a topology-preserving 3D skeletonization algorithm. Each voxel is treated as a node with 26-connected neighbors; voxels are iteratively removed if their deletion does not break connectivity. Public entry point: `skeletonize3d(...)`. |
| `skeleton3d_utils` | Utility functions for the skeleton3d package. |
| `utils` | General utility functions for working ML pipelines using the MAMA MIA dataset |
| `batch_graph_pruning_centerlines.py` | Batch runner for graph pruning centerline extraction on vessel segmentation `.npy` files. |

## Typical workflow

1. Start from a segmented 3D vessel volume (binary or labeled).
2. Run the skeletonization routine(s) in this folder to obtain:
   - A skeleton volume (centerlines only).
   - Optionally, a graph representation of the vessel network.
3. Use downstream scripts/notebooks here to:
   - Visualize the skeleton and projections.
   - Export nodes/edges/attributes to JSON or other formats.
   - Compute geometric features along vessel centerlines.

## Batch graph pruning (SLURM array)

Use the SLURM submit helper to run graph pruning centerline extraction on the
vessel segmentation outputs.

```bash
cd /path/to/vanguard
FILES_PER_TASK=40 START_INDEX=0 END_INDEX=198 ARRAY_THROTTLE=20 \
./slurm_submit_scripts/submit_graph_pruning_array.sh
```

Optional overrides via environment variables:

```bash
INPUT_DIR=/net/projects2/vanguard/vessel_segmentations \
OUTPUT_DIR=/net/projects2/vanguard/graph_pruning_outdir \
THRESHOLD=0.5 \
PATTERN="*_vessel_segmentation.npy" \
RECURSIVE=0 \
FILES_PER_TASK=40 \
./slurm_submit_scripts/submit_graph_pruning_array.sh
```

## Notes for contributors

- When adding a new notebook or script, please:
  - Use a descriptive filename.
  - Add a one-line summary in the **File overview** table above.
  - Document any command-line interfaces or public functions in docstrings.
- If you change the location or name of this folder again, search the repo
  for `graph_pruning_centerline_extraction/` and update any hard-coded paths.

