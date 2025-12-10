# `skeleton3d` module

This folder contains all code and experiments related to the 3D vessel
skeletonization model used in the Vanguard project. The goal of this code
is to take segmented 3D volumes of blood vessels and produce a
topology-preserving skeleton and graph representation that can be used
for downstream geometric and network analysis.

> **Note:** This folder was previously named `notebooks/`. It was renamed
> to `skeleton3d/` to better reflect its purpose.

## File overview

Fill in the descriptions below to keep this README up to date.

| File | Description |
| ---- | ----------- |
| `skeleton3d_usage` | (Example) Implementation of a topology-preserving 3D skeletonization algorithm. Each voxel is treated as a node with 26-connected neighbors; voxels are iteratively removed if their deletion does not break connectivity. Public entry point: `skeletonize3d(...)`. |
| `skeleton3d_utils` | Utility functions for the skeleton3d package. |
| `utils` | General utility functions for working ML pipelines using the MAMA MIA dataset |

## Typical workflow

1. Start from a segmented 3D vessel volume (binary or labeled).
2. Run the skeletonization routine(s) in this folder to obtain:
   - A skeleton volume (centerlines only).
   - Optionally, a graph representation of the vessel network.
3. Use downstream scripts/notebooks here to:
   - Visualize the skeleton and projections.
   - Export nodes/edges/attributes to JSON or other formats.
   - Compute geometric features along vessel centerlines.

## Notes for contributors

- When adding a new notebook or script, please:
  - Use a descriptive filename.
  - Add a one-line summary in the **File overview** table above.
  - Document any command-line interfaces or public functions in docstrings.
- If you change the location or name of this folder again, search the repo
  for `skeleton3d/` and update any hard-coded paths.

