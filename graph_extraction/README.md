# Graph Extraction

This directory contains the supported vessel-to-graph pipeline for this repository.

In plain terms, this stage takes vessel segmentations, finds vessel centerlines, converts those centerlines into a graph, and then writes summaries that the downstream models can use.

Internally, the centerline stage uses the tc4d method. New students do not need to understand every tc4d implementation detail before using this pipeline, but they should know what the main outputs mean.

Optional helper scripts for flat-table analysis live in
`graph_extraction/analysis/`. The files in the top level of `graph_extraction/`
are the production path.

## Module Layout

The production code is split by responsibility:

- `pipeline.py`
  - top-level study runner used by `run_skeleton_processing.py`
- `masks.py`
  - loading segmentation arrays, tumor masks, and radiologist masks into one shared orientation
- `graph_outputs.py`
  - turning saved skeleton/support masks into graph JSON outputs
- `skeleton_to_graph_primitives.py`
  - low-level graph-building and segment-measurement utilities used by `graph_outputs.py`
- `feature_stats.py`
  - shared geometry, distance-shell, and summary-stat helpers used by the feature builders

The actual feature definitions now live in the top-level `features/` package:

- `features/graph.py`
  - tumor-centered structural vessel features and the graph JSON loader
- `features/kinematic.py`
  - tumor-centered dynamic vessel features and the kinematic JSON extractor
- `features/tumor_size.py`
  - tumor-mask and peritumoral shell-size features
- `features/morph.py`
  - whole-network morphometry features from `*_morphometry.json`
- `features/clinical.py`
  - clinical feature definitions used by the tabular model

## What This Stage Produces

For each study, the pipeline can write:

- a `support` mask
  - the 3D vessel region that the centerline is allowed to live inside
- a `skeleton` mask
  - a one-voxel-wide centerline representation of the vessel network
- a `morphometry` JSON
  - graph-style summaries such as segment length, estimated radius, and branch structure
- a `tumor_graph_features` JSON
  - higher-level features that describe how vessels behave near the tumor
- optional debug images
  - vessel coverage MIPs and other inspection outputs

## Environment and Compute

- Activate `vanguard` before running Python commands.
- Use the headnode only for editing, inspection, job submission, and log review.
- Submit non-trivial extraction runs through Slurm.

## What To Know Before You Start

- The shared processed cohort currently lives under `/net/projects2/vanguard/centerlines_tc4d/studies`.
- If you only want to work on downstream modeling, you can usually start from those existing outputs.
- If you need to regenerate graph features for a study, use `graph_extraction/run_skeleton_processing.py`.

## Single-Study Commands

Set paths first:

```bash
export STUDY_ID=DUKE_041
export SEG_ROOT=/net/projects2/vanguard/vessel_segmentations/DUKE
export OUTDIR=/net/projects2/vanguard/centerlines_tc4d/studies/DUKE/${STUDY_ID}
```

Full centerline extraction plus feature generation:

```bash
micromamba activate vanguard
python graph_extraction/run_skeleton_processing.py \
  --study-id "${STUDY_ID}" \
  --input-dir "${SEG_ROOT}" \
  --output-dir "${OUTDIR}"
```

Fast iteration on graph and tumor features only, reusing existing centerline outputs:

```bash
micromamba activate vanguard
python graph_extraction/run_skeleton_processing.py \
  --study-id "${STUDY_ID}" \
  --input-dir "${SEG_ROOT}" \
  --output-dir "${OUTDIR}" \
  --features-only \
  --force-features \
  --strict-qc \
  --no-render-mip
```

`--features-only` requires these existing files in `OUTDIR`:

- `<study>_skeleton_4d_exam_mask.npy`
- `<study>_skeleton_4d_exam_support_mask.npy`

## Main Outputs

Core files written to `OUTDIR`:

- `<study>_skeleton_4d_exam_mask.npy`
- `<study>_skeleton_4d_exam_support_mask.npy`
- `<study>_center_manifold_4d_mask.npy` when enabled
- `<study>_morphometry.json`
- `<study>_tumor_graph_features.json`
- `<study>_vessel_coverage_mip.png` when enabled
- `run_summary.json`

## What The Feature JSONs Mean

`<study>_morphometry.json` contains lower-level graph measurements such as:

- connected components
- branch points
- segment lengths
- estimated vessel radius along each segment

`<study>_tumor_graph_features.json` contains higher-level summaries intended for modeling. The current groups are:

- vessel amount near the tumor
  - how much vessel length or estimated vessel volume is inside or near the tumor
- boundary-crossing features
  - whether vessel segments pass from outside the tumor to inside it
- per-shell summaries
  - vessel behavior in distance bands such as inside tumor, 0 to 2 mm away, 2 to 5 mm away, and so on
- topology summaries
  - counts or densities of endpoints, branch points, and loops
- caliber and shape summaries
  - vessel radius, curvature, and tortuosity near the tumor
- kinetic summaries
  - how quickly and how strongly vessel segments enhance over time in DCE-MRI

## QC Checks

The current feature path includes a few explicit consistency checks:

- duplicate segment paths are removed so the same segment is not counted twice in opposite directions
- branch points count all nodes with degree 3 or greater
- support masks are repaired if the saved centerline falls partly outside the support volume
- quality-check counters are written to `run_summary.json`
- `--strict-qc` turns invalid radius-related values into a hard failure instead of a warning

## Cohort Runs

For full-cohort reruns, use the Slurm wrappers in [`graph_extraction/slurm/README.md`](slurm/README.md).
