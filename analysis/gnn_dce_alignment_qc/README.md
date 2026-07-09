# GNN raw-DCE / skeleton alignment QC

Visual QC confirming that raw DCE-MRI volumes (loaded by `gnn.raw_dce`) land in
the same `(z, y, x)` voxel space as the saved skeleton/support masks, before
they are used to compute GNN kinetic node features
(`gnn.data_loader._attach_node_features`). This closes out the PR review
comment asking to fix the kinetic feature source (it was previously sampled
from `*_vessel_segmentation.npz`, a segmentation-model output, instead of the
raw DCE series) and to verify the coordinate alignment "carefully" before
trusting the new source.

## What to look at

Each PNG overlays one axial slice (the slice with the most skeleton voxels)
of the raw DCE volume (grayscale, 1st/99th percentile window) with:

- the support mask contour (cyan) -- the region `gnn.data_loader` estimates
  local vessel radius from,
- the skeleton voxels on that slice (colored dots, `peak_time` = raw
  argmax-enhancement index at that voxel).

The cyan contours should trace the bright, thread-like vascular structures
visible in the raw DCE background, not empty space or unrelated tissue -- that
is the alignment check. In all three cases below, the support-mask contour
sits directly on visible vessel structure, confirming the raw DCE volumes are
correctly aligned to the skeleton/support coordinate space (shape-matched via
`deepsets.volume_align.align_zyx_4d_to_shape`, the same convention already
used elsewhere in the repo for raw DCE / vessel-NPZ / tumor-mask alignment).

- `gnn_dce_alignment_DUKE_001_z212.png`
- `gnn_dce_alignment_DUKE_002_z254.png`
- `gnn_dce_alignment_DUKE_005_z205.png`

## How to regenerate

```bash
micromamba activate vanguard
PYTHONPATH=. python gnn/qc_dce_alignment.py \
  --centerline-root /gpfs/data/karczmar-lab/workspaces/saritbose/centerlines_tc4d/studies \
  --dce-root /gpfs/data/karczmar-lab/MAMA-MIA-syn60868042/images \
  --case-ids DUKE_001,DUKE_002,DUKE_005 \
  --out-dir analysis/gnn_dce_alignment_qc
```

- Inputs: `<centerline_root>/DUKE/<case_id>/{*_skeleton_4d_exam_mask.npy,
  *_skeleton_4d_exam_support_mask.npy, run_summary.json}` and
  `<dce_root>/<case_id>/<case_id>_NNNN.nii.gz` for each index in
  `run_summary.json["study_timepoints"]`.
- Git commit this was generated against: `cff0fb1264cb31926c55816e77048360f0e9aa13`
  (branch `implement-gnn`), plus the uncommitted raw-DCE kinetic-feature fix in
  this working tree (`gnn/raw_dce.py`, `gnn/data_loader.py`,
  `gnn/qc_dce_alignment.py`).
- No config file needed; paths are passed directly on the CLI. Runtime was
  ~10s wall-clock for all three cases (I/O-bound: a handful of ~25MB NIfTI
  reads + two ~32MB mask loads per case), run directly on the login node.
