# Deep Sets vessel 4D alignment (issue #120)

## Grid lineage

Per-case vessel time series come from tc4d-style NPZs under
`data_paths.vessel_segmentation_root` (for example
`/net/projects2/vanguard/vessel_segmentations/ISPY2/<case_id>/images/`).
Skeleton masks come from the centerline study directory as
`{case_id}_skeleton_4d_exam_mask.npy`, the same grid used in issue #119 QA.

## Axis order

`build_deepsets_dataset.py` loads 4D stacks with
`graph_extraction.core4d.load_time_series_from_files`, then aligns spatial axes
with `deepsets_volume_align.align_zyx_4d_to_shape`, which applies the same three
candidate `(z, y, x)` permutations as tumor mask loading and
`scripts/deepsets_alignment_check.py` (via `align_zyx_volume_to_shape`).

## Runtime check

After alignment, the builder requires `signal_4d.shape[1:]` to match the
skeleton shape. On mismatch, vessel dynamics are dropped for that case with a
warning and kinetic columns are zeroed (`kinetic_signal_ok` reflects validity).

## Manual QA on the cluster

From the repo root (vanguard env):

```bash
PYTHONPATH=. python scripts/deepsets_alignment_check.py \
  --config configs/deepsets_ispy2_pointfeat_geom_topo_dynamic.yaml \
  --use-vessel-segmentation-phases \
  --vessel-segmentation-root /net/projects2/vanguard/vessel_segmentations/ISPY2 \
  --case-ids ISPY2_100899,ISPY2_102011,ISPY2_102212
```

Confirm overlays show skeleton voxels on the expected phase slice. Any shape
errors from `align_zyx_volume_to_shape` indicate a case that cannot be aligned
to the skeleton grid and will not receive dynamics in the dataset build.
