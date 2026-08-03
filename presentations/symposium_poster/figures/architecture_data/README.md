# Architecture-figure inputs

Real-data inputs for `../architecture.tex`, the poster's two-stage architecture
figure. Generated -- never hand-edited. Regenerating overwrites everything here.

## Regenerate

```
sbatch gnn/slurm/submit_poster_architecture_data.slurm
  (CASE=DUKE_107 TARGET_NODES=28 SLAB_HALF_WIDTH=4)
```

Then rebuild the figure with `../build_architecture.sh`.

Source: `analysis/poster_architecture_data.py`, git commit
`2ce2b25bd4a7d321dc9dbd58c22b4039e06008e4`.

## What is here

| File | Contents |
| --- | --- |
| `graph.tex` | The one shared subgraph: voxel nodes/edges, junction and segment derivations, and per-frame node shading, as TikZ `\foreach` lists in unit coordinates. |
| `curves.tex` | Real baseline-referenced enhancement at three highlighted nodes, split into the input and forecast horizons. |
| `mri_frame_NN.png` | One real DCE-MRI frame each, slab max-projection, all on one shared intensity window. |
| `mri_centerline.png` | The last frame with the real centerline painted on. |
| `provenance.json` | Every parameter and count below, machine-readable. |

## Inputs

- Case `DUKE_107` from the **public MAMA-MIA Duke cohort**, chosen
  because the poster is public and UChicago slices must not be printed on it.
- Skeleton: `/gpfs/data/karczmar-lab/workspaces/saritbose/centerlines_tc4d/studies/DUKE/DUKE_107/DUKE_107_skeleton_4d_exam_mask.npy`
- DCE root: `/gpfs/data/karczmar-lab/MAMA-MIA-syn60868042/images`
- Frames: 5; input horizon
  3, the rest are forecast targets.

## Numbers

- Whole-case centerline: 18,952 voxels.
- Drawn subgraph: 30 nodes,
  31 edges, 11 junction/end
  nodes, 12 segments.
- Layout: principal-plane projection of the subgraph's 3D voxel coords.
- MRI slab z [169, 178], crop (x0,x1,y0,y1)
  [78, 102, 400, 419], shared window [165.97, 1770.5150000000003].

## Things a reader should know

- The node shading in the figure is **measured** enhancement, on one scale shared
  across frames. No forecaster is run here, so the "forecast" panel shows the
  target field the decoder is trained on, not a prediction.
- Duke exams carry 5 DCE phases. The method itself
  is applied to 13--24-frame ultrafast UChicago exams, so this figure understates
  the length of a real input sequence.
