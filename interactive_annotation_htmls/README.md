# Duke vessel-annotation QC — interactive HTMLs

Self-contained, browser-viewable 3D scenes that overlay, for each Duke case:

- the **radiologist vessel annotation**, split by distance to our centerline
  (≤2 mm green / 2–5 mm orange / >5 mm red = miss);
- our **skeleton / centerline**;
- our **vessel segmentation** (subsampled).

Every layer toggles on/off (legend click or the quick-view buttons), so you can
tell whether a red miss sits where we have **no segmentation** (an upstream model
miss) or where segmentation exists but the skeleton failed (a skeletonization gap).
Each `DUKE_###_interactive.html` in this directory is one such scene and opens in
any browser with no server (plotly.js is embedded).

## Files

| File | What it is |
| --- | --- |
| `duke_qc_viz.py` | Library + Phase-A QC (loading, alignment, distance histograms). All input/output paths live here. |
| `duke_qc_interactive.py` | Builds the interactive per-case HTMLs. Imports the helpers from `duke_qc_viz.py`. |
| `run_interactive.sbatch` | Slurm wrapper to render one case or all cases on the cluster. |
| `DUKE_###_interactive.html` | Pre-rendered scenes (committed for convenience). |

## Inputs and how to configure them

Every path is read from an environment variable; none is hardcoded to a personal
workspace.

The three **required** inputs live only on our private cluster and differ per
user, so there is no safe default — if one is unset the scripts exit immediately
with a message naming the variable, rather than silently reading a directory that
does not exist for whoever cloned the repo. Shared data and outputs keep defaults.

| Variable | Required? | Default | What it points to |
| --- | --- | --- | --- |
| `DUKE_ANN_ROOT` | **yes** | — | Radiologist annotation NRRDs (download — see below). |
| `DUKE_SEG_ROOT` | **yes** | — | Our vessel-probability segmentations (your pipeline output). |
| `DUKE_CL_ROOT` | **yes** | — | Our extracted skeletons (your pipeline output). |
| `MAMA_MIA_IMAGES` | no | `/gpfs/data/karczmar-lab/MAMA-MIA-syn60868042/images` | MAMA-MIA source NIfTIs — supply the physical grid. |
| `DUKE_QC_INTERACTIVE_OUT` | no | this directory | Where the interactive HTMLs are written. |
| `DUKE_QC_OUT` | no | `<repo>/interactive_annotation_htmls/phaseA_qc` | Phase-A output (distance arrays, histograms). |

The easiest way to set the required ones is the provided template:

```bash
cp interactive_annotation_htmls/duke_qc.env.example interactive_annotation_htmls/duke_qc.env
# edit duke_qc.env with your paths, then:
source interactive_annotation_htmls/duke_qc.env
```

`duke_qc.env` is gitignored (never commit real paths); `run_interactive.sbatch`
sources it automatically if present.

## Downloading the Duke vessel annotations

The radiologist annotations come from the **Duke-Breast-Cancer-MRI** supplement
on The Cancer Imaging Archive (TCIA), which ships per-case Slicer labelmaps
(`label 1 = Vessels`, `label 2 = Dense/FGT`).

1. Open the collection page:
   <https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/>
2. Under the supplementary data, download the segmentation-mask package
   **"Duke-Breast-Cancer-MRI-Supplement-v3"** (the NRRD segmentations).
3. Extract it. You should get a tree like:

   ```
   PKG - Duke-Breast-Cancer-MRI-Supplement-v3/
     Duke-Breast-Cancer-MRI-Supplement-v3/
       Segmentation_Masks_NRRD/
         Breast_MRI_002/
           Segmentation_Breast_MRI_002_Dense_and_Vessels.seg.nrrd
         Breast_MRI_021/
         ...
   ```

4. Point `DUKE_ANN_ROOT` at the `Segmentation_Masks_NRRD` directory, e.g.:

   ```bash
   export DUKE_ANN_ROOT="/path/to/PKG - Duke-Breast-Cancer-MRI-Supplement-v3/Duke-Breast-Cancer-MRI-Supplement-v3/Segmentation_Masks_NRRD"
   ```

   (or put it in your `duke_qc.env`).

This download is large imaging data — keep it outside git (do **not** commit it).

## Reproducing the HTMLs

From the repo root, with the `vanguard` environment active
(`micromamba activate vanguard`) and the inputs above resolvable:

```bash
# one case
python interactive_annotation_htmls/duke_qc_interactive.py DUKE_693

# every annotated case (the full set committed here)
python interactive_annotation_htmls/duke_qc_interactive.py ALL
```

On the cluster, submit via Slurm instead of running on the head node:

```bash
sbatch interactive_annotation_htmls/run_interactive.sbatch DUKE_693
sbatch interactive_annotation_htmls/run_interactive.sbatch ALL
```

Logs land in `logs/duke_qc_html-<jobid>.out`. HTMLs are written to
`DUKE_QC_INTERACTIVE_OUT` (this directory by default), regenerating the committed
files in place.

To (re)run the Phase-A alignment check + distance histograms:

```bash
python interactive_annotation_htmls/duke_qc_viz.py DUKE_002 DUKE_141 DUKE_693
```
