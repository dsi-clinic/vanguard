# DUKE Final Symmetry-Line QC

This folder contains QC panels for the final DUKE symmetry-line side assignment
used by the vessel asymmetry work.

## Method

The current DUKE export uses `inner_walls` only. For each case, the breast mask
is projected into the same y/x MIP view used by the QC image. The script starts
from the image center and, for rows where the center lies in the space between
breasts, searches left and right to find the two inner breast-mask walls. The
case-level symmetry line is the median of those row-wise wall-pair midpoints.
Cases that cannot produce enough clean inner-wall rows are treated as QC
failures rather than being assigned by a second midline-placement method.

The output line is a single vertical split in projected y/x MIP space. It is
used to infer tumor side, contralateral side, side-specific skeleton counts, and
downstream ipsilateral/contralateral vessel features.

The generation code is contained in `tabular/duke_final_symmetry_lines.py` for
the DUKE final workflow; it does not require committing the older general
MAMA-MIA side-assignment scripts.

## Reproduce

Run from the repo root in the Vanguard environment:

```bash
MANIFEST=/path/to/shared/mama_mia_bilateral_unilateral_cancer_manifest.csv
micromamba run -n vanguard python tabular/duke_final_symmetry_lines.py \
  --manifest "${MANIFEST}" \
  --overwrite
micromamba run -n vanguard python scripts/summarize_duke_final_symmetry_lines.py
```

The manifest is required explicitly because it is shared input metadata, not a
file committed with this final export. Use the canonical shared manifest for
the DUKE/MAMA-MIA side-assignment cohort available in your environment.

The script reads:

- the manifest passed with `--manifest`
- DUKE DCE images under `/gpfs/data/karczmar-lab/MAMA-MIA-syn60868042/images`
- referenced tumor masks, breast masks, centerline masks, support masks, and
  morphometry JSON files from the manifest

The script writes:

- `tabular/duke_final_symmetry_lines.csv`
- `qc/duke_final_symmetry_lines/*_side_qc_mip.png`
- `qc/duke_final_symmetry_lines/index.html`
- `qc/duke_final_symmetry_lines/summary.md`

## Interpretation

- `side_split_method` should be `inner_walls` for every final DUKE case.
- `image_midline_x` is the x-coordinate of the vertical split in projected y/x
  MIP space.
- `tumor_side_confidence` reflects distance from the midline, coordinate-order
  confidence, and whether a breast mask was used.
- `inferred_tumor_side` and `inferred_contralateral_side` are derived from the
  tumor-mask centroid relative to `image_midline_x`.

The raw imaging data and masks are read-only inputs. The files in this folder
are derived QC artifacts.
