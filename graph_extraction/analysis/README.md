# Graph Extraction Analysis

This directory is optional. It is not part of the production graph-extraction
pipeline.

Keep this directory small. The supported helpers here are:

- `build_features_with_metadata.py`
  - builds a flat feature table from saved morphometry JSONs and joins basic metadata
- `run_feature_qc.py`
  - runs column-level QC and simple biological sanity checks on that feature table

If you need a more complex exploratory analysis, prefer a notebook under
`analysis/` or a short one-off script outside the production pipeline rather
than growing this directory again.
