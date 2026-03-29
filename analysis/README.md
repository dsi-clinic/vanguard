# Analysis

This directory is optional. It is not part of the production pipeline.

Keep only lightweight notebooks here that help explain or inspect results after
the main workflows have run.

Current notebooks:

- `graph_weak_signal_diagnostic.ipynb`
  - diagnostic notebook for the older weak-signal feature-analysis workflow
- `graph_laterality_feature_analysis.ipynb`
  - compares feature distributions between unilateral and bilateral cases
- `clinical_imaging_exploration.ipynb`
  - general exploratory notebook for clinical and imaging metadata

If a notebook becomes important to the production workflow, move that logic into
Python code and document it in the main README instead of expanding this
directory.
