# Analysis

This directory is optional. It is not part of the production pipeline.

**Notebooks only.** Keep only lightweight `.ipynb` notebooks here that help
explain or inspect results after the main workflows have run. Python scripts,
shell scripts, and `.slurm` files do not belong here even when they are one-off
diagnostics: put Python in `scripts/` or the owning package and sbatch scripts
in `slurm/`. The `.py` files still present in this directory predate the rule
and are pending migration.

Current notebooks:

- `deepsets_issue120_notebook.ipynb`
  - primary Issue #120 benchmark artifact with feature-regime tables, deferred launch commands, and generated figures
- `figures/deepsets_issue120/BENCHMARK_NOTE.md`
  - short written benchmark conclusions (interim + final) and 4D alignment notes for Issue #120
- `graph_weak_signal_diagnostic.ipynb`
  - diagnostic notebook for the older weak-signal feature-analysis workflow
- `graph_laterality_feature_analysis.ipynb`
  - compares feature distributions between unilateral and bilateral cases
- `clinical_imaging_exploration.ipynb`
  - general exploratory notebook for clinical and imaging metadata

If a notebook becomes important to the production workflow, move that logic into
Python code and document it in the main README instead of expanding this
directory.
