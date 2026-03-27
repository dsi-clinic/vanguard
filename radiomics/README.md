# Radiomics

This directory contains the radiomics-only pCR modeling workflow.

Radiomics is kept separate from the vessel-feature pipeline because it starts from image regions and hand-engineered image features rather than vessel graphs.

## Layout

- `configs/`
  - YAML experiment definitions
- `scripts/`
  - main Python entrypoints for extraction and model training
- `slurm/`
  - cluster submission wrappers for long radiomics jobs
- `analysis/`
  - plotting and site-level analysis helpers
- `labels.csv`, `splits_train_test_ready.csv`
  - curated labels and split assignments used by some experiments
- `pyradiomics_params*.yaml`
  - PyRadiomics parameter sets

## Core Entry Points

- `scripts/radiomics_extract.py`
  - extract radiomics features from images and masks
- `scripts/radiomics_train.py`
  - train and evaluate a radiomics model from extracted features
- `scripts/run_experiment.py`
  - run one YAML-defined experiment end to end
- `scripts/run_ablations.py`
  - run a sweep of related experiments
- `scripts/generate_kinetic_maps.py`
  - generate kinetic maps used by some radiomics experiments
- `scripts/make_labels_from_json.py`
  - build `labels.csv` from source metadata

## Typical Use

```bash
micromamba activate vanguard
cd radiomics
python scripts/run_experiment.py configs/exp_peri5_multiphase_logreg.yaml
```

For a sweep:

```bash
python scripts/run_ablations.py configs/sweep_2d_vs_3d.yaml --generated-dir configs/generated
```

## Slurm

For cohort-scale kinetic-map generation:

```bash
sbatch slurm/slurm_generate_kinetic_maps.sh
```
