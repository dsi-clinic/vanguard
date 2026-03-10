# ML Pipeline - PCR Prediction

## Overview
This project provides a machine learning pipeline designed to predict pCR (pathologic complete response) outcomes. It aggregates morphological features from image analysis and pairs them with clinical labels to train and evaluate predictive models.

## Features and Functionality
* **Config-Driven Orchestration:** Uses `ML-Pipeline/config_pcr.yaml` to manage data paths, feature selection, and model parameters.
* **Modular Feature Selection:** Supports toggling between Vascular, Clinical, and Radiomics feature sets.
* **Statistical Rigor:** Includes options for random baselines, DeLong tests, and ensemble stability runs.

## Configuration
The pipeline is driven by `ML-Pipeline/config_pcr.yaml`. This file is the primary entry point for experiment setup.

| Section | Description |
| :--- | :--- |
| `feature_toggles` | Boolean flags (`use_vascular`, `use_clinical`, `use_radiomics`) to include or exclude specific feature sets. |
| `data_paths` | Absolute paths to your feature directories and labels CSV. |
| `model_params` | Configuration for model selection (`rf`/`lr`), cross-validation splits, and training parameters. |

> **Note:** The `use_radiomics` toggle and corresponding radiomics loading logic are currently stubs and are pending implementation for next quarter.

## Usage
To run the pipeline using the configuration file:
```bash
python ML-Pipeline/pcr_prediction.py --config ML-Pipeline/config_pcr.yaml