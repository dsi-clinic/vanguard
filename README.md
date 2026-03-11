# VAscular Networks for Graphical Understanding And Response Detection (VANGUARD)

## Project Background

A major challenge in breast cancer care is identifying whether treatment is effective early enough to adjust therapy. Standard imaging measures, like tumor shrinkage, often appear only after weeks or months. This delay can leave patients on ineffective regimens for too long.

Tumor-associated blood vessels offer a potential early biomarker. Vascular networks support tumor growth and change rapidly during therapy. This project represents vascular structures as graphs, where vessel branch points are nodes and vessel segments are edges with features such as length, diameter, and curvature.

By leveraging Graph Neural Networks (GNNs) and vascular graph analysis, we aim to detect treatment response earlier, enabling more personalized and adaptive cancer therapy.

## Project Goals

- **Pipeline Development:** Convert breast MRI data into graph-based representations of blood vessel networks.
- **GNN Training:** Analyze vascular graphs to capture local structural changes and global network patterns.
- **Baseline Comparison:** Benchmark GNN performance against traditional imaging and clinical features.
- **Feature Identification:** Determine which vascular features (tortuosity, connectivity, etc.) are most predictive of patient response.

## Team

Julia Luo, Bella Summe, Daniel Gong, Daniel Luna

## Installation

### 1. Environment Setup (Micromamba)

We use micromamba for fast, reproducible environment management.

**One-time micromamba install:**
```bash
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
./bin/micromamba shell init -s bash -r ~/micromamba
source ~/.bashrc
micromamba config append channels conda-forge
```

**Clone and create environment:**
```bash
micromamba config prepend channels conda-forge
micromamba config set channel_priority strict
git clone --recursive git@github.com:dsi-clinic/vanguard.git
cd vanguard
micromamba env create -y -n vanguard -f environment.yml
micromamba activate vanguard
```

Clone with `--recursive` so submodules (e.g. [vanguard-blood-vessel-segmentation](https://github.com/dsi-clinic/vanguard-blood-vessel-segmentation)) are included. All Python dependencies are in [environment.yml](environment.yml).

**Update environment:**
```bash
micromamba activate vanguard
micromamba env update -y -n vanguard -f environment.yml
```

For scripts run from the repo root, set `PYTHONPATH=.` or install in development mode: `pip install -e .`

### 2. Developer Quality Control

We use [ruff](https://docs.astral.sh/ruff/) to enforce style standards and [pre-commit](https://pre-commit.com/) to catch errors before they reach the repository.

```bash
pip install pre-commit && pre-commit install
pre-commit run --all-files
```

Work on feature branches; keep them up to date with `main`. All code must pass `pre-commit run --all-files` before merge.

## Config-Driven Methodology

As of this quarter, the pipeline is entirely config-based. Every experiment—from cohort selection to model hyperparameters—is fully documented and reproducible. Avoid hardcoding parameters in scripts; instead, reference a YAML file in the `configs/` directory.

## Repository Structure

| Directory | Description |
| --------- | ----------- |
| **configs/** | Central repository for YAML files defining experiment parameters. |
| **evaluation/** | Centralized system for k-fold CV, cohort selection, and metrics. See [evaluation/README.md](evaluation/README.md). |
| **graph_extraction/** | Pipeline for converting 3D vessel masks into graph-based JSONs. See [graph_extraction/README.md](graph_extraction/README.md) and [graph_pruning_centerline_extraction/README.md](graph_pruning_centerline_extraction/README.md) for 4D centerlines. |
| **ML-Pipeline/** | Training suite for GNNs and traditional ML models (Random Forest, Logistic Regression). See [ML-Pipeline/README.md](ML-Pipeline/README.md). |
| **batch_processing/** | Automated scripts for large-scale segmentation and extraction. See [batch_processing/README.md](batch_processing/README.md). |
| **slurm_submit_scripts/** | Pre-configured scripts for cluster-scale processing on HPC. See [slurm_submit_scripts/README.md](slurm_submit_scripts/README.md). |

Additional components: **non_imaging_baseline/**, **radiomics_baseline/**, **examples/**, **tests/**, **clinical_and_imaging_exploration/**—see in-repo READMEs for usage.

## End-to-End Workflow

1. **Vessel Segmentation & Graph Extraction**  
   Process raw MRI volumes into graph representations using parameters defined in your config. On the cluster: use `submit_batch_segmentation_array.sh` (vessel segmentations → `/net/projects2/vanguard/vessel_segmentations`) then `submit_graph_pruning_array.sh` (4D centerlines → `/net/projects2/vanguard/centerlines_4d`). See [batch_processing/README.md](batch_processing/README.md) and [graph_pruning_centerline_extraction/README.md](graph_pruning_centerline_extraction/README.md).

2. **Model Training**  
   Train GNN or traditional ML models by pointing the pipeline to your specific experiment config (e.g. in `configs/`). See [ML-Pipeline/README.md](ML-Pipeline/README.md).

3. **Comparison & Validation**  
   Compare your graph-based results against Radiomics and Clinical (Non-Imaging) baselines to assess model uplift. See [radiomics_baseline/README.md](radiomics_baseline/README.md), [non_imaging_baseline/](non_imaging_baseline/), and [examples/README.md](examples/README.md).

## Additional Resources

- [resources.md](resources.md): Project and domain references
- [workflow.md](workflow.md): Detailed workflow documentation
- [DataPolicy.md](DataPolicy.md): Data handling and privacy policies
