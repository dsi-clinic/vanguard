# Vascular Networks for Graphical Understanding And Response Detection (vanguard)

## Project Background

A major challenge in breast cancer care is figuring out whether treatment is working early enough to change course. Standard imaging measures such as tumor shrinkage often do not change until weeks or months into therapy. That delay can leave patients on an ineffective regimen for too long.

This project studies blood vessels around the tumor as a possible earlier signal of response. Tumors depend on nearby vessels for oxygen and nutrients, and those vessels can change during treatment. Breast dynamic contrast-enhanced MRI (DCE-MRI) is useful here because it shows both anatomy and how contrast moves through tissue over time.

Our central idea is to turn the vessel network into something we can measure more directly. We extract vessel centerlines, convert them into graphs, summarize the graph near the tumor, and use those summaries for pathologic complete response (pCR) modeling together with clinical and radiomics features.

## Project Goals

- Build a pipeline that turns breast MRI vessel segmentations into centerlines and graph representations.
- Extract vessel features that describe size, shape, connectivity, and contrast behavior near the tumor.
- Train and evaluate pCR models using clinical, vessel, and radiomics inputs.
- Measure which vessel feature groups appear to add signal beyond clinical and tumor-size baselines.

## Team

- Bella Summe
- Julia Luo
- Jose Cardona Arias
- Rebecca Wu

## Installation

Install micromamba once:

```bash
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
./bin/micromamba shell init -s bash -r ~/micromamba
source ~/.bashrc
micromamba config append channels conda-forge
```

Set up the repository once:

```bash
micromamba config prepend channels conda-forge
micromamba config set channel_priority strict
git clone --recursive git@github.com:dsi-clinic/vanguard.git
cd vanguard
micromamba env create -y -n vanguard -f environment.yml
micromamba activate vanguard
```

Update an existing environment:

```bash
micromamba activate vanguard
micromamba env update -y -n vanguard -f environment.yml
```

Clone with `--recursive` so the segmentation submodule is available.

## Data

### MAMA-MIA Dataset

This project uses the MAMA-MIA breast cancer MRI dataset. It combines 1,506 patients across four collections:

- I-SPY1
- I-SPY2
- NACT-Pilot
- Duke-Breast-Cancer-MRI

Relevant inputs for this repository:

- multi-timepoint breast DCE-MRI volumes
- expert 3D tumor segmentations
- harmonized clinical variables, including pCR labels

References:

- dataset page: <https://github.com/LidiaGarrucho/MAMA-MIA>
- reference: Garrucho et al., Synapse `syn60868042`

Runtime defaults now live in [`config.py`](config.py). The YAML files under [`configs/`](configs/) only need to override the values for a specific run. On the DSI cluster, many of those defaults point at shared paths under `/net/projects2/vanguard/...`. If your environment differs, override the relevant `data_paths` values in your YAML file instead of editing code.

## Repository Structure

This repository has four main workflows.

- `segmentation/`
  - runs the vessel-segmentation models that produce the binary vessel masks used downstream
- `graph_extraction/`
  - turns vessel masks into exam-level centerlines, graphs, vessel summaries, and tumor-focused feature JSONs
- `train_tabular.py`
  - trains tabular pCR models from clinical, vessel, and radiomics feature tables
- `radiomics/`
  - separate radiomics-only modeling workflow

Supporting pieces:

- `features/`
  - canonical definitions of the five modeling blocks: `clinical`, `tumor_size`, `morph`, `graph`, and `kinematic`
- `train_deepsets.py`
  - plain-PyTorch Deep Sets baseline over tumor-local vessel points that reuses the shared evaluator
- `evaluation/`
  - shared split generation, metrics, result aggregation, and output saving used across model families
- `modeling/`
  - helper scripts for array-parallel ablation jobs
- `configs/`
  - `ispy2.yaml` for standard tabular training
  - `ablation.yaml` for broad feature-block ablations
  - `independent_signal.yaml` for the focused independent-signal matrix
- `config.py`
  - central source of runtime defaults shared across tabular training, Deep Sets training, point-set building, and ablation runs
- `slurm/`
  - top-level Slurm submission wrappers for modeling runs
- `results/`
  - compact tracked result summaries
- `analysis/`
  - optional notebooks and lightweight exploratory analyses that are not part of the production pipeline
- `docs/`
  - reference documents that are helpful but not part of the main run path

## Segmentation

Start here:

- [`segmentation/README.md`](segmentation/README.md)
- [`segmentation/slurm/README.md`](segmentation/slurm/README.md)

Typical cohort submission:

```bash
cd segmentation/slurm
./submit_batch_segmentation_array.sh
```

Check these variables before running:

- `IMAGES_DIR`
- `OUTPUT_DIR`
- `BREAST_MODEL`
- `VESSEL_MODEL`

## Graph Extraction

Start here:

- [`graph_extraction/README.md`](graph_extraction/README.md)
- [`graph_extraction/slurm/README.md`](graph_extraction/slurm/README.md)

This repository has one supported graph-extraction pipeline, implemented in `graph_extraction/`. Internally, that pipeline uses the tc4d centerline method.

Single-study run:

```bash
micromamba activate vanguard
python graph_extraction/run_skeleton_processing.py \
  --study-id DUKE_041 \
  --input-dir /net/projects2/vanguard/vessel_segmentations/DUKE \
  --output-dir /net/projects2/vanguard/centerlines_tc4d/studies/DUKE/DUKE_041
```

Feature-only recompute from existing centerline outputs:

```bash
micromamba activate vanguard
python graph_extraction/run_skeleton_processing.py \
  --study-id DUKE_041 \
  --input-dir /net/projects2/vanguard/vessel_segmentations/DUKE \
  --output-dir /net/projects2/vanguard/centerlines_tc4d/studies/DUKE/DUKE_041 \
  --features-only \
  --force-features \
  --strict-qc \
  --no-render-mip
```

## Tabular pCR Modeling

Single training run:

```bash
micromamba activate vanguard
python train_tabular.py --config configs/ispy2.yaml --outdir experiments/debug_run
```

Primary training config:

- [`configs/ispy2.yaml`](configs/ispy2.yaml)

Config pattern:

- [`config.py`](config.py) defines the full default config shape
- YAML files in [`configs/`](configs/) override only the values for a given run
- in practice, most students only need to edit:
  - `data_paths.*`
  - `experiment_setup.name`
  - selected `feature_toggles`
  - selected `model_params`

Canonical feature blocks used by the tabular pipeline:

- `clinical`
  - non-imaging case-level and tumor metadata
- `tumor_size`
  - tumor size and local tumor-region vessel burden summaries
- `morph`
  - whole-network morphometry aggregates from the centerline graph
- `graph`
  - tumor-centered structural graph features
- `kinematic`
  - tumor-centered dynamic vessel features over time

The code definitions for those blocks live in [`features/`](features).

Before running on a new system, review these config fields in your YAML override:

- `data_paths.centerline_root`
- `data_paths.tumor_mask_root`
- `data_paths.patient_info_dir`
- `data_paths.clinical_excel`
- `data_paths.labels_csv`

## Deep Sets Modeling

Deep Sets is the current learned set-model baseline in this repo. It does not use graph message passing. Instead, it treats each case as a variable-length set of tumor-local vessel points, maps each point through a shared MLP, sums those embeddings, and then predicts pCR from the pooled case representation.

Reference:

- Zaheer et al., Deep Sets: <https://arxiv.org/abs/1703.06114>

Current entrypoints:

- [`build_deepsets_dataset.py`](build_deepsets_dataset.py)
  - builds one tumor-local point set per case from saved centerline and support masks
- [`train_deepsets.py`](train_deepsets.py)
  - trains the baseline Deep Sets classifier using the shared evaluator
- [`configs/deepsets_ispy2.yaml`](configs/deepsets_ispy2.yaml)
  - starting config for the I-SPY2 Deep Sets baseline
- [`slurm/submit_deepsets_pipeline.sh`](slurm/submit_deepsets_pipeline.sh)
  - submits the full Deep Sets workflow from one command

The starter point-level feature is intentionally minimal:

- a simple pointwise curvature proxy

Richer point features are opt-in through `feature_toggles.deepsets_point_features`.
The default remains `["curvature_rad"]` so existing configs keep the legacy
curvature-only payload. Example expanded feature list:

```yaml
feature_toggles:
  deepsets_point_features:
    - curvature_rad
    - signed_distance_tumor_mm
    - abs_distance_tumor_mm
    - inside_tumor
    - skeleton_node_degree
    - is_endpoint
    - is_chain
    - is_junction
    - offset_from_tumor_centroid_norm_x
    - offset_from_tumor_centroid_norm_y
    - offset_from_tumor_centroid_norm_z
    - direction_to_tumor_centroid_x
    - direction_to_tumor_centroid_y
    - direction_to_tumor_centroid_z
    - local_vessel_radius_mm
```

When `local_vessel_radius_mm` is requested, the builder expects
`{case_id}_skeleton_4d_exam_support_mask.npy` beside the skeleton mask. Cases
without a usable support mask are skipped and counted in the build logs.

Before running on a new system, review:

- `data_paths.centerline_root`
- `data_paths.tumor_mask_root`
- `data_paths.patient_info_dir`
- `data_paths.clinical_excel`
- `data_paths.labels_csv`

Typical run on the DSI cluster:

```bash
cd slurm
CONFIG=../configs/deepsets_ispy2.yaml \
OUT_ROOT=../experiments/deepsets_ispy2_test1 \
./submit_deepsets_pipeline.sh
```

The wrapper writes a run-local config under the output directory and fills in
`data_paths.deepsets_manifest_csv` automatically after the dataset build step.

Internally, the wrapper chains three dependent Slurm stages:

- parallel point-set building
- manifest merging
- model training

- `data_paths.deepsets_manifest_csv` if you already built the dataset
- or rerun [`build_deepsets_dataset.py`](build_deepsets_dataset.py) from the current centerline outputs

## Evaluation Framework

The `evaluation/` package is the shared comparison layer for this repo. It creates train/validation splits, computes metrics, saves fold outputs, and keeps the output format consistent across different model families.

Current users:

- `train_tabular.py`
  - tabular clinical, vessel, and radiomics models
- `train_deepsets.py`
  - Deep Sets baseline over tumor-local vessel point sets

Start here:

- [`evaluation/README.md`](evaluation/README.md)

## Independent-Signal Matrix

This experiment asks a practical question: after accounting for clinical variables and tumor size, do the vessel feature groups still help?

Config:

- [`configs/independent_signal.yaml`](configs/independent_signal.yaml)

How to modify the experiment:

- edit `ablation_arms` in [`configs/independent_signal.yaml`](configs/independent_signal.yaml) to change which block combinations are tested
- edit `baseline_arm_name` in the same file if you want deltas reported against a different reference arm
- keep the canonical block names:
  - `clinical`
  - `tumor_size`
  - `morph`
  - `graph`
  - `kinematic`

Recommended Slurm submission:

```bash
cd slurm
./submit_independent_signal_matrix_array.sh
```

Outputs:

- `experiments/<run_name>/ablation_summary.csv`
- `experiments/<run_name>/ablation_fold_auc.csv`
- `experiments/<run_name>/ablation_auc_summary.png`

Current tracked checkpoint:

- [`results/independent_signal_q3_summary.csv`](results/independent_signal_q3_summary.csv)
- [`results/independent_signal_q3_auc_summary.png`](results/independent_signal_q3_auc_summary.png)

Current result summary:

- baseline `clinical + tumor size`: `0.572 +/- 0.041`
- `+ morph`: `0.591 +/- 0.033`
- `+ graph`: `0.588 +/- 0.055`
- `+ kinematic`: `0.594 +/- 0.043`
- `+ graph + kinematic`: `0.594 +/- 0.051`
- `+ all vessel blocks`: `0.596 +/- 0.032`

Interpretation:

- all three vessel families improved mean AUC over the `clinical + tumor size` baseline in this rerun
- the best mean result came from the full vessel block
- the gains are modest, so future work should focus on more stable feature definitions and cleaner selection within each block

Tracked q3 summary figure:

![independent signal q3 auc summary](results/independent_signal_q3_auc_summary.png)

## Radiomics

Radiomics is maintained as a separate modeling workflow.

- [`radiomics/README.md`](radiomics/README.md)

## Analysis Utilities

Optional exploratory notebooks live under:

- [`analysis/`](analysis)

Optional graph-extraction analysis helpers live under:

- [`graph_extraction/analysis/`](graph_extraction/analysis)

## Running On The Cluster

- Use the `vanguard` micromamba environment for Python commands.
- Use the headnode only for editing, inspection, submission, and log review.
- Submit non-trivial extraction and modeling jobs through Slurm.
- Treat shared cluster paths in YAML files as editable defaults.

## Additional Documentation

- [`docs/data_policy.md`](docs/data_policy.md)
- [`docs/resources.md`](docs/resources.md)
- [`docs/workflow.md`](docs/workflow.md)
