# VAscular Networks for Graphical Understanding And Response Detection (vanguard)

## Project Background

A major challenge in breast cancer care is identifying whether treatment is effective early enough to adjust therapy. Standard imaging measures, like tumor shrinkage, often appear only after weeks or months. This delay can leave patients on ineffective regimens for too long.

Tumor-associated blood vessels offer a potential early biomarker. Vascular networks support tumor growth and change rapidly during therapy. Dynamic contrast-enhanced MRI (DCE-MRI) can capture these changes, and recent work suggests that network properties—branching, tortuosity, density, and connectivity—may reveal subtle treatment responses that pixel-level metrics miss.

This project represents vascular structures as graphs, where vessel branch points are nodes and vessel segments are edges with features such as length, diameter, and curvature. Graph Neural Networks (GNNs) provide a natural framework for learning predictive patterns from these structures. By leveraging vascular graph analysis, we aim to detect treatment response earlier, enabling more personalized and adaptive cancer therapy.

## Project Goals

- Develop a computational pipeline to convert breast MRI data into graph-based representations of blood vessel networks, where nodes correspond to vessel branch points and edges represent vessel segments annotated with length, width, and curvature
- Train and evaluate Graph Neural Network (GNN) models to analyze vascular graphs, capturing both local structural changes and global network patterns
- Compare GNN performance to baseline models using traditional imaging features
- Identify which vascular features are most predictive of patient-specific treatment response

## Team
- Bella Summe, Julia Luo, José Cardona Arias, Rebecca Wu

## Installation

Micromamba install (one time):
```
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
./bin/micromamba shell init -s bash -r ~/micromamba 
source ~/.bashrc
micromamba config append channels conda-forge
```

Installation recipe (one time):
```bash
micromamba config prepend channels conda-forge
micromamba config set channel_priority strict
git clone --recursive git@github.com:dsi-clinic/vanguard.git
cd vanguard
micromamba env create -y -n vanguard -f environment.yml
micromamba activate vanguard
```

Note: All Python dependencies (including pip-only packages like PyRadiomics) are installed via `environment.yml`; you should not need to run `pip install -r requirements.txt`.

To Update:
```bash
micromamba activate vanguard
micromamba env update -y -n vanguard -f environment.yml
```

(Be sure to clone this repo with `--recursive` so that submodules like [dsi-clinic/vanguard-blood-vessel-segmentation](https://github.com/dsi-clinic/vanguard-blood-vessel-segmentation) are included.)

## Repository Structure and Methodologies

This repository contains a complete pipeline for predicting pathologic complete response (pCR) from breast DCE-MRI using vascular graph analysis. The following sections describe each component and how to reuse them for future cohorts.

---

## 1. Data

### 1.1 MAMA-MIA Dataset
This project utilizes the **MAMA-MIA (Multi-Center Breast Cancer DCE-MRI Benchmark)** dataset, a comprehensive collection of breast cancer imaging and clinical data designed to advance AI research in tumor segmentation and treatment response prediction.

The dataset aggregates data from **1,506 patients** across four major clinical collections (I-SPY1, I-SPY2, NACT-Pilot, and Duke-Breast-Cancer-MRI). It provides a robust foundation for analyzing vascular networks due to its high-quality expert annotations and harmonized clinical variables.

**Key Features:**
* **Imaging Data:** Dynamic Contrast-Enhanced MRI (DCE-MRI) scans for 1,506 patients, including multiple time points to track changes during therapy.
* **Annotations:** Expert-validated 3D tumor segmentations for precise localization of the tumor volume.
* **Clinical Variables:** Includes 49 harmonized variables such as patient age, menopausal status, tumor subtypes, and treatment outcomes (specifically Pathologic Complete Response or pCR).

**Link to Dataset:**
[https://github.com/LidiaGarrucho/MAMA-MIA](https://github.com/LidiaGarrucho/MAMA-MIA)

**Reference:**
> Lidia Garrucho et al. MAMA-MIA: A large-scale multi-center breast cancer DCE-MRI benchmark dataset with expert segmentations. Synapse https://doi.org/10.7303/SYN60868042 (2024).

## 2. Baseline Models

### 2.1 Non-Imaging Baseline (`non_imaging_baseline/`)

**Purpose**: Predict pCR using only demographic and clinical metadata features (no imaging data).

**Methodology**:
- Extracts features from patient JSON metadata files: age, tumor subtype, bounding box volume
- Uses logistic regression with L2 regularization
- Includes feature importance analysis via permutation importance and coefficient analysis

**Key Files**:
- `baseline_pcr_simple.py`: Main training script for non-imaging baseline
- `feature_importance.py`: Analyzes which non-imaging features are most predictive

**Usage**:
```bash
# Train baseline model
python non_imaging_baseline/baseline_pcr_simple.py \
  --json-dir /path/to/patient_info_files \
  --split-csv splits_v1.csv \
  --output outdir

# Analyze feature importance
python non_imaging_baseline/feature_importance.py \
  --model outdir/model.pkl \
  --json-dir /path/to/patient_info_files \
  --split-csv splits_v1.csv \
  --output feature_outdir
```

**Outputs**: `metrics.json`, `predictions.csv`, `roc_test.png`, `model.pkl`, feature importance plots

---

### 2.2 Radiomics Baseline (`radiomics_baseline/`)

**Purpose**: Predict pCR using PyRadiomics features extracted from MRI volumes and tumor masks.

**Methodology**:
- **Stage 1 (Extraction)**: Extracts PyRadiomics features (first-order, shape, texture features) from DCE-MRI volumes and tumor masks
- **Stage 2 (Training)**: Trains classifiers (logistic regression, random forest, XGBoost) on extracted feature tables
- Supports multiple DCE phases and optional peritumoral shell analysis
- Includes feature selection (correlation pruning, SelectKBest) and optional subtype inclusion

**Key Files**:
- `radiomics_extract.py`: Extracts PyRadiomics features to CSV
- `radiomics_train.py`: Trains models on extracted features
- `pyradiomics_params.yaml`: Configuration for PyRadiomics extraction

**Usage**:
```bash
# Extract features
python radiomics_baseline/radiomics_extract.py \
  --images /path/to/images \
  --masks /path/to/segmentations \
  --labels radiomics_baseline/labels.csv \
  --split radiomics_baseline/splits_train_test_ready.csv \
  --output radiomics_baseline/experiments/extract_peri5_multiphase \
  --params radiomics_baseline/pyradiomics_params.yaml \
  --image-pattern "{pid}/{pid}_0001.nii.gz,{pid}/{pid}_0002.nii.gz" \
  --mask-pattern "{pid}.nii.gz" \
  --peri-radius-mm 5

# Train model
python radiomics_baseline/radiomics_train.py \
  --train-features experiments/extract_peri5_multiphase/features_train_final.csv \
  --test-features experiments/extract_peri5_multiphase/features_test_final.csv \
  --labels labels.csv \
  --output outputs/elasticnet_corr0.9_k50_cv5 \
  --classifier logistic \
  --include-subtype
```

**Outputs**: Feature CSVs, `metrics.json`, `predictions.csv`, ROC/PR/calibration plots, `model.pkl`

**See**: [`radiomics_baseline/README.md`](radiomics_baseline/README.md) for detailed documentation

---

## 3. Centerline Extraction Methods

### 3.1 Graph Extraction (`graph_extraction/`)

**Purpose**: Topology-preserving 3D skeletonization algorithm for vessel segmentation volumes using graph pruning methodology.

**Methodology**:
- Iteratively removes voxels from binary vessel masks while preserving 26-connectivity
- Each voxel is treated as a node with 26-connected neighbors
- Produces skeleton volumes and optional graph representations
- Uses graph-based pruning to maintain vessel topology

**Key Features**:
- Preserves vessel topology (no breaking connectivity)
- Can export to JSON or other formats for downstream analysis
- Includes visualization utilities for skeleton inspection

**Usage**: See [`graph_extraction/README.md`](graph_extraction/README.md) for implementation details and API

**When to use**: When you need a topology-preserving skeletonization with graph-based pruning for downstream analysis.

---

### 3.2 Thinning-Based (`thinning_based_centerline_extraction/`)

**Purpose**: Thinning-based centerline extraction with island connection and graph structure building.

**Methodology**:
1. **Binarizes** segmentation at specified threshold
2. **Skeletonizes** binary mask to extract 3D skeleton using thinning algorithm
3. **Connects fragmented islands** using k-nearest neighbor search (optional)
4. **Builds graph structure** from skeleton (nodes = branch points, edges = vessel segments)
5. **Extracts centerlines** as polylines from graph structure
6. **Outputs** centerlines as VTK PolyData (`.vtp`) or JSON format

**Key Features**:
- Thinning-based extraction using 26-connectivity (based on Matlab Skel2Graph3D)
- Island connection to heal fragmented skeletons before graph building
- Supports multiple input formats: `.nii.gz`, `.nrrd`, `.npy` (4D arrays with channel selection)
- Off-screen PyVista visualizations for debugging

**Key Files**:
- `extract_centerlines.py`: Main extraction script
- `run_centerline_extraction.py`: Convenience wrapper that combines extraction and JSON conversion

**Usage**:
```bash
# Extract centerlines only
python thinning_based_centerline_extraction/extract_centerlines.py \
  vessel_segmentation.npy \
  output_centerlines.vtp \
  --no-visualizations \
  --max-connection-distance-mm 15.0

# Extract and convert to JSON in one step
python thinning_based_centerline_extraction/run_centerline_extraction.py \
  vessel_segmentation.npy \
  output_centerlines.json \
  --spacing 1.0 1.0 1.0
```

**When to use**: When you need robust centerline extraction with island connection and graph structure for downstream ML analysis.

**See**: [`thinning_based_centerline_extraction/README.md`](thinning_based_centerline_extraction/README.md) for detailed options and examples

---

## 4. ML Testbed (`ML-Pipeline/`)

**Purpose**: Machine learning pipeline for training and evaluating pCR prediction models on graph-based vascular features.

**Methodology**:
- Loads per-case JSON feature files (extracted from centerlines)
- Engineers features (aggregation, normalization, feature selection)
- Trains models (Random Forest, Logistic Regression)
- Evaluates with cross-validation, bootstrap confidence intervals, and statistical tests
- Supports ensemble runs for stability assessment

**Key Features**:
- Handles missing data and feature engineering
- Includes random baseline comparison and DeLong test for statistical validation
- Ensemble runs to assess model stability across random seeds
- Comprehensive metrics and visualizations

**Key Files**:
- `pcr_prediction.py`: Main training and evaluation script

**Usage**:
```bash
python ML-Pipeline/pcr_prediction.py \
  --feature-dir vessel_segmentations/processed_3D \
  --labels pcr_labels.csv \
  --label-column pcr \
  --id-column patient_id \
  --outdir out_pcr \
  --model rf \
  --plots \
  --test-size 0.2 \
  --val-size 0.2 \
  --bootstrap-n 1000 \
  --delong
```

**Outputs**: 
- `metrics.json`: Performance metrics (AUC, AP, accuracy, precision, recall, F1)
- `predictions.csv`: Per-case predictions
- `feature_importance.csv` and plots: Top predictive features
- ROC/PR curves, confusion matrices
- `model.pkl`: Trained model for inference

**See**: [`ML-Pipeline/README.md`](ML-Pipeline/README.md) for all command-line options

---

## 5. Output Directories

**Note**: Output directories are excluded from git (see `.gitignore`). The following directories contain outputs:

- `graph_pruning_outdir/`: Graph pruning method results (excluded from git)
- `thinning_based_outdir/`: Thinning-based method results (excluded from git)

**Full paths to reproducible outputs** (if needed for reference):
- Graph pruning results: `/net/projects2/vanguard/output/skeleton_to_graph_output/` (if regenerated)
- Centerline JSON outputs: `/net/projects2/vanguard/centerline_json_outputs/` (if regenerated)

**Note**: To reproduce results, run the pipeline with the same configuration. Output directories follow consistent structure for easy comparison across experiments.

---

## 6. Helper Tools

### 6.1 Batch Processing (`batch_processing/`)

**Purpose**: Automated batch processing scripts for large-scale vessel segmentation and centerline extraction.

**Key Scripts**:
- `batch_segmentation.py`: Batch vessel segmentation from `.nii.gz` files
  - Preprocesses images (normalization, axis rotation)
  - Runs 3-step segmentation pipeline (preprocessing → breast mask → vessel segmentation)
  - Supports parallel processing and resume functionality
- `batch_extract_centerlines.py`: Batch centerline extraction from vessel segmentations
- `batch_process_centerlines.py`: Batch processing that extracts centerlines and converts to JSON with proper spacing extraction
- `batch_convert_vtp_to_json.py`: Converts VTP centerline files to JSON format

**Usage**:
```bash
# Batch vessel segmentation
python batch_processing/batch_segmentation.py \
  --images-dir /path/to/images \
  --output-dir /path/to/vessel_segmentations \
  --max-workers 8 \
  --resume

# Batch centerline extraction
python batch_processing/batch_extract_centerlines.py

# Batch process centerlines to JSON
python batch_processing/batch_process_centerlines.py
```

**See**: [`batch_processing/README.md`](batch_processing/README.md) for detailed documentation

---

### 6.2 SLURM Submit Scripts (`slurm_submit_scripts/`)

**Purpose**: Pre-configured SLURM batch job scripts for running pipeline components on HPC clusters.

**Key Scripts**:
- `submit_vessel_segmentation.slurm`: Main vessel segmentation job (1 GPU, 16 CPUs, 128GB RAM)
- `submit_vessel_segmentation_array.slurm`: Array job for parallel processing (one file per task)
- `submit_vessel_segmentation_optimized.slurm`: High-resource version for faster processing
- `submit_centerline_extraction.slurm`: Centerline extraction job
- `submit_vtp_to_json_conversion.slurm`: VTP to JSON conversion job

**Usage**:
```bash
# Submit vessel segmentation
sbatch slurm_submit_scripts/submit_vessel_segmentation.slurm

# Submit array job (processes all files in parallel)
sbatch slurm_submit_scripts/submit_vessel_segmentation_array.slurm

# Monitor jobs
squeue -u $USER
tail -f logs/vessel-seg-<JOB_ID>.out
```

**See**: [`slurm_submit_scripts/README.md`](slurm_submit_scripts/README.md) for all available scripts and monitoring commands

---

### 6.3 Clinical and Imaging Exploration (`clinical_and_imaging_exploration/`)

**Purpose**: Exploratory data analysis (EDA) notebooks for understanding clinical and imaging data distributions.

**Contents**:
- `exploration.ipynb`: Jupyter notebook for data exploration
- `eda_out/`: Generated outputs including:
  - Figures: age distributions, pCR rates by subtype/laterality/menopausal status, missing data patterns
  - Tables: summary statistics, missing data summaries

**Usage**: Open `exploration.ipynb` in Jupyter and run cells to regenerate EDA outputs for new cohorts.

---

### 6.4 Evaluation Framework (Dataset Selection and Splits)

**Purpose**: Centralized evaluation system with k-fold cross-validation and cohort selection for focused A/B experiments.

**Dataset selection**: Restrict evaluation to specific datasets, sites, tumor types, or laterality (unilateral/bilateral). Criteria are combined with AND logic; within a criterion (e.g. datasets), OR applies (IN semantics).

**Usage examples** (require `--excel-metadata` with clinic metadata Excel):

```bash
# Run evaluation only on iSpy2
python examples/baseline_model_example.py --model random \
  --excel-metadata path/to/clinical_and_imaging_info.xlsx \
  --datasets iSpy2 --output results/ispy2

# Run on iSpy2 + Duke
python examples/baseline_model_example.py --model random \
  --excel-metadata path/to/clinical_and_imaging_info.xlsx \
  --datasets iSpy2 Duke --output results/ispy2_duke

# Stacked criteria: iSpy2 AND unilateral cases
python examples/baseline_model_example.py --model random \
  --excel-metadata path/to/clinical_and_imaging_info.xlsx \
  --datasets iSpy2 --unilateral-only --output results/ispy2_unilateral

# Stratified evaluations (use stratify_cols in export_splits)
python -m src.utils.export_splits --excel metadata.xlsx --output splits.csv \
  --stratify-cols dataset tumor_subtype --group-col site

# Selection from YAML config (CLI flags override config)
python examples/baseline_model_example.py --model random \
  --excel-metadata path/to/metadata.xlsx \
  --config config/eval_selection_example.yaml --output results/config_run
```

**Config file** (`config/eval_selection_example.yaml`): Define `selection.datasets`, `selection.sites`, `selection.tumor_types`, `selection.unilateral_only`, `selection.bilateral_only`, or `selection.column_filters` for generic column filters.

---

## Pipeline Workflow

### End-to-End Example

This example shows a complete walkthrough from raw data to model training using relative paths:

**Expected input folder layout:**
```
vanguard/
├── data/
│   ├── images/              # DCE-MRI images (patient subdirectories)
│   │   └── DUKE_001/
│   │       ├── DUKE_001_0001.nii.gz
│   │       └── DUKE_001_0002.nii.gz
│   ├── masks/               # Tumor segmentations
│   │   └── DUKE_001.nii.gz
│   └── metadata/            # Patient JSON files
│       └── DUKE_001.json
├── splits_v1.csv           # Train/test split
└── labels.csv              # pCR labels
```

**Complete pipeline:**

1. **Vessel Segmentation**:
   ```bash
   python batch_processing/batch_segmentation.py \
     --images-dir data/images \
     --output-dir data/vessel_segmentations \
     --max-workers 4 \
     --resume
   ```
   Output: `data/vessel_segmentations/*.npy` (vessel segmentation masks)

2. **Centerline Extraction** (thinning-based method):
   ```bash
   python batch_processing/batch_extract_centerlines.py \
     --input-dir data/vessel_segmentations \
     --output-dir data/centerlines_vtp \
     --method thinning
   ```
   Or use graph pruning method:
   ```bash
   python batch_processing/batch_extract_centerlines.py \
     --input-dir data/vessel_segmentations \
     --output-dir data/centerlines_vtp \
     --method graph_pruning
   ```
   Output: `data/centerlines_vtp/*.vtp` (centerline polylines)

3. **Convert to JSON**:
   ```bash
   python batch_processing/batch_process_centerlines.py \
     --input-dir data/centerlines_vtp \
     --output-dir data/centerlines_json \
     --spacing 1.0 1.0 1.0
   ```
   Output: `data/centerlines_json/*.json` (centerline features)

4. **Train Baselines** (for comparison):
   ```bash
   # Non-imaging baseline
   python non_imaging_baseline/baseline_pcr_simple.py \
     --json-dir data/metadata \
     --split-csv splits_v1.csv \
     --output outputs/non_imaging_baseline
   
   # Radiomics baseline
   python radiomics_baseline/radiomics_extract.py \
     --images data/images \
     --masks data/masks \
     --labels labels.csv \
     --split splits_v1.csv \
     --output outputs/radiomics_features
   
   python radiomics_baseline/radiomics_train.py \
     --train-features outputs/radiomics_features/features_train_final.csv \
     --test-features outputs/radiomics_features/features_test_final.csv \
     --labels labels.csv \
     --output outputs/radiomics_baseline
   ```

5. **Train ML Models** (graph-based):
   ```bash
   python ML-Pipeline/pcr_prediction.py \
     --feature-dir data/centerlines_json \
     --labels labels.csv \
     --label-column pcr \
     --id-column patient_id \
     --outdir outputs/graph_model \
     --model rf \
     --plots \
     --test-size 0.2
   ```
   Output: `outputs/graph_model/metrics_rf.json`, `predictions.csv`, `feature_importance.csv`

6. **Compare Results**: Compare metrics across `outputs/*` directories to assess model performance.

---

### Alternative: Using SLURM

For HPC clusters, use SLURM submit scripts:

1. **Vessel Segmentation**:
   ```bash
   sbatch slurm_submit_scripts/submit_vessel_segmentation.slurm
   ```

2. **Centerline Extraction**:
   ```bash
   sbatch slurm_submit_scripts/submit_centerline_extraction.slurm
   ```

3. **Convert to JSON**:
   ```bash
   sbatch slurm_submit_scripts/submit_vtp_to_json_conversion.slurm
   ```

See [`slurm_submit_scripts/README.md`](slurm_submit_scripts/README.md) for detailed SLURM usage.

---

## Style

We use [`ruff`](https://docs.astral.sh/ruff/) to enforce style standards and grade code quality. This is an automated code checker that looks for specific issues in the code that need to be fixed to make it readable and consistent with common standards. `ruff` is run before each commit via [`pre-commit`](https://pre-commit.com/). If it fails, the commit will be blocked and the user will be shown what needs to be changed.

To check for errors locally, first ensure that `pre-commit` is installed by running `pip install pre-commit` followed by `pre-commit install`. Once installed, check for errors by running:
```
pre-commit run --all-files
```

---

## Additional Resources

- [`resources.md`](resources.md): Curated list of project, domain, and general references
- [`workflow.md`](workflow.md): Detailed workflow documentation
- [`DataPolicy.md`](DataPolicy.md): Data handling and privacy policies

---

## Notes for Next Cohort

- All scripts support `--resume` flags to safely restart interrupted jobs
- Use SLURM array jobs (`submit_vessel_segmentation_array.slurm`) for maximum parallelization
- Check individual README files in each folder for detailed usage and options
- Compare baseline models (non-imaging, radiomics) against graph-based methods to assess improvement
- Use `clinical_and_imaging_exploration/` to understand data distributions before modeling
- All output directories follow consistent structure for easy comparison across experiments
