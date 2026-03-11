# Graph Extraction

Skeleton extraction and skeleton-to-graph feature code for 3D and 4D vessel segmentations.

**Paths:** Examples below use relative paths (e.g. `output/`, `report/4d_morphometry`) where possible. Some docs and cluster scripts use project-specific absolute paths (e.g. `/net/projects2/vanguard/...`); replace those with your own paths when running locally.

## Layout

- `run_skeleton_processing.py`: main processing pipeline (3D or 4D), with skeleton extraction + morphometry generation.
- `run_compare_3d_4d_debug.py`: debugging/visualization script for side-by-side 3D vs 4D comparison.
- `batch_process_4d.py`: batch 4D morphometry extraction for weak-signal diagnostics.
- `build_features_with_metadata.py`: extract features from morphometry JSONs and join with patient_info.
- `run_feature_qc.py`: per-feature QC, sanity checks, and distribution plots.
- `run_ablation_study.py`: ablation study (Phase 4).
- `run_fp_fn_inspection.py`: FP/FN case inspection and skeleton viz (Phase 5).
- `feature_sanity.py`: validation helpers for morphometry values (tortuosity, angles, radii, etc.).
- `processing.py`: shared 3D/4D loading, extraction, collapse, and morphometry helpers.
- `skeleton3d.py`, `skeleton4d.py`, `visuals.py`, `skeleton_to_graph.py`: core extraction and graph/morphometry modules.

## Main Pipeline

Use one script for production processing.

### 3D mode (single volume)

```bash
python graph_extraction/run_skeleton_processing.py 3d \
  --input-file /path/to/ISPY2_202539_ISPY2_202539_0000_vessel_segmentation.npy \
  --output-dir output \
  --threshold-low 0.5 \
  --npy-channel 1
```

### 4D mode (exam-level across all timepoints)

```bash
# Use relative paths when running from repo root (e.g. input-dir vessel_segmentations)
python graph_extraction/run_skeleton_processing.py 4d \
  --input-dir vessel_segmentations \
  --study-id ISPY2_202539 \
  --output-dir output \
  --npy-channel 1 \
  --threshold-low 0.5 \
  --threshold-high 0.85
```

4D input layout (when using `--input-dir` and `--study-id`):
- Traverses `{input_dir}/{SITE}/{STUDY_ID}/images/`, where SITE is the first underscore-separated component of study_id (e.g. ISPY2_202539 → SITE=ISPY2).
- Expects `.npz` files matching `*{study_id}_{0000}_vessel_segmentation.npz` (same naming pattern as before, with .npz extension).

Behavior:
- If skeleton outputs already exist, they are reused unless `--force-skeleton` is set.
- If morphometry JSON already exists, it is reused unless `--force-features` is set.

## Debug Comparison Script

Use this for 3D-vs-4D QC plots and rotating MP4:

```bash
python graph_extraction/run_compare_3d_4d_debug.py \
  --input-dir vessel_segmentations \
  --study-id ISPY2_202539 \
  --npy-channel 1 \
  --threshold-low 0.5 \
  --threshold-high 0.85 \
  --output-dir debug_output
```

Notes:
- Tumor overlay paths are cluster-specific; on cluster, overlay may be auto-resolved from `.../segmentations/expert/{study_id}.nii.gz`.
- MP4 frame count is reduced for faster render.

## Weak-signal diagnostic pipeline (Phase 1 & 2)

For diagnosing weak downstream prediction signal:

**Phase 1.1 – batch 4D morphometry:**
```bash
python graph_extraction/batch_process_4d.py \
  --input-dir vessel_segmentations \
  --npy-channel 1 \
  --threshold-low 0.5 \
  --threshold-high 0.85 \
  --output-dir report/4d_morphometry
```

Use `--skip-existing` to skip studies that already have morphometry JSON, and `--quiet` to suppress verbose output.

**Phase 1.2 – feature extraction + metadata join:**
```bash
python graph_extraction/build_features_with_metadata.py \
  --morphometry-dir report/4d_morphometry \
  --patient-info-dir path/to/patient_info_files \
  --output-dir report
```

**Phase 2 – per-feature QC, sanity checks, distribution plots:**
```bash
python graph_extraction/run_feature_qc.py \
  --features-csv report/features_with_metadata.csv \
  --morphometry-dir report/4d_morphometry \
  --output-dir report
```

Outputs: `qc_per_feature.csv`, `sanity_violations.csv`, `report/plots/distributions_core.png`.

**Phase 3 – batch effect and nuisance analysis:**
```bash
python graph_extraction/run_batch_effect_analysis.py \
  --features-csv report/features_with_metadata.csv \
  --output-dir report
```

Use `--no-umap` to skip UMAP (faster; UMAP requires `pip install umap-learn`). Use `--qc-csv report/qc_per_feature.csv` to drop high-missing features before reduction.

Outputs:
- `report/plots/pca_colored_by_*.png`, `umap_colored_by_*.png`, `tsne_colored_by_*.png` (label, site, dataset, manufacturer)
- `report/embedding_coords.csv` (2D coordinates for report notebook)
- `report/site_prediction_metrics.json` (batch-effect indicator: AUC macro)
- `report/plots/site_prediction_confusion.png`
- `report/site_top_features.csv`

**Phase 4 – ablation study:**
```bash
python graph_extraction/run_ablation_study.py \
  --features-csv report/features_with_metadata.csv \
  --output-dir report
```

Uses fixed 70/15/15 train/val/test split. Ablations: full, no_coordinate_like, geometry_only, count_only.

Outputs: `ablation_results.csv`, `report/plots/ablation_auc_comparison.png`.

**Phase 5 – FP/FN case inspection:**
```bash
python graph_extraction/run_fp_fn_inspection.py \
  --features-csv report/features_with_metadata.csv \
  --morphometry-dir report/4d_morphometry \
  --output-dir report
```

Requires morphometry dir with `{study_id}_skeleton_4d_exam_mask.npy` (from batch_process_4d) for skeleton viz.

Outputs: `fp_fn_cases.csv`, `fp_fn_outliers.csv`, `fp_fn_viz_status.csv`, `report/viz/fp_*.png`, `report/viz/fn_*.png`.

**Phase 6 – report notebook:**
Run `report/weak_signal_diagnostic_report.ipynb` to load and display all Phase 2–5 outputs and fill in the conclusion template.
