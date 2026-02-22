# Graph Extraction

Skeleton extraction and skeleton-to-graph feature code for 3D and 4D vessel segmentations.

## Layout

- `run_skeleton_processing.py`: main processing pipeline (3D or 4D), with skeleton extraction + morphometry generation.
- `run_compare_3d_4d_debug.py`: debugging/visualization script for side-by-side 3D vs 4D comparison.
- `batch_process_4d.py`: batch 4D morphometry extraction for weak-signal diagnostics.
- `build_features_with_metadata.py`: extract features from morphometry JSONs and join with patient_info.
- `run_feature_qc.py`: per-feature QC, sanity checks, and distribution plots.
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
python graph_extraction/run_skeleton_processing.py 4d \
  --input-dir /net/projects2/vanguard/vessel_segmentations \
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
  --input-dir /net/projects2/vanguard/vessel_segmentations \
  --study-id ISPY2_202539 \
  --npy-channel 1 \
  --threshold-low 0.5 \
  --threshold-high 0.85 \
  --output-dir debug_output
```

Notes:
- Tumor overlay is auto-resolved in study mode from:
  `/net/projects2/vanguard/MAMA-MIA-syn60868042/segmentations/expert/{study_id}.nii.gz`
- MP4 frame count is reduced for faster render.

## Weak-signal diagnostic pipeline (Phase 1 & 2)

For diagnosing weak downstream prediction signal:

**Phase 1.1 – batch 4D morphometry:**
```bash
python graph_extraction/batch_process_4d.py \
  --input-dir /net/projects2/vanguard/vessel_segmentations \
  --npy-channel 1 \
  --threshold-low 0.5 \
  --threshold-high 0.85 \
  --output-dir report/4d_morphometry
```

Use `--skip-existing` to skip studies that already have morphometry JSON, and `--quiet` to suppress verbose output.

**Phase 1.2 – feature extraction + metadata join:**
```bash
python graph_extraction/build_features_with_metadata.py \
  --morphometry-dir /net/projects2/vanguard/report/4d_morphometry \
  --patient-info-dir /net/projects2/vanguard/MAMA-MIA-syn60868042/patient_info_files \
  --output-dir report
```

**Phase 2 – per-feature QC, sanity checks, distribution plots:**
```bash
python graph_extraction/run_feature_qc.py \
  --features-csv report/features_with_metadata.csv \
  --morphometry-dir /net/projects2/vanguard/report/4d_morphometry \
  --output-dir report
```

Outputs: `qc_per_feature.csv`, `sanity_violations.csv`, `report/plots/distributions_core.png`.
