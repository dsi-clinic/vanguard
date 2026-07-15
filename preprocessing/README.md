# Paired HR/UFAST preprocessing

This package owns the complete preprocessing path used for the UChicago
vessel-skeleton workflow. It does not import or execute HFDP preprocessing
code. Its two inputs are raw DICOM archives and a read-only DICOM file
inventory (CSV or Parquet).

## Scientific contract

- The case manifest supplies exact study, native high-resolution (HR), and
  ultrafast (UFAST) series UIDs. Vanguard never guesses the series or changes
  the cohort.
- Every temporal position is loaded in DICOM acquisition order. Physical
  timestamps are retained and must be strictly increasing.
- UFAST is resampled once onto a source-aligned 1-mm grid with linear
  interpolation. It is never clipped, normalized, or z-scored.
- Translation-only motion proposals align each UFAST phase to phase 0. A
  proposal is saved only when it is physically bounded and improves
  correlation; otherwise the raw phase and identity transform are retained.
- The frozen breast/vessel models run on every native-grid HR phase. Only this
  model adapter uses the original 0.1% tail clipping and per-volume z-score.
- TC4D receives every HR vessel-probability phase. Its static skeleton and
  support are mapped to the 1-mm UFAST grid through the shared DICOM
  `FrameOfReferenceUID`; all UFAST phases and timestamps remain available for
  downstream kinetics.

Each output case has one `preprocessing_provenance.json` containing source
series identifiers and hashes, physical geometries and timestamps, one shared
4D display window per series, model checksums, motion decisions, TC4D
diagnostics, and HR-to-UFAST mapping metrics.
The physical acquisition times are also saved directly as
`hr_times_seconds.npy` and `ufast_times_seconds.npy`; neither time axis is
replaced by phase indices.

## Case manifest

Copy `case_manifest.example.csv` and add one reviewed row per exam. The UFAST
baseline count is explicit because it is a cohort property, not something the
pipeline should infer.

`configs/uchicago_preprocessing_cases.csv` contains the reviewed SimBioSys
case used for the HR/UFAST validation. Add cases only after reviewing their
exact series UIDs; do not replace this with automatic series selection.

```text
exam_id,dataset,study_instance_uid,hr_series_instance_uid,ufast_series_instance_uid,ufast_baseline_frame_count
```

## Staged run

Heavy stages belong on Slurm. `prepare` is CPU/memory heavy, `infer` needs a
GPU, and `tc4d`/`map` are CPU stages. Stages refuse to overwrite an existing
result.

```bash
micromamba activate vanguard

python -m preprocessing.pipeline prepare \
  --inventory /path/to/dicom_file_inventory.parquet \
  --case-manifest /path/to/vanguard_cases.csv \
  --exam-id uchicago_example \
  --output-root /path/to/derived/vanguard_preprocessing

python -m preprocessing.pipeline infer \
  --exam-id uchicago_example \
  --output-root /path/to/derived/vanguard_preprocessing

python -m preprocessing.pipeline tc4d \
  --exam-id uchicago_example \
  --output-root /path/to/derived/vanguard_preprocessing

python -m preprocessing.pipeline map \
  --exam-id uchicago_example \
  --output-root /path/to/derived/vanguard_preprocessing
```

The mapped skeleton is named with the standard Vanguard centerline pattern in
`<output-root>/centerlines/<dataset>/<exam-id>/`; the matching motion-corrected
raw UFAST phases and physical-time sidecar are in
`<output-root>/dce/<exam-id>/`. Model intermediates and the main provenance are
kept under `<output-root>/work/<exam-id>/`.
Each centerline directory also contains `mapping_qc.png`, which overlays the
mapped skeleton using the one shared intensity window recorded for the full
UFAST 4D series.

For the GNN loader, use `<output-root>/centerlines` as `centerline_root` and
`<output-root>/dce` as `dce_root`. It reads `ufast_times_seconds.npy`, so slopes
and AUC use physical seconds rather than filename indices.

Raw archives and inventories are immutable inputs and must not be deleted or
modified after producing a derivative.
