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
  interpolation. Accepted motion translations are composed into that operation,
  so saved phases aren't interpolated twice. UFAST is never clipped, normalized,
  or z-scored.
- Translation-only motion proposals align each UFAST phase to phase 0. A
  proposal is saved only when it is physically bounded and improves
  correlation; otherwise the raw phase and identity transform are retained.
- The frozen breast/vessel models run on every native-grid HR phase. Only this
  model adapter uses the original 0.1% tail clipping and per-volume z-score.
- TC4D receives every HR vessel-probability phase. Its static skeleton and
  support are mapped to the 1-mm UFAST grid through the shared DICOM
  `FrameOfReferenceUID`; all UFAST phases and timestamps remain available for
  downstream kinetics.
- Because a shared frame doesn't prove that anatomy stayed still between the HR
  and UFAST acquisitions, Vanguard compares HR phase 0 with the mean protocol
  UFAST baseline and flags cases where a meaningful translation would improve
  agreement. Flagged cases can't be used by the GNN until they're reviewed.

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
`<output-root>/dce` as `dce_root`. It uses the mean of the five protocol baseline
frames, computes relative signal change, and reads `ufast_times_seconds.npy`, so
peak time, arrival time, slopes, and AUC all use physical seconds rather than
filename indices.

Raw archives and inventories are immutable inputs and must not be deleted or
modified after producing a derivative.

## Shared UChicago source data

The self-contained reviewed HR/UFAST DICOM source package is staged at:

```text
/gpfs/data/karczmar-lab/vanguard/dce2d_internal_ultrafast_manifest/paired_hr_ufast_source_dicom
```

Use `dicom_file_manifest.parquet` for ZIP-backed loading,
`paired_preprocessing_case_manifest.csv` for exact runnable series UIDs, and
`dicom_spatial_geometry_manifest.csv` for explicit DICOM LPS origin, direction,
spacing, shape, and frame-of-reference. The adjacent
`dce2d_internal_ultrafast_with_paired_source_manifest.csv` links these inputs to
the original cohort without changing the original UFAST manifest. Shared
manifests have no patient columns or source member names, but the byte-preserved
DICOM payloads are not deidentified and must stay in the restricted lab share.

The selection is explicit and reviewed: 179 exams have one complete native HR
series and are eligible to run. One HITS exam has a single shared HR/UFAST
acquisition and is excluded because it has no distinct high-resolution series.
One Siemens exam has seven DCE phases stored as separate series plus a static HR
series; all eight series are retained, but the exam is marked
`split_series_not_runnable` rather than misrepresenting its legacy 84 exported
images as physical UFAST phases.

`preprocessing/stage_high_resolution_dicom.py` and the three
`slurm/*_high_resolution_dicom.slurm` wrappers reproduce the restricted copy,
manifest reduction, and SHA-256 verification. The source selection manifest
and per-exam checksums live beside the staged data.

### Cohort preprocessing from shared DICOM

The cohort runner uses only the paired Karczmar-lab package and Vanguard code.
It does not read Huo-lab inventories or old NIfTI affines. The CPU prepare stage
loads every original HR/UFAST temporal position, preserves physical acquisition
times, writes true RAS NIfTI qform/sform affines derived from DICOM LPS, and
motion-corrects raw-signal UFAST data. GPU inference runs the frozen models on
every native-HR phase. CPU postprocessing runs TC4D, maps the static skeleton
through the shared DICOM frame, and writes QC.

```bash
export REPO_ROOT=/path/to/vanguard
export PAIRED_ROOT=/gpfs/data/karczmar-lab/vanguard/dce2d_internal_ultrafast_manifest/paired_hr_ufast_source_dicom
export PAIRED_INVENTORY=${PAIRED_ROOT}/dicom_file_manifest.parquet
export CASE_MANIFEST=${PAIRED_ROOT}/paired_preprocessing_case_manifest.csv
export OUTPUT_ROOT=/path/to/derived/vanguard_paired_preprocessing

bash slurm/submit_paired_preprocessing.sh
```

The submission prints three job IDs. Corresponding array tasks are linked with
`aftercorr`, so one failed case does not block unrelated cases. Completed stages
are skipped on resubmission. The documented Vanguard environment can be
overridden explicitly with `VANGUARD_PYTHON` when validating another checkout.
