# UChicago Site-Specific Data

**Scope: this file describes one site's data only.** Every path, cohort count, and access
rule below is valid only on the University of Chicago cluster, under the Karczmar-lab
shares. Nothing here is bundled with the repository, and none of it is reachable from
outside that cluster.

If you are running Vanguard elsewhere, treat this file as a worked example of the data
contract the code expects, not as something to point your configs at. You need either:

- **your own dataset**, prepared to the same contract — a
  `dce2d_internal_ultrafast_manifest.csv` with the columns described below, and the
  `images/`, `centerlines/`, `tumor/` layout under your own root; or
- **access to the UChicago cohorts**, which requires an agreement with the Karczmar lab.
  The raw DICOM sources are not deidentified and are not shareable.

Override the relevant `data_paths` values in your own YAML under [`configs/`](configs/)
rather than editing code or assuming these roots exist. See
[`docs/data_policy.md`](docs/data_policy.md) for the governing data policy.

This file documents cohort composition, layout, and known caveats. It is deliberately not
a complete provenance record.

## UChicago ultrafast cohorts

Two published dataset roots under `/gpfs/data/karczmar-lab/vanguard/`, complete through
cohort construction, Vanguard v5 preprocessing, and tumor/vessel artifact publication:

- `uchicago_ultrafast_longitudinal_cohort_v1` — 240 exams from 196 patients. Repeated
  visits are retained on purpose: 179 runnable legacy exams plus 61 newly transferred
  exams from 55 patients. Patient-level pCR is 122 non-pCR / 74 pCR.
- `uchicago_ultrafast_pretreatment_cohort_v1` — 137 exams from 137 patients, one
  reviewed pretreatment exam per patient: 82 runnable legacy baselines plus 55
  new-patient baselines. pCR is 78 non-pCR / 59 pCR.

The cohort manifest is `dce2d_internal_ultrafast_manifest.csv` in either root. It keeps
the existing UChicago manifest contract: `exam_id`, `patient_key`, fixed patient-grouped
`fold`, `pcr`, physical timestamps, phase paths, and clinical/cohort metadata. Repeated
exams from the same patient always share a label and a fold.

Each root contains:

- `images/<dataset>/<exam_id>/` — motion-corrected UFAST phase NIfTIs and
  `ufast_times_seconds.npy`. The UFAST signal is raw: no clipping, no z-scoring, five
  protocol baselines, physical DICOM times, and one spatial interpolation. Dynamics are
  not collapsed and not independently normalized by phase.
- `centerlines/<dataset>/<exam_id>/` — the HR-derived vessel skeleton mask, support
  mask, `*_morphometry.json`, `run_summary.json`, and mapping QC. Vessel extraction uses
  the higher-spatial-resolution acquisition; node kinetics come from the aligned UFAST
  acquisition.
- `tumor/masks/<exam_id>.nii.gz` — primary tumor masks mapped from the
  higher-spatial-resolution segmentation onto the exact UFAST/centerline grid, plus
  `tumor_mask_manifest.csv`, checksums, provenance, and review flags.
- `README.md`, `SHA256SUMS`, and `pending_cases.csv`.

Tumor-mask caveats:

- All 240/240 longitudinal and 137/137 pretreatment masks were published with passing
  alignment/mapping status.
- 14 longitudinal masks are empty. All fall outside the selected pretreatment cohort,
  and most are later visits where a visible tumor may no longer be present.
- The manifests flag 69 longitudinal and 36 pretreatment exams as possible bilateral
  cases requiring review and downstream exclusion. These are conservative candidates,
  not 69 or 36 confirmed bilateral cancers.

`pending_cases.csv` lists four cases explicitly rather than dropping them silently: two
workbook patients whose images were not present in the transferred inventories, and two
legacy exams that cannot be run through the exact HR+UFAST contract (one shared HR/UFAST
acquisition with no distinct HR series, and one split-series exam with eight HR
candidates but no identifiable UFAST series).

## UChicago raw DICOM sources

The cohort roots above expose derived outputs by symlink. Full preprocessing from raw
DICOM uses these restricted Karczmar-lab sources:

- Legacy paired HR+UFAST package:
  `/gpfs/data/karczmar-lab/vanguard/dce2d_internal_ultrafast_manifest/paired_hr_ufast_source_dicom`.
  `archives/<dataset>/<exam_id>.zip` contains byte-for-byte selected DICOM payloads.
  `dicom_file_manifest.parquet`, `dicom_spatial_geometry_manifest.csv`,
  `hr_ufast_spatial_alignment_manifest.csv`, and
  `paired_preprocessing_case_manifest.csv` are the runnable inventory/geometry contract.
  The package README explains how to launch Vanguard's complete DICOM-to-vessel pipeline.
- Newly transferred DICOM sources: `/gpfs/data/karczmar-lab/Retro NACT`,
  `/gpfs/data/karczmar-lab/9127/9127 NAC`, and `/gpfs/data/karczmar-lab/9127/9127 Staging`.
- Reviewed new-data inventories and selected-series manifests, under
  `uchicago_ultrafast_longitudinal_cohort_v1/_build/source_inventory`,
  `_build/zhen_extension`, and `_build/zhen_staging_extension`.

The DICOM payloads are restricted and are not deidentified. Keep them within the
approved Karczmar-lab shares. The published derived cohorts and their provenance do not
require Huo-lab access.
