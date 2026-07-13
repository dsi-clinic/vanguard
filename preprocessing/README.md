# SPGR-safe DCE preprocessing

This package loads the internal 181-exam DCE cohort without dropping time
points and provides the motion-correction step used before extracting dynamic
image features.

## Input contract

The source manifest is:

```text
/gpfs/data/karczmar-lab/vanguard/dce2d_internal_ultrafast_manifest/
dce2d_internal_ultrafast_manifest_spgr_safe.csv
```

Each row points to individually gzipped NIfTI frames inside a TAR archive and
to an exact-grid whole-breast label map. The signal was spatially resampled to
1 mm with linear interpolation, but it was not clipped, z-scored, temporally
resampled, or motion corrected. Every frame has an SHA-256 checksum.

The one exam with an implausible recorded time interval remains usable for
image registration and ordinary prediction experiments. Its timing warning
matters only to analyses that interpret elapsed seconds physically; motion
correction itself does not use frame intervals.

## Loading an exam

```python
from preprocessing.spgr import (
    baseline_relative_enhancement,
    load_exam,
    read_manifest,
)

record = read_manifest(manifest_path)[0]
exam = load_exam(record)

relative = baseline_relative_enhancement(
    exam.raw_signal,
    baseline_frame_count=record.baseline_frame_count,
    support_mask=exam.whole_breast_mask,
)
```

`exam.raw_signal` has shape `(time, x, y, z)`. All original frames and times are
returned. `relative.values` is unclipped `(S(t) - S0) / S0`; excluded/background
voxels are identified by `relative.valid_support` rather than silently filled
with a made-up denominator.

## Motion correction

Motion correction aligns every phase to phase 0 with a translation estimated
by downsampled phase correlation. Estimation uses the reviewed whole-breast
support. Temporary robust normalization is used only to estimate a translation;
the saved images remain nonnegative signal and are resampled once with linear
interpolation.

Every phase records both the proposed transform and the transform actually
saved. A proposal is accepted only when it does not worsen correlation to phase
0 inside the same breast support. Otherwise the saved transform is identity and
the original phase is retained. No frame is dropped and timestamps are copied
unchanged.

The cohort runner uses one restartable Slurm array task per exam and then a
semantic merge. The merge runs after any scheduler outcome but refuses to
publish unless all expected shard archives and metadata are complete.

```bash
micromamba activate vanguard

export MANIFEST=/gpfs/data/karczmar-lab/vanguard/dce2d_internal_ultrafast_manifest/dce2d_internal_ultrafast_manifest_spgr_safe.csv
export OUTPUT_ROOT=/gpfs/data/karczmar-lab/vanguard/dce2d_internal_ultrafast_manifest/spgr_safe_motion_corrected

bash preprocessing/slurm/submit_motion_correction.sh
```

Optional runtime controls are `CONCURRENCY` and `MAXIMUM_QC_PANELS`. The
registration settings are intentionally one reviewed code contract rather than
a collection of student-facing knobs.

The merged output contains:

- `manifest.csv`: same exams, labels, folds, frame count, and timestamps, with
  `motion_correction_applied=true`;
- `phase_images.tar`: motion-corrected frames and member checksums;
- `registration_metrics.csv`: proposed/saved shifts and correlations for every
  non-reference phase;
- `motion_qc_contact_sheet.png`: highest-motion cases under one shared intensity
  window per panel;
- `motion_qc_selection.csv`: exact cases and phases shown in the contact sheet;
- `summary.json`: counts, checksums, and registration bounds; and
- `shards/`: restartable per-exam archives and provenance.

Do not delete the source archive after creating the motion-corrected derivative.
It is the immutable reference needed to reproduce or audit registration.
