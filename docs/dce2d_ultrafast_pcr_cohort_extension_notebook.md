# Extending Sarit's pCR cohort onto newly landed retro-CAPS exams

Working notebook for the v5 -> v6 extension. Records inputs, job IDs, and decisions so the
release is reproducible and so a fresh session can resume without re-deriving the picture.

## Outcome, 2026-08-28

Done. `dce2d_internal_ultrafast_pcr_cohort_v6` and `_v6_raw_dicoms` are published under
`/gpfs/data/karczmar-lab/vanguard`: 283 main exams over 283 patients (172 pCR-, 111 pCR+), 488
source exams, 198 longitudinal exams, folds 78/77/77/77/77, and raw-DICOM locators for 566
series over 879167 files. All 243 v5 exams carry forward with every label column reproduced
exactly.

Two sections below are superseded by how it was actually built and should not be followed:

- The plan to run the builder into a scratch root and take only the rows absent from v5 is
  gone. `--previous-fold-assignments` freezes released folds directly, which was the only
  thing the four-stage chain bought, so v6 is one build from stated inputs. The frozen human
  decisions live in
  `/gpfs/data/karczmar-lab/vanguard/dce2d_ultrafast_pcr_cohort_curation_v1`.
- The builder is no longer untracked. It is `scripts/build_dce2d_ultrafast_pcr_cohort.py`,
  tracked in vanguard PR #210, and the sha256 quoted below belongs to the v2-era
  `build_sarit_pcr_cohort.py`.

Two v5 defects were found and fixed on the way, both recorded in
`dce2d_internal_ultrafast_pcr_cohort_v6_raw_dicoms/ERRATUM.md` and in the retired
`configs/pcr/sarit_pcr_raw_dicom_release.yaml` in hfdp: v5's consumer manifest points its
image paths into v4, and v5 shipped a `MIP SUB UFAST` projection as one exam's ultrafast
acquisition. Do not remove v4 while anyone is pinned to v5.

## Why

More retro-CAPS imaging landed after v5 was frozen. v5 covers 243 exams, 125 of them
retro-CAPS, and its raw-DICOM locators were resolved against a DICOM inventory of only 632
delivered exams. The delivery now holds 867 exam directories over 600 patients, so the
ultrafast/high-resolution (UFAST/HR) pair determination is stale and the cohort is smaller
than the delivered data now supports.

## How UFAST + HR is determined

There is no clinical field for it; both signals are read from delivered DICOM.

1. `scripts/scan_landed_series_ultrafast.py` (hfdp) matches `SeriesDescription` against
   `UFAST`, which is how the Philips protocols in this cohort name those series. Output:
   `derived_datasets/retro_caps_pcr_cohort/landed_series_inventory_v1/landed_series_ultrafast.csv`.
   It is incremental and rescans only exams it has not seen. `ufast_frame_interval_s` is
   parsed from the description (`dyn_eTHRIVE UFAST 3s`), so it is what the protocol claims,
   not a measurement of delivered frame times.
2. `scripts/audit_retro_caps_two_series_gate_readiness.py` (hfdp) decides whether an HR
   dynamic partner exists and is usable, through three typed gates: axis-aligned geometry,
   one repetition time and one flip angle per series (the SPGR inversion precondition), and
   an HR series that straddles the UFAST injection reference on the study's shared clock.

Standing caveat from the producers: landed exams only. A blank `has_ultrafast` means the exam
has not been scanned, not that ultrafast is absent.

## State at the start of this work

- Delivery root: `/gpfs/data/karczmar-lab/DR_662135_Karczmar_Breast_MRI_ML/retro_caps_pcr_imaging_cohort`
- 600 patient directories, 867 raw exam directories, 866 with durable transfer receipts.
- Transfer is still live: batch 31 receipts written 2026-08-28 07:48, 29 exams.
- Priority bucket `pretreatment_post2015` is 483 of 515 exams delivered (94%); roughly one
  more day of transfer. Later buckets (on-treatment, screening, tumour-bearing-no-NAT) are
  much larger and are not needed for this cohort.
- `exam_metadata.csv` in the delivery root, rebuilt 2026-08-27 10:35, covers 837 exams:
  `has_ultrafast` true 381 / false 456.
- Last gate-readiness run (2026-08-25) used the stale 632-exam inventory
  `retro_caps_pcr_imaging_inventory_stability_gated`: 263 exams carried both series, 253
  passed all gates, layouts 245 `single_series` + 8 `split_precontrast_postcontrast_pair`.

Delta measured against v5 on that stale gate table, before any refresh:

| filter | exams | patients |
| --- | --- | --- |
| gate-passing UFAST+HR not already in v5 | 128 | 123 |
| + usable binary pCR label | 91 | 87 |
| + single primary (v5's own rule) | 79 | 75 |

Of the 128: 22 have no pCR label yet, 15 are `not_evaluable_or_conflict`, 19 are
multiple-primary patients whose patient-level label is not attributable to one cancer.

## The 43 already-adjudicated held-out exams

v5's `source_eligible_cohort_manifest.csv` carries 286 image-deduplicated labeled exams;
its consumer manifest carries 243. The 43-exam difference is listed in
`pending_unprocessed_pretreatment_sources.csv` and is the cleanest extension available:

- all `retro_caps_pcr`, all tumour-bearing, pCR 0/1 = 33/10
- UFAST and HR `SeriesInstanceUID` already resolved for all 43
- folds already assigned, labels already adjudicated (39 `oncotrace_glm52`, 4 `regex_proxy`)
- layouts 38 `single_series` + 5 `split_precontrast_postcontrast_pair`
- held out only because the raw-signal UFAST phase export was never published (40 `pending`,
  3 `ready`), and for the split pairs because cross-series HR intensity scaling is unresolved

This matters for packaging: `scripts/build_sarit_pcr_cohort.py` deliberately admits an exam
to the consumer manifest only when it has a published phase export with its native clock.
Sarit's own path is raw DICOM -> `preprocessing.pipeline` -> segmentation -> skeletons, so
raw-DICOM locators plus exact series UIDs are what he needs to generate those phases himself.

## The phase export is no longer the blocker

`derived_datasets/hfdp/retro_caps_pcr_raw_signal_cache/raw_signal_manifest.csv` was rebuilt
2026-08-26 04:51, after v5 pinned it, and now holds 395 exams (sha256
`72916a86071e1bcb16a5b166fb0956b3e23c77173f5e1988bb275cb7e322e97f`, versus the pinned
`edac2c5fcd29b5dfc03dd1e1e191ee4c15e1850f61b38e92a0a705b9e98ec735`). Phase exports live under
`<delivery>/derived/<patient>/<visit>/<series>/motion_corrected/phase_NNNN.nii.gz` with
`times_seconds.csv`, every acquired phase kept on its native clock, plus a breast mask; 375
patients have them, 278 GB total.

Coverage against that cache:

- 41 of the 43 held-out exams now have a published phase export
- 125 of the 128 gate-passing exams absent from v5 now have one

So added exams can carry populated `phase_files` in exactly v5's format. No schema
compromise, no empty-`phase_files` rows, and Sarit's existing code needs no change. This
supersedes the packaging question raised below.

## Fold policy: the builder must not simply be re-run

`_assign_folds` in `scripts/build_sarit_pcr_cohort.py` is additive only with respect to the
137-exam UChicago canonical cohort. It seeds its running `balance` from the canonical
assignments alone and then assigns *every* retro patient in `sarit-pcr-fold-v2` hash order,
each to the fold then holding the fewest patients of its pCR class. New retro patients
interleave into that hash order, so the running counts at each step change and previously
assigned retro patients can land in different folds.

Re-running the builder over a larger retro set would therefore silently change train/test
splits for exams Sarit has already used, and would also discard the v3 image-deduplication
revision and the two v4 manual non-pCR adjudications, which are layered on top of v2 rather
than derived by the builder.

The extension must instead be strictly additive on top of v5:

- carry v5's 243 rows unchanged, preserving the v3 and v4 decisions
- seed the fold `balance` from all 243 existing patient assignments, not from the canonical
  137, so existing folds are fixed points
- assign only the new patients, same rule, recorded under a distinct `assignment_source` so
  the addition is auditable

The builder is still useful for constructing the new rows' metadata: run it into a scratch
output root and take only the rows absent from v5, discarding its `fold` column.

Note the builder is untracked in git (`?? scripts/build_sarit_pcr_cohort.py`); releases pin
it by sha256 snapshot instead. The working copy is byte-identical to the one that built v2,
`77314ce2309879de812d0cc2420d1b32d8a38be09f51e5faa78e525d3511ed9e`.

## Runs

- v5 base integrity: `sha256sum -c SHA256SUMS` over
  `/gpfs/data/karczmar-lab/vanguard/dce2d_internal_ultrafast_pretreatment_cohort_v5`,
  299 of 299 entries OK.
- DICOM inventory refresh over all 867 landed exams, submitted from clean pinned worktree
  `/ess/scratch/scratch1/annawoodard/hfdp-worktrees/6caebb075e16cb7f70af277a11a60a4b33cd18c2`
  (commit `6caebb075e16cb7f70af277a11a60a4b33cd18c2`, 0 dirty tracked paths; the shared hfdp
  checkout has 15 unrelated modified files, so it must not be the submit root):

  ```bash
  HFDP_CLUSTER=randi INVENTORY_SHARD_COUNT=32 INVENTORY_WORKERS=8 HFDP_REPO_ROOT="$PWD" \
    bash scripts/slurm_build_delivered_dicom_inventory.sh submit \
    /gpfs/data/karczmar-lab/DR_662135_Karczmar_Breast_MRI_ML/retro_caps_pcr_imaging_cohort \
    /gpfs/data/huo-lab/Image/annawoodard/hfdp/outputs/retro_caps_pcr_imaging_inventory_all_delivered_exams_v2
  ```

  Roster frozen at 867 exams, 0 deferred at 1800s stability. Array job `14499240` (32 shards),
  reducer `14499241` queued `afterany`.
- Incremental UFAST rescan into the canonical
  `derived_datasets/retro_caps_pcr_cohort/landed_series_inventory_v1`, run from the same
  pinned worktree. 867 real exam directories, 396 with UFAST, intervals 274 at 3 s, 2 at
  2.5 s, 1 at 2 s.
- Shard 15 of array `14499240` died 34 s in with `cannot resolve a single walltime maximum
  for tier1q: ''`, a transient `sinfo` miss. Resubmitted alone as `14499576`; the stale
  `afterany` reducer `14499241` was cancelled and rechained as `14499577` on
  `afterany:14499240:14499576`. Per-exam artifacts made the shard resumable, so nothing else
  was recomputed. Do not re-run the wrapper's `submit` action to repair one shard: its
  roster stage calls `--overwrite` and would re-freeze the denominator against a delivery
  that is still landing.

## Scratch removed a commit-pinned checkout mid-run

`/ess/scratch/scratch1/annawoodard/hfdp-worktrees/6caebb075e16cb7f70af277a11a60a4b33cd18c2`
was emptied at 2026-08-28 10:17 while its jobs were still running, leaving only the `logs/`
subdirectory the wrapper had created. All 32 shards plus the shard-15 resubmit had already
completed and all 867 exam artifacts were safely written under `/gpfs`, so no imaging work
was lost, but reducer `14499577` then failed with

```
repository root does not contain the cluster environment owner: /ess/scratch/.../6caebb07...
```

which reads like a code fault and is actually a vanished checkout. Slurm snapshots the
submitted wrapper at `sbatch` time, but the wrapper re-resolves `REPO_ROOT` and re-sources the
cluster profile from disk at task start, so a job outliving its checkout dies late.

Recovered by recreating the same pinned commit under
`/gpfs/data/huo-lab/Image/annawoodard/hfdp-worktrees/inv-reduce-<sha>` and resubmitting reduce
as `14499961` with `HFDP_REPO_ROOT` set explicitly.

This matters beyond this run: hfdp docs use `/ess/scratch/.../hfdp-worktrees/<sha>` as the
standard location for commit-pinned checkouts, and the v5 release README tells a reader to
reproduce the release from exactly such a path. If scratch is purged on a short timer, those
reproduction instructions do not survive their own retention window. Commit-pinned checkouts
backing a release or a queued job should live on `/gpfs`.

## Two defects found in passing

`scripts/slurm_build_delivered_dicom_inventory.sh` (hfdp) computes
`INVENTORY_TIME="${INVENTORY_TIME:-$(partition_maximum_walltime)}"` at the top of the script,
but the `submit` branch does not put `INVENTORY_TIME` in either `sbatch --export` list. Every
worker and the reducer therefore re-run `sinfo` at task start for a value they never use, and
a transient scheduler miss kills the task. Fix is to export the resolved value with the other
`INVENTORY_*` variables, or to resolve it only inside the `submit` branch. Not applied here:
hfdp is not this repo and the change wants its own review.

`scripts/scan_landed_series_ultrafast.py` (hfdp) skips only `delivery_integrity` when walking
the delivery root, so it treats `derived/<patient>/<visit>` as exam directories. Its
`provenance.json` reports `exams_scanned: 1235` where only 867 are exams. The 368 spurious
rows all carry `has_ultrafast=false` and never join anything, because every consumer joins on
transfer-receipt relative paths, so no count downstream is wrong -- but the provenance figure
is. Fix is to skip `derived` alongside `delivery_integrity`.

## Results on the refreshed inventory

Inventory `retro_caps_pcr_imaging_inventory_all_delivered_exams_v2`, reducer `14499961`,
commit `6caebb075e16cb7f70af277a11a60a4b33cd18c2` clean: 867 exams, 9406 image series, 15230
series directories, 4,298,208 instances, 0 unreadable, 0 direct read errors, no consumer
column missing.

Gate audit `retro_caps_two_series_gate_readiness_all_delivered_v2`, up from the 632-exam run:

| | stale (632 exams) | refreshed (867 exams) |
| --- | --- | --- |
| carry both series | 263 | 352 |
| pass all gates | 253 | 341 |
| patients passing | 247 | 277 |
| layouts | 245 single + 8 split | 333 single + 8 split |

Exclusions: 9 `orientation_not_axis_aligned`, 1 `orientation_absent_or_unparseable`, 1
`split_pair_repetition_time_differs`.

Funnel from the 341 gate-passing exams to what Sarit can actually use:

| filter | exams | patients |
| --- | --- | --- |
| pass all gates | 341 | 277 |
| not already in v5 | 216 | 186 |
| + published phase export | 125 | 120 |
| + usable binary pCR label | 89 | 85 |
| + single primary | 78 | 74 |
| + single-series HR (v5's pair contract) | **73** | 69 |

So v5's 243 exams become **316**, pCR 0/1 going 144/99 -> 192/124. The addition is pCR 48/25,
labelled 59 `oncotrace_glm52` (provisional), 12 `regex_proxy`, 2 `anna_manual`. Payloads fully
verified: 1424 of 1424 phase files and 73 of 73 times arrays present on disk, every exam at
`baseline_frame_count = 5`, matching the builder's `POLICY_BASELINE_FRAMES`.

### The binding constraint moved to the phase export

Of the 216 gate-passing exams absent from v5, 91 are blocked only because no raw-signal phase
export has been published; 82 of those are `pretreatment`. Running the export on them would
unlock a further **79** exams that already clear label, single-primary and single-series
checks (pCR 51/28). That is the highest-value next compute step:

    243 today  ->  316 addable now  ->  395 once the 91 phase exports run

Blocking reasons across all 143 blocked exams: 91 no phase export, 22 label absent, 15
`not_evaluable_or_conflict`, 29 multiple-primary, 8 split-HR pair. Per-exam detail with both
series UIDs is in
`/gpfs/data/huo-lab/Image/annawoodard/hfdp/outputs/sarit_v6_extension_candidates/extension_candidates.csv`
(216 rows, `tier` = `addable_now` or `blocked`, plus `summary.json`).

## Open decision

Whether newly eligible exams join the consumer manifest as flagged rows with empty
`phase_files` and `phase_export_status=pending`, sit in a sibling extension CSV, or wait for
their phase exports to be produced first. Raised with Anna; proceeding meanwhile with the
work that is common to all three (inventory, gate refresh, candidate determination, raw
locator resolution, extended `paired_preprocessing_case_manifest.csv`).

## Launching the 91 missing phase exports (2026-08-28)

Anna authorised launching the phase exports and accepted provisional `oncotrace_glm52`
labels, with one caveat worth checking first: retro-CAPS now automates phase exports into
`/gpfs/data/karczmar-lab/DR_662135_Karczmar_Breast_MRI_ML/retro_caps_pcr_imaging_cohort/derived`,
so some of the 91 might already exist.

### They did not already exist

`derived/` holds 377 series exports across 375 patient directories, laid out as
`derived/<patient>/<visit>/<SeriesInstanceUID>/motion_corrected/phase_NNNN.nii.gz` plus
`times_seconds.csv`, `breast_mask.nii.gz` and a per-series `README.md`. Joining on
`SeriesInstanceUID` against the 216 extension candidates:

| | count |
|---|---|
| candidates with a phase export in the raw-signal cache | 125 |
| ...of those, also published under `derived/` | 124 |
| candidates with no phase export in the cache | 91 |
| ...of those, present under `derived/` | **0** |

So the partition is exact and the cache manifest is the authority: `derived/` is a
*publication* of the cache, not an independent producer, and the automation has not reached
any of the 91. The one cache entry not yet in `derived/` is a publication lag, not a missing
export. The remaining 253 `derived/` series belong to v5 exams or to exams outside the
candidate set.

### The producer chain, and the stage that was actually needed

`derived/` is the last step of a five-stage chain, not something to write into directly:

1. `scripts/build_retro_caps_two_series_phase_manifest.py` — turns the gate-readiness audit
   into a dynamic-T1 phase manifest, ultrafast half only.
2. `scripts/build_union_selection_manifest.py` — unions the two lineages that resolved
   acquisitions for this delivery, deduplicated on `anchor_series_instance_uid`.
3. `scripts/build_ultrafast_raw_signal_cache.py` — `prepare`/`shard`/`merge`; this is what
   writes the phase `.nii.gz` files and `times.json` and emits `raw_signal_manifest.csv`.
4. motion correction and breast-mask caches.
5. `scripts/publish_derived_alongside_delivery.py` — copies the result next to the delivery.

`scripts/slurm_retro_caps_two_series_raw_signal_ingest.sh` looks like the phase-export owner
and is not: it feeds the two-series concentration cohort, and stage 3 runs
`preprocess_dynamic_t1.py` itself against its own task's inventory root. Running the ingest
would have duplicated ~340 exams of DICOM preprocessing into a store nothing in Sarit's
chain reads. Stages 1, 2 and 3 are the whole job.

### Stage 1 — phase manifest over all 867 delivered exams

Job `14501153`, submitted from the clean `/gpfs`-backed pinned worktree
`/gpfs/data/huo-lab/Image/annawoodard/hfdp-worktrees/inv-reduce-6caebb075e16cb7f70af277a11a60a4b33cd18c2`
(commit `6caebb075e16cb7f70af277a11a60a4b33cd18c2`, 0 dirty). Completed in 1:46. Output root
`outputs/runs/hfdp/retro_caps_two_series_phase_manifest_all_delivered_v2`.

341 gate-passing exams → 2 excluded as already held in the two-series cohort, 1 deferred,
**338 buildable, 7112 native phases**, 16 shard manifests of 21 exams each. All 250 anchors
and all 250 `exam_id` values from the previous selection are reproduced exactly, which is
what makes the downstream expansion additive rather than a repartition.

Three of the 91 fall outside this build, all typed rather than dropped:

| patient | exam | pCR | why |
|---|---|---|---|
| 36119213 | 2016-12-22 | 0 (`oncotrace_glm52`) | already held in the two-series cohort on series-UID identity |
| 55499701 | 2021-07-01 | — (no label) | same; has no usable label anyway |
| 91143867 | 2024-12-06 | 0 (`oncotrace_glm52`) | gate audit named `SubMIP dyn_eTHRIVE UFAST EXTEND`, a `PROJECTION IMAGE` MIP subtraction, as its ultrafast half |

The third is the same pre-existing defect the 2026-08-25 hfdp entry recorded: the
gate-readiness audit has no derived/projection exclusion, and the phase-manifest builder
catches it downstream by replaying the pipeline's own dynamic-series exclusion. Recovering it
needs the gate audit fixed, not a workaround here. So **88 of 91 are buildable**, and only 2
of the 3 residuals would have contributed a labelled exam.

### Stage 2 — union selection

Ran directly (a few hundred CSV rows). Config written outside the repo at
`outputs/runs/hfdp/sarit_phase_export_extension_v2/configs/`, with absolute paths, so the
pinned worktree stays clean and no repo edit was needed. Output root
`outputs/retro_caps_pcr_union_selection_all_delivered_v2`.

Released lineage first and byte-identical (288 acquisitions, 0 dropped), then the new
two-series lineage: 338 rows, 143 superseded on anchor UID, 195 contributed.
**483 union acquisitions over 396 patients**, up from 395 over 386.
`superseded_with_disagreeing_phase_counts: 0`.

### Stage 3 — raw-signal cache, extended in place

The cache is append-only by design (`hfdp/data/incremental_shard_plan.py`): `prepare` retains
every existing task row verbatim and creates tasks only for exams no existing task covers,
because the shard partition is a stride over the sorted unit list and repartitioning would
move nearly every exam into a different output directory and recompute the whole cohort. So
this extends `derived_datasets/hfdp/retro_caps_pcr_raw_signal_cache` in place rather than
starting a second cache root that every consumer would then have to reconcile. The run root
keeps an append-only `config_snapshot_history.csv`, so which config produced which tasks
stays recoverable. Pre-change state is backed up under
`outputs/runs/hfdp/sarit_phase_export_extension_v2/pre_prepare_snapshot/`;
`raw_signal_manifest.csv` was sha256 `72916a86...` with 395 rows going in.

`prepare` reported exactly the intended increment:

```
retained_tasks 17   retained_units 395
remaining_units 88  first_added_index 17   tasks 23
```

Only the `retro_caps_pcr_imaging` dataset entry changed, repointed at the rebuilt inventory
`retro_caps_pcr_imaging_inventory_all_delivered_exams_v2` and the stage-1 phase manifest.
Retained task rows keep the inventory root they were built against, which is what makes
repointing safe.

Array `14501207` (tasks 17-22, 14-15 exams and 270-327 phases each, 8 CPU / 96 G / 6 workers
on `tier1q`) → merge `14501208` chained `afterok`. `afterok` rather than `afterany` here so a
failed shard cannot produce a partial `raw_signal_manifest.csv`.

### A submission trap worth remembering

Two jobs died at 00:00:00 before anything ran, for two separate reasons:

- `--output` pointed into the agent scratchpad under `/tmp`, which is node-local. The compute
  node cannot write the submitting host's `/tmp`, so the job fails before the script starts
  and there is no log to explain it. Slurm logs must land on `/gpfs`.
- `sbatch -D <dir>` sets the working directory but does *not* set `SLURM_SUBMIT_DIR`, so a
  script that does `cd "${SLURM_SUBMIT_DIR}"` lands in the original cwd. Pass the repo root
  explicitly (`--export=ALL,WT_ROOT=...`) instead of relying on `-D`.

### Result

All six tasks COMPLETED in 8:26-12:43 (peak RSS ~54 G against the 96 G request); merge
`14501208` COMPLETED in 39 s. Cache summary: **483 raw exams, 0 failures**, 466 physics
eligible (was 378), `input_exclusions` 17 unchanged.

`raw_signal_manifest.csv` went 395 -> 483 rows, 88 added, **0 removed**. Every one of the 88
resolves: 1804 phase files and 88 `times.json`, none missing, all
`baseline_frame_count=5`, all `hfdp_t1_raw_signal_v2`, 13-24 phases.

The append left the existing rows alone where it matters. All 395 pre-existing rows differ in
exactly one column, `source_manifest_original`, which correctly now names the v2 union
selection this merge ran under. `phase_files_json`, `times_path`, `times_seconds_json`,
`n_phases`, `baseline_frame_count`, `study_instance_uid`, `dataset`, `policy_name` and
`preprocessing_contract` are unchanged on all 395. Independently: v5's 243 rows still resolve
all 4974 of their phase files and all 243 `times_path` values.

### What this unlocks

213 of the 216 gate-passing candidates now carry a published phase export, up from 125.
Applying the same admission rules v5 uses (usable pCR label, single primary, `single_series`
HR partner, `exam_role=pretreatment`):

| filter | exams |
|---|---|
| gate-passing, not already in v5 | 216 |
| with a published phase export | 213 |
| usable pCR label | 177 |
| single primary | 156 |
| `single_series` HR partner | 151 |
| pretreatment | **139** |

So the addable increment went from 73 to **139**, and v5's 243 would become **382**. The
addition is 89 pCR-negative / 50 pCR-positive over 116 patients, and its labels are 121
`oncotrace_glm52`, 14 `regex_proxy`, 4 `anna_manual` — so about 87% of the addition rests on
unreviewed first-draft LLM chart abstraction, which Anna has accepted as provisional. All
2790 phase files for the 139 are present.

Remaining blockers, now that the phase export is no longer one: 20 multiple-primary, 17 label
absent, 11 `not_evaluable_or_conflict`, 12 not pretreatment, 5 split-HR partner, 3 the typed
phase-export residuals above, plus combinations. Refreshed per-exam detail, now carrying
`cache_exam_id`, `times_path` and `phase_files_json` per row, is in
`/gpfs/data/huo-lab/Image/annawoodard/hfdp/outputs/sarit_v6_extension_candidates_post_phase_export/`.

### Scope not covered

Only the phase export ran. The downstream motion-correction cache, breast-mask cache and
`publish_derived_alongside_delivery.py` were not run for the new 88, so `derived/` still holds
377 series and the new exports live only in the cache shards. That is the right scope for
Sarit: v5's `phase_files` already point at the cache shards, not at `derived/`, so the
extension needs nothing new from him. The `derived/` publication matters to the PK/AIF
consumers, and it is a separate launch.

The v6/v7 releases are still unbuilt, still waiting on the additive-fold decision in **Open
decision** above.
