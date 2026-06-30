# Lab Notebook — Faster Segmentation

Chronological record of the optimization effort. Newest entries at the bottom.
Each entry: what was tried, the result, and key findings.

---

## 2026-06-18 — Setup & investigation

**Goal:** Speed up the ~3-day vessel-segmentation inference run (currently 52%
complete) without losing accuracy. Reuse completed outputs as ground truth.

**What I did:**
- Read the full pipeline: `segmentation/batch_segmentation.py` →
  `vanguard-blood-vessel-segmentation/{predict.py, model_utils.py, dataset_3d.py}`.
- Located real paths on randi (the `/net/projects2/...` defaults don't exist here):
  - Images: `/ess/scratch/scratch1/annawoodard/MAMA-MIA-syn60868042/images` (1507 cases, read OK).
  - Ground truth: `/ess/scratch/scratch1/t-9sbose/vessel_segmentations/` (4287 `.npz`).
- Confirmed env `vanguard`: torch 1.11.0, torchio 1.2.1, unet 0.8.1, SimpleITK 2.5.5.
- Inspected `.npz`: single key `vessel`, shape e.g. `(512,512,174)`, float16, values in [0,1].

**Key findings (bottlenecks):**
1. Vessel stage = 192 forward passes/volume at `batch_size=1`, `num_workers=1`, FP32.
   This is the dominant cost — GPU mostly idle.
2. Breast stage is cheap (1 forward pass/volume).
3. Pipeline shells out to `predict.py` twice per task (Python startup + model reload each time).
4. STEP-1 preprocessing is serial.

**Decision:** Reuse the existing `Dataset3DDivided` patch extraction verbatim and
only change *how patches are fed to the GPU* (batch them) + accumulation, so the
batched path is mathematically identical to the original except for FP precision.
Batch size 16, AMP enabled (per user). Build incrementally; test each feature.

**Note:** The live array job `12070022` (the 52% run) is still running. All new
work writes to a separate output dir and uses SLURM for any GPU execution.

---

## 2026-06-18 — Step 1: Batched vessel inference

**What I tried:** Wrote `predict_fast.predict_vessel_batched` — reuses the unchanged
`Dataset3DDivided` (identical patches/order/placement) but feeds patches to the
model in mini-batches and scatter-accumulates results per subject. Added an
`accum_dtype` knob (prod=float32; test=float16 to match the original exactly).

**Test (CPU, login node):** `tests/test_batching_equiv.py` — tiny synthetic 2-subject
data + deterministic Conv3d(2→3), runs the original `pred_and_save_masks_3d_divided`
vs the batched path with `batch_size=7` (straddles the subject boundary at patch 18).

**Result:** PASS.
```
subjA: shape=(40, 40, 30) bit_identical=True max_abs_diff=0
subjB: shape=(36, 44, 28) bit_identical=True max_abs_diff=0
```

**Key findings:**
- Batching + scatter-accumulate is mathematically exact: float16-accum mode is
  bit-identical to the original because softmax is per-voxel, eval-mode forward is
  batch-independent, and per-voxel add order is preserved.
- Production will use float32 accumulation (strictly more accurate; final cast to
  float16 on save to match the ground-truth `.npz` dtype). Expect negligible, not
  bit-zero, diffs vs the existing run — quantified later in Step 5.
- Practical note: `import torch` (1.11) on this networked FS takes ~60–80s per
  process; factor that into smoke-test wall-clock, not just GPU time.

---

## 2026-06-18 — Step 2: AMP (mixed precision)

**What I tried:** Added `_autocast()` + `use_amp` to both inference functions in
`predict_fast.py`. Guarded so autocast is enabled only when `use_amp and CUDA`;
on CPU it is a no-op (outputs stored as float16 regardless).

**Test:** AMP genuinely cannot be exercised on CPU (no-op), so there is no
honest CPU test for it. Its real effect (speed + numerical impact) is measured
in the Step 4 GPU smoke run. The Step 1 CPU equivalence test already confirms the
surrounding code path runs correctly with `use_amp=False`.

**Finding:** Target GPUs on `gpuq` are **A100-PCIE-40GB** — AMP is well supported
and 40 GB easily fits batch_size=16 of [2,96,96,96], so no OOM concern.

---

## 2026-06-18 — Step 3: In-process driver + parallel preprocessing + num_workers

**What I tried:** `batch_segmentation_fast.py` — reuses the original's
`find_nii_files` / `preprocess_image` / `build_output_path` (identical discovery &
output layout), but:
- STEP-1 preprocessing runs across a `ThreadPoolExecutor` (`--preprocess-workers`).
- STEP-2 (breast) and STEP-3 (vessel) run in-process (no subprocess), each model
  loaded once and kept on GPU; `--num-workers` for DataLoader prefetch.
- Added batched breast stage (`predict_breast_batched`) — breast inputs are a
  uniform (144,144,96) after resize, so subjects batch directly.

**Test (CPU, login node):** `tests/test_preprocess_parallel.py` — 3 tiny synthetic
.nii.gz, parallel vs serial preprocessing.

**Result:** PASS.
```
CASE_001_0000: shape=(16, 16, 8)  identical_to_serial=True
CASE_002_0000: shape=(14, 18, 6)  identical_to_serial=True
CASE_003_0000: shape=(12, 12, 10) identical_to_serial=True
```

**Finding:** Parallel preprocessing is order-independent and matches serial exactly.
Full in-process GPU inference path is validated next (Step 4). Ground-truth cases
for validation: DUKE_001 files are at global indices 0–3 (out of 7926) and all
have completed ground truth — so the smoke default range 0–1 overlaps ground truth.

---

## 2026-06-18 — Step 4: GPU smoke test submitted

**What I did:** Submitted `submit_fast_smoke.slurm` as a single (non-array) GPU job
→ job `12075155` on `gpuq`. Config: files 0–1 (DUKE_001_0000/0001, both have ground
truth), batch_size=16, num_workers=3, preprocess-workers=4, AMP on. Output to a
SEPARATE dir `/ess/scratch/scratch1/t-9sbose/vessel_segmentations_fast_smoke` and
temp in `/tmp` — cannot disturb the live run.

**Result:** COMPLETED (job `12075155`, node `cri22cn403`, A100-PCIE-40GB).
- Preprocessing: 2.4s for 2 files (parallel workers)
- Inference: 24.4s for 2 files (384 subvolumes total, 192/file, batch_size=16 → 24 GPU calls)
- Total: 26.9s → **13.5s/file** (vs ~3 days / 7926 files ≈ ~33s/file for original)
- Wall-clock from job start to finish: 29s (Python startup ~3s — already cached/warm)
- Both DUKE_001 outputs confirmed present in output dir

---

## 2026-06-22 — Step 5: Full-dataset array submission

**What I did:** Created `faster-segmentation-test/submit_fast_array.slurm` and
`faster-segmentation-test/submit_fast_array.sh` to run the fast pipeline over all
7926 `.nii.gz` files in the same output dir
`/ess/scratch/scratch1/t-9sbose/vessel_segmentations_fast_smoke`.

**Config:**
- FILES_PER_TASK=20 → 397 tasks (array 0–396), up to 16 concurrent
- Time limit: 4:00:00 per task (conservative; ~5–7 min expected per task)
- `--resume` flag: safe to resubmit if jobs fail; already-done files are skipped
- Logs: `/ess/home/home1/t-9sbose/vanguard/logs/fast-seg-array-<JOBID>-<TASKID>.{out,err}`

**To submit:**
```bash
bash faster-segmentation-test/submit_fast_array.sh
```

**Pending:**
- Record actual job ID and wall-clock once submitted
- Run `validate_outputs.py` on a sample of outputs vs ground truth in
  `/ess/scratch/scratch1/t-9sbose/vessel_segmentations/` to confirm Dice/diff

---

## 2026-06-22 — Step 5b: Output validation on DUKE_001

**Goal:** Confirm the fast pipeline preserves accuracy vs the existing ground-truth
run for DUKE_001 (the two completed smoke-test outputs from Step 4).

**Action:**
```bash
module load gcc/11.3.0 python/3.10.5
python3 faster-segmentation-test/validate_outputs.py \
    --fast-dir /ess/scratch/scratch1/t-9sbose/vessel_segmentations_fast_smoke/DUKE/DUKE_001 \
    --truth-dir /ess/scratch/scratch1/t-9sbose/vessel_segmentations
```

**Evidence:**
```
case                               shape_ok  max_abs   mean_abs   dice    vox_truth   vox_fast   ratio
DUKE_001_0000_vessel_segmentation.npz True   0.0039    0.000001   1.0000  16145       16145      1.000
DUKE_001_0001_vessel_segmentation.npz True   0.0095    0.000001   0.9998  39217       39207      1.000

Compared 2 case(s).
  Dice:     min=0.9998  mean=0.9999
  Max|diff|: max=0.0095  mean=0.0067
```
Report: `/ess/scratch/scratch1/t-9sbose/vessel_segmentations_fast_smoke/DUKE/DUKE_001/validation_report.txt`

**Result:** Both timepoints pass — shapes match, Dice ≥ 0.9998, mean|diff| < 0.000002.

**Conclusion:** Differences are consistent with float32→float16 rounding on save (max
representable gap at these values is ~0.001–0.01). Voxel counts agree to within 10 voxels
on a 39 k-voxel mask — negligible. The fast pipeline is numerically equivalent to the
original for this case. **Inference confidence: high.**

**Decision impact:** Step 5 validation item is now closed. The fast pipeline is confirmed
accurate on at least 2 timepoints; the full-array run (Step 5) can proceed with confidence.

---

## 2026-06-23 — Skeleton generation bottleneck: 33 missing cases

**Goal:** Identify and resubmit skeleton extraction for cases that did not have
centerline files generated. The pipeline should have produced 1506 skeleton files
(matching 1506 segmentation cases) but only 1473 were found.

**What I did:**
- Used `scripts/inventory.py --stage centerlines` to compare segmentation directory
  (1506 cases) against centerline output directory (1473 skeleton files).
- Found 33 missing cases:
  - DUKE: 19 cases (DUKE_012, _019, _021, _022, _045, _046, _069, _101, _119,
    _142, _168, _233, _234, _258, _307, _378, _400, _489, _491)
  - ISPY2: 14 cases (ISPY2_239061, _255388, _255535, _275626, _277848, _277888,
    _287300, _287961, _299840, _311316, _311455, _313243, _317641, _318293)
- Created `graph_extraction/slurm/submit_missing33_centerlines.sh` wrapper to
  resubmit centerline extraction for exactly these 33 cases using the existing
  `submit_tc4d_array.slurm` job template.

**Resubmission:**
```bash
bash graph_extraction/slurm/submit_missing33_centerlines.sh
```

**Status:** Submitted job 12152628 (array of 33 tasks, max_concurrent=10, partition=tier1q).

**Job 12152628 Completion:** All 33 tasks completed successfully (0:0 exit codes).
- Total elapsed: ~9 minutes
- Skeleton files created: 1506 total (all 33 missing cases now have skeleton_4d_exam_mask.npy)
- Verified: All 33 cases present and valid

---

## 2026-06-23 — MP4 rendering for 33 recovered skeletons

**What I did:** Submitted MP4 rendering job for the 33 newly created skeleton files using
the standard `graph_extraction/slurm/submit_skeleton_mp4_array.sh` script.

**Job Submission:**
```
Total skeletons found : 1506
Already rendered      : 1473
To render             : 33
Array job ID          : 12152829 (0-32, max_concurrent=32)
```

**Status:** Submitted and queued for execution.

**Next steps:**
- Monitor: `squeue -j 12152829`
- Verify MP4 files created in skeleton case directories

---

## 2026-06-25 — Fast segmentation over the RESAMPLED dataset → karczmar-lab GPFS

**Goal:** Run the validated fast vessel-segmentation pipeline over all resampled
volumes and write results to the lab GPFS workspace, fully reproducibly.

**Reproducibility (per CLAUDE.md):** every step below is a committed script; no
ad-hoc commands produced artifacts.
- Code state: branch `feature/segmentation-speedup-and-qc`, base commit `eb7ae18`.
  NOTE: the two new scripts are still UNCOMMITTED (`git status` shows `??`); they
  must be committed for the run to be fully reproducible from git alone.
- Environment: micromamba env `vanguard` (activated inside the slurm scripts).

**What I did:**
- Diagnosed a layout mismatch: `find_nii_files` expects a FLAT `images_dir/<CASE>/`,
  but the resampled data is nested `<COHORT>/<CASE>/`.
- `scripts/build_resampled_flat_farm.sh` — idempotent symlink farm
  (`/ess/scratch/scratch1/t-9sbose/resampled_segmentation_flat`, 1506 case links →
  7926 `.nii.gz`). No data copied; pipeline runs unmodified; output cohort is still
  derived from the case-name prefix.
- `scripts/run_resampled_vessel_segmentation.sh` — single documented entrypoint
  (MODE=smoke|full, DRY_RUN=1): builds the farm, ensures the output dir, submits to
  Slurm via `faster-segmentation-test/submit_fast_array.sh`.

**Paths:**
- Input (read-only): `/ess/scratch/scratch1/t-9sbose/resampled_segmentation` via the flat farm.
- Output: `/gpfs/data/karczmar-lab/workspaces/saritbose/resampled-vessel-segmentations`
  (nested `<COHORT>/<CASE>/images/<CASE>_NNNN_vessel_segmentation.npz`).

**Runs:**
- Smoke (job `12284899`, A100): files 0–1 (DUKE_001) → COMPLETED, 0 failed, output
  in correct nested GPFS path. DUKE_001 was in-range (copied), so its mask is
  byte-identical to the earlier validated smoke — confirms no regression.
- Full array (job `12284980`): 397 tasks (`0-396%16`), 20 files/task, partition `gpuq`.
  Submitted via `IMAGES_DIR=<flat farm> OUTPUT_DIR=<gpfs> bash
  faster-segmentation-test/submit_fast_array.sh`. Early health: 16 done / 16 running
  at ~13 s/file, 320 `.npz` written, 0 errors. `--resume` makes it restartable.

**Key findings:**
- gpuq = 5 nodes × 8 A100-40GB = 40 GPUs, so the %16 cap fits comfortably.
- Output ~250–400 GB expected (NACT ~6 MB/file … DUKE ~49 MB/file); GPFS had 2.9 TB
  free (87% used) — fits, will rise to ~89%.

**Pending:** record full-array wall-clock + final count on completion; commit the two
new scripts + CLAUDE.md reproducibility rule.

---

## 2026-06-25/26 — Full resampled-dataset run + ISPY1_1228 fix

**Goal:** Run fast segmentation over all 7,926 resampled volumes; fix the single failure.

### Main array (job 12284980) — 2026-06-25

**Script:** `scripts/run_resampled_vessel_segmentation.sh` → `submit_fast_array.sh`
**Input:** `/ess/scratch/scratch1/t-9sbose/resampled_segmentation_flat/`
**Output:** `/gpfs/data/karczmar-lab/workspaces/saritbose/resampled-vessel-segmentations/`
**Config:** 397 tasks, FILES_PER_TASK=20, MAX_CONCURRENT_TASKS=16, gpuq
**Branch:** `feature/segmentation-speedup-and-qc` @ `55b2478`
**Env:** `micromamba activate vanguard`
**Outcome:** 396 COMPLETED, 1 FAILED (task 91) → 7,906/7,926 `.npz`

### ISPY1_1228 failure diagnosis

`Dataset3DDivided(z_division=3, input_dim=96)` with z=307 slices:
  step = (307−96)/(3−1) = 105 > 96 → middle slices uncovered → AssertionError

Fix: exposed `--z-division` (default=3). z_division=4 gives step=70 ≤ 96 for z=307.
Committed `a50e65a`: updated `batch_segmentation_fast.py`, `submit_fast_array.slurm`,
added `submit_task91_rerun.{sh,slurm}`.

### Rerun (job 12295362) — 2026-06-26

**Script:** `faster-segmentation-test/submit_task91_rerun.sh`
**Files:** 1820–1839, z_division=4
**Branch:** `feature/segmentation-speedup-and-qc` @ `a50e65a`
**Env:** `micromamba activate vanguard`
**Outcome:** COMPLETED, 20/20 — wall-clock 3 min 44 s
**Final total:** 7,926 / 7,926 `.npz` ✓

**Key findings:**
- Any volume with z > 288 will fail with z_division=3; add `--z-division 4` for those cases.
- The `--resume` flag makes reruns safe and idempotent.

---
