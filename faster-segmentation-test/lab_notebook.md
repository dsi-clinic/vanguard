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

**Status:** PENDING (Priority) — the live segmentation array `12070022` is saturating
`gpuq`. Per user direction, letting it queue naturally rather than perturbing the
live pipeline. Monitoring in background; will record timing + run Step 5 validation
(`validate_outputs.py`: Dice + max/mean abs diff vs DUKE_001 ground truth) on completion.

**Pending results to capture here when the job runs:**
- preprocess vs inference wall-clock split, and s/file vs the ~3-day baseline rate
- AMP numerical impact (Dice, max|diff|) vs the float16 ground truth
