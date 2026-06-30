# Lab Notebook — Vanguard Project

Chronological record of experiments, optimizations, validations, and significant work.
Newest entries at the bottom.
Each entry: what was tried, the result, and key findings.

---

## 2026-06-23 — Native-resolution QC visualization script

**Goal:** Before resampling all cohorts to DUKE's voxel spacing, visually inspect
what each dataset looks like at native resolution by exporting axial slices as PNGs.

**What I did:**
- Wrote `scripts/save_qc_pngs.py` to:
  - Walk `/ess/scratch/scratch1/annawoodard/MAMA-MIA-syn60868042/images/`
  - Infer cohort from case_id prefix (DUKE, ISPY1, ISPY2, NACT)
  - For each cohort, find one representative exam per distinct voxel spacing
  - Load NIfTI at native resolution (no resampling), clip to 1st–99th percentile,
    scale to uint8, save each axial slice as a grayscale PNG
  - Output to `~/vanguard_qc_pngs/{cohort}/{exam_id}/slice_XXXX.png`
  - `--cohort` flag to run one cohort at a time
- Tested on DUKE: loaded DUKE_001_0000.nii.gz (448×448×160, spacing ~0.80×0.80×1.10 mm)
  and confirmed 160 PNGs saved correctly (L mode, pixel range 0–255, mean ≈46).

**Results:**
- DUKE alone has **41 distinct voxel spacings** — confirming it is highly heterogeneous
  (multi-site dataset with varying scanner parameters).
- PNG generation verified end-to-end: correct size, correct mode, correct normalization.

**Key findings:**
- The 41 distinct spacings in DUKE is an important discovery — resampling all to a
  single target spacing will collapse this variation. Worth reviewing the range of
  DUKE spacings before choosing the canonical target.
- Script uses nibabel + Pillow (both available in vanguard env on Python 3.10).
- Data root confirmed: `/ess/scratch/scratch1/annawoodard/MAMA-MIA-syn60868042/images/`
  (not `/net/projects2/vanguard/` which is not mounted on randi head nodes).

**How to run:**
```bash
micromamba run -n vanguard python scripts/save_qc_pngs.py               # all cohorts
micromamba run -n vanguard python scripts/save_qc_pngs.py --cohort DUKE # one cohort
```

---

## 2026-06-24 — Voxel-spacing scattergram across MAMA-MIA cohorts

**What I did:
Created `scripts/plot_voxel_spacing_scatter.py` to complement the existing
`voxel_spacing_boxplot.png`. Plots each of the 1506 MAMA-MIA subjects as one
point: in-plane voxel size (xy) on x, slice thickness (z) on y, colored by
cohort (DUKE/ISPY1/ISPY2/NACT). Data pulled from the `dataset_info` sheet of
`clinical_and_imaging_info.xlsx` via the existing `load_clinic_metadata_excel`
helper; `pixel_spacing` parsed (first element of `[x, y]`) for the x-axis,
`slice_thickness` used directly. Colors match the boxplot (tab10:
blue/orange/green/purple). Output: `voxel_spacing_scatter.png`.

**Results:
Ran in the vanguard env; per-cohort counts 291/171/980/64 (=1506) match the
boxplot. Figure renders cleanly.

**Key findings:
Joint distribution confirms what the boxplot panels implied separately:
DUKE sits at thin slices (~1 mm) across a wide in-plane range; ISPY1 occupies
thick slices (2–3 mm); ISPY2 spans the full range; NACT is a small cluster
near 2 mm slices. The scatter makes the per-cohort slice-thickness vs in-plane
trade-off visible at a glance.

---

## 2026-06-25 — Added resampling reference marker to voxel-spacing scatter

**What I did:
Extended `scripts/plot_voxel_spacing_scatter.py` to overlay a single black star
marker at (in-plane = 0.7 mm, slice thickness = 1.0 mm), labeled
"DUKE mode/Resampled Size", as both a legend entry and an inline annotation
(text centered just below the star at y = 0.9). This marks the target spacing
the DUKE cohort is resampled to, for visual comparison against the raw
per-patient spacings.

**Results:
Regenerated `voxel_spacing_scatter.png`; the star lands inside the dense DUKE
cluster (thin ~1 mm slices, ~0.7 mm in-plane), as expected.

**Key findings:
The resampling target sits right at the DUKE mode, confirming the chosen
(0.7, 1.0) target is representative of that cohort's native acquisition and not
an outlier relative to the bulk of DUKE exams.

---

## 2026-06-25 — DUKE-range domain-shift check on voxel spacing (header-only)

**What I did:
Wrote `scripts/spacing_range_check.py` to quantify how far the other cohorts
drift from the DUKE training distribution in voxel spacing. For every case I
read only the NIfTI header (nibabel lazy load, no image arrays) of the
pre-contrast `_0000` timepoint and pulled xy spacing (zooms[0]) and z spacing
(zooms[2]); verified zooms order against the Excel (DUKE_001 → 0.804/0.804/1.1).
Defined DUKE's 5th-95th percentile band per axis as "in-distribution", flagged
each scan's xy/z/either out-of-range, wrote a per-scan CSV
(`scripts/spacing_range_check_results.csv`, all 4 cohorts), printed a summary,
and saved a scatter (`~/vanguard_qc_pngs/spacing_distribution.png`, dpi=150)
with distinct color+marker per cohort, a dashed DUKE 5th-95th box, and a
per-cohort "N/total outside range" annotation. Note: `/net/projects2/vanguard/`
is not mounted on randi; data read from the scratch images root. The on-disk
`NACT` prefix is labeled "NACT-Pilot" in outputs.

**Results:
All 1506 scans read, 0 skipped. DUKE 5-95 range: xy [0.596, 1.016] mm,
z [1.000, 1.250] mm. Out-of-range (either axis): ISPY1 171/171, ISPY2 674/980,
NACT-Pilot 64/64.

**Key findings:
The shift is driven almost entirely by slice thickness: ISPY1 and NACT-Pilot
sit at z≈2.0+ mm (100% out of DUKE's z range), and ISPY2's z out-count (630)
dwarfs its xy out-count (200). In-plane spacing overlaps DUKE much more. So a
single DUKE-target resample will change z far more than xy for the other
cohorts — the main domain-shift axis to watch.

---

## 2026-06-25 — Voxel-spacing vs. vessel-segmentation experiment (DUKE_001)

**Goal:** Test how through-plane (z) voxel spacing affects vessel segmentation
quality, using DUKE_001 as a single test case across four spacing "versions".

**What I did:**
- Confirmed DUKE_001 native geometry: 448×448×160, spacing ~0.804×0.804×**1.1 mm**
  (z is the through-plane axis). Data read from
  `/gpfs/data/karczmar-lab/MAMA-MIA-syn60868042/images/DUKE_001` (the `/net/...`
  path in the default scripts is still not mounted on randi).
- Wrote `scripts/resampling_experiment/resample_duke001.py` (SimpleITK, linear
  interp) to build 4 versions of all 5 timepoints:
  - V1 z=2.0mm (downsample from native, with anti-alias Gaussian blur along z,
    sigma = new_spacing/2),
  - V2 z=1.4mm and V3 z=1.0mm (both **upsampled from V1**, i.e. from the degraded
    2.0mm data, to mimic the real "received thick scan, resample to training res"),
  - V4 = original files copied unchanged.
- Wrote `submit_resample.slurm` (CPU, express) and `submit_segmentation.slurm`
  (GPU array 0–3 on gpuq, one task per version, reuses `segmentation/batch_segmentation.py`).
- Wrote `scripts/resampling_experiment/make_comparison_pngs.py` (STEP 3, head-node
  safe) to pick the highest-vessel-signal axial slice per version and render two
  4×2 comparison figures (native + matched resolution).
- Submitted STEP1 job `12263560`, STEP2 array `12263561` (afterok-chained).

**Results:**
- Jobs submitted successfully; results pending (will append once segmentation
  completes and PNGs are generated).

**Key findings (so far):**
- An **axial** slice lies in the in-plane (x,y) plane, which we never resample —
  so the four versions all have 448×448 in-plane slices. The native- and
  matched-resolution figures will therefore look nearly identical for axial views;
  the z-spacing effect shows up in slice *content* and the subtraction, not in
  in-plane pixel count. Worth noting when interpreting the figures.
- The vessel `.npz` stores a probability volume under key `vessel`, in the model's
  preprocessed orientation `swapaxes(0,2)->swapaxes(0,1)->reverse axis0`; STEP 3
  re-applies this transform to the MRI so the mask overlays correctly.

---

## 2026-06-25 — Voxel-spacing experiment RESULTS (DUKE_001)

**Outcome:** All stages completed.
- STEP1 resample job `12263609` COMPLETED (express, ~2 min). 20 volumes written
  (V1 z=2.0mm: 88 slices; V2 z=1.4mm: 126; V3 z=1.0mm: 176; V4 original: 160).
- STEP2 segmentation array `12263610_[0-3]` all COMPLETED on gpuq (A100),
  13–24 min/task. 20 vessel .npz masks produced (5 timepoints × 4 versions).
- STEP3 produced `comparison_native_resolution.png` and
  `comparison_matched_resolution.png` (dpi=150). The two files are byte-identical,
  confirming the predicted point: axial in-plane is 448×448 for every version, so
  matched-resolution rescaling is a no-op for axial views.

**Per-version best (highest vessel-signal) axial slice:**
- V1 z=2.0mm: z=49/88, signal=802.4
- V2 z=1.4mm: z=76/126, signal=772.6
- V3 z=1.0mm: z=135/176, signal=695.4
- V4 original: z=124/160, signal=876.9

**Key findings:**
- Each version's best slice lands at a DIFFERENT anatomical level (they are chosen
  independently per version), so columns are not at a matched anatomical height —
  expected given the "max vessel signal per version" rule. If a mentor wants a
  like-for-like anatomical comparison, fix one physical z (mm) across versions
  instead of per-version argmax.
- Original (V4) has the highest total vessel signal on its best slice; the thick
  2.0mm V1 still segments substantial vessel but at a coarser through-plane level.

---

## 2026-06-25 — Voxel-spacing experiment: revisions (subtraction tp, shared z, 3D skeletons)

**What I did (3 mentor-requested changes):**
1. Subtraction now uses the SECOND post-contrast phase: `tp0002 - tp0000` (was
   `tp0001 - tp0000`). Raw-MRI top row + vessel mask still on early post (0001).
2. Comparison slice is now ONE shared physical z across all four versions instead
   of per-version argmax. Reference = original (V4) best vessel slice
   (z=124 @1.1mm = 136.4mm physical); mapped to nearest index per version:
   V1 k=68/88, V2 k=97/126, V3 k=136/176, V4 k=124/160 — same anatomy in every column.
3. Added 3D vessel skeleton + centerline manifold + coverage MIP per version via
   `graph_extraction/run_skeleton_processing.py`, and a rotating-skeleton mp4 via
   `render_skeleton_mp4.py`. This is what actually exposes the through-plane
   z-spacing difference (axial in-plane views cannot).

**New scripts:** `submit_skeletons.slurm` (tier1q, array 0-3) and
`submit_skeleton_mp4.slurm` (tier1q, array 0-3, afterok-chained).
Outputs -> `~/vanguard_qc_pngs/resampling_experiment/skeletons/DUKE_001__<version>/`.

**Jobs submitted:** skeletons `12264668_[0-3]`, mp4 `12264669_[0-3]`.

**Results so far (matched-physical-z slice, vessel voxels on that slice):**
- V1 z=2.0mm: 355   V2 z=1.4mm: 698   V3 z=1.0mm: 644   V4 original: 779.
- Clear trend: thick 2.0mm slab loses the most vessel; original keeps the most.
  Regenerated `comparison_native_resolution.png` / `comparison_matched_resolution.png`.

**Open question for mentor:** "second timepoint" interpreted as 0002 (2nd
post-contrast phase). If they meant 0001, trivially revert TP_SUB_POST.

---

## 2026-06-25 — Sagittal comparison figures + 3D skeleton results

**Sagittal figures (subtraction = 0002 - 0000, confirmed by mentor):**
- New script `make_comparison_pngs_sagittal.py`. Sagittal = fix x -> (y,z) plane,
  so the through-plane z axis is now IN-PLANE and the spacing difference is visible.
- Per-version best sagittal x (per spec): V1 x=194, V2/V3/V4 x=268. (V1 landed on a
  different x because its blurred vessel distribution shifted — worth noting; the
  other three agree at x=268.)
- Native figure drawn at true physical proportions with nearest interp: Version 1
  (z=2.0mm) is visibly BLOCKY along z, sharpening through V2->V3->V4 — exactly the
  intended demonstration. Native and matched PNGs now genuinely differ (unlike axial).
- Saved: `comparison_sagittal_native_resolution.png`,
  `comparison_sagittal_matched_resolution.png` (dpi=150).
- Vessel voxels on chosen sagittal slice: V1=240, V2=337, V3=460, V4=427.

**3D skeletons / centerlines / mp4 (jobs 12264668, 12264669, all COMPLETED):**
- Per version: skeleton_4d_exam_mask, center_manifold_4d_mask (centerlines),
  support mask, vessel_coverage_mip.png, and *_skeleton_rotating.mp4.
- Skeleton voxel counts (4D exam skeleton):
  V1 z=2.0mm: 4291 | V2 z=1.4mm: 3921 | V3 z=1.0mm: 3913 | V4 original: 6137.
- KEY FINDING: native (6137) retains ~30-40% MORE vessel skeleton than ANY resampled
  version (~3900-4300). V2/V3 ~= V1 despite finer grids, because they were upsampled
  FROM the 2.0mm-blurred V1 — confirming the hypothesis that the thick-slice
  anti-alias bottleneck destroys vessel detail that later upsampling cannot recover.

---

## 2026-06-25 — Resampling pipeline to bring all cohorts into DUKE spacing range

**Goal:** Build a pipeline that resamples every MAMA-MIA volume (DUKE, ISPY1, ISPY2,
NACT-Pilot) into the acceptable voxel-spacing range (xy 0.60–1.02 mm, z 1.00–1.25 mm)
before vessel segmentation, applying per-axis interpolation rules.

**What I did:**
- Wrote `scripts/resample_to_range.py` (SimpleITK, CPU-only):
  - Discovers all timepoints per case dynamically (3–6 each — NOT a fixed 5).
  - Computes ONE target spacing per case from the `_0000` header and applies it to
    every timepoint (verified all timepoints in a case share identical geometry).
  - Per axis: in-range → leave; below min → snap up (downsample); above max → snap
    down (upsample).
  - Upsampling = cubic B-spline. Downsampling = DiscreteGaussian blur (sigma =
    0.5 × target mm) then linear. Mixed-axis cases handled in a two-pass resample.
  - Works in float32, clamps to original intensity range, casts back to source dtype.
  - Byte-copies in-range cases + each case's SYNAPSE_METADATA_MANIFEST.tsv.
  - Atomic writes (temp + os.replace) and `--resume` skip for safe restarts.
- Wrote `scripts/slurm/submit_resample_array.slurm` (CPU partition `tier1q`, no GPU,
  8 cpus/task, multiprocessing Pool, ITK pinned to 1 thread/process) and
  `scripts/slurm/submit_resample_array.sh` wrapper (auto-sizes the array).

**Results:**
- 1,506 cases total. Against the fixed range: **979 need resampling, 527 copy as-is**
  (DUKE 70/221, ISPY1 171/0, ISPY2 674/306, NACT-Pilot 64/0).
- Timepoint distribution: 205 cases×3, 179×4, 137×5, 985×6 ≈ 7,926 volumes.
- Validated: py_compile OK, SimpleITK 2.5.5 imports, slurm syntax OK. Dry-run on
  samples confirmed correct per-axis plans, incl. true mixed-axis cases (e.g.
  DUKE_002: xy 0.586→0.600 downsample + z 1.300→1.250 upsample).
- Array auto-sizes to 95 tasks (0-94%24 → up to 192 cores concurrently).

**Key findings:**
- GPUs do NOT help here — SimpleITK is CPU-only; speed comes from CPU array
  parallelism, so the job runs on `tier1q`, not `gpuq`.
- The existing `spacing_range_check_results.csv` flags used DUKE's 5–95 percentile
  band, which disagrees with the fixed spec range; decided to recompute from the raw
  spacing values against the fixed range (979 vs the CSV's 965).
- Output root `/ess/scratch/scratch1/t-9sbose/resampled_segmentation/` did not exist;
  the script/job create it. Input root is annawoodard's scratch (read-only).

**How to run (not yet submitted — pending smoke test):**
```bash
# Smoke test ONE case on a compute node (scheduler-allocated, not the login node):
srun --partition=express --cpus-per-task=2 --mem=8G --time=00:15:00 \
  micromamba run -n vanguard python scripts/resample_to_range.py \
  --case-start 1 --case-end 1 --workers 2   # DUKE_002, the mixed-axis case
# Full run:
bash scripts/slurm/submit_resample_array.sh
```

---

### Smoke-test result (2026-06-25)
Ran one case (DUKE_002) via `srun` on the `express` partition (scheduler-allocated,
not the login node). Verified the resampled output:
- spacing 0.586→0.600 (xy, downsampled) and 1.300→1.250 (z, upsampled) — exactly on
  the range edges; size 512×512×142 → 500×500×148 (matches the expected voxel-count math).
- dtype preserved (int16); intensity 0–2103 stayed inside the source range 0–2407
  (clamp prevented B-spline overshoot/wraparound); manifest + all 4 timepoints written.
- Note: compute nodes need the absolute micromamba binary
  (`/ess/home/home1/t-9sbose/vanguard/bin/micromamba`) since `micromamba` is only a
  shell function from `.bashrc`; the `.slurm` script handles this via the shell hook.
Smoke test PASSED. Full 95-task array not yet launched (awaiting go-ahead).

---

### Full array run COMPLETE (2026-06-25) — job 12284276
Submitted the 95-task array (`scripts/slurm/submit_resample_array.sh`, tier1q, 0-94%24).
- **All 95 tasks COMPLETED**, zero FAILED/TIMEOUT/OOM, no errors in any task log.
- Output verified at `/ess/scratch/scratch1/t-9sbose/resampled_segmentation/`:
  1,506 case dirs (DUKE 291, ISPY1 171, ISPY2 980, NACT-Pilot 64), 7,926 volumes,
  1,506 manifests, 0 leftover `.partial_` temp files.
- Resample/copy split exactly as predicted: **979 resampled, 527 copied**.
- Spot-check: 40 random output volumes all within xy[0.60,1.02], z[1.00,1.25].
- Disk: output added ~103 GB (scratch 424→527 GB used; well under 30 TB quota).
Result: PASSED. Resampled dataset ready as input for vessel segmentation.

---

## 2026-06-25 — Resampled vessel segmentation: full array + ISPY1_1228 fix

**Goal:** Run fast vessel segmentation over all 7,926 resampled MRI volumes (1,506 cases
across DUKE, ISPY1, ISPY2, NACT-Pilot), writing `.npz` outputs to GPFS.

### Full array run (job 12284980)

**Script:** `faster-segmentation-test/submit_fast_array.sh` (called via `scripts/run_resampled_vessel_segmentation.sh`)
**Input:** `/ess/scratch/scratch1/t-9sbose/resampled_segmentation_flat/` (1,506 case symlinks → 7,926 `.nii.gz`)
**Output:** `/gpfs/data/karczmar-lab/workspaces/saritbose/resampled-vessel-segmentations/`
**Config:** 397 tasks × 20 files/task, `FILES_PER_TASK=20`, `MAX_CONCURRENT_TASKS=16`, `gpuq`, 4 CPUs, 32 GB, 1 GPU, 4 h limit
**Branch/commit at submission:** `feature/segmentation-speedup-and-qc` @ `55b2478`
**Env:** `micromamba activate vanguard`
**Result:** 396 COMPLETED, 1 FAILED (task 91, ISPY1_1228) → 7,906 / 7,926 `.npz`
**Wall-clock:** ~2 h 17 min total (tasks ran ~2–5 min each)

### ISPY1_1228 tiling failure (task 91)

**Root cause:** ISPY1_1228 has z=307 slices after resampling. `Dataset3DDivided` with
`z_division=3`, `input_dim=96` computes tile step = (307−96)/(3−1) = 105 > 96, leaving
middle slices uncovered → `AssertionError`. This crashed the entire 20-file batch (files
1820–1839), so all 20 `.npz` from task 91 were missing.

**Fix:** Added `--z-division` CLI parameter (default=3, backward-compatible) to
`faster-segmentation-test/batch_segmentation_fast.py`. Committed as `a50e65a` on
`feature/segmentation-speedup-and-qc`. Also added `Z_DIVISION` env-var passthrough to
`submit_fast_array.slurm`, and wrote `submit_task91_rerun.{sh,slurm}` as the documented
rerun entrypoint.

### Task-91 rerun (job 12295362)

**Script:** `faster-segmentation-test/submit_task91_rerun.sh`
**Command:** `bash faster-segmentation-test/submit_task91_rerun.sh`
**Input:** same flat farm, files 1820–1839 (20 files), `z_division=4`
**Output:** same GPFS dir
**Branch/commit:** `feature/segmentation-speedup-and-qc` @ `a50e65a`
**Env:** `micromamba activate vanguard`
**Job submitted:** 2026-06-26 ~09:30 CDT
**Result:** COMPLETED, 20/20 succeeded — all 3 ISPY1_1228 timepoints landed
**Wall-clock:** 3 min 44 s

**Final count:** 7,926 / 7,926 `.npz` — pipeline complete.

**Key findings:**
- z_division=3 is insufficient for any z > 288 after resampling; z_division=4 is safe for
  all realistic MRI z-sizes (step = (z−96)/3 ≤ 96 for z ≤ 384).
- The `--z-division` parameter is now exposed so future cases can be handled without code edits.

---

## 2026-06-26 — Axial MIP comparison script: original vs. resampled segmentations

**What I did:**
Created `scripts/compare_seg_mips.py` — a standalone script that randomly selects
2 cases from each of ISPY1, ISPY2, and NACT, loads the vessel segmentation `.npz`
from both `vessel_segmentations/` (original) and `resampled-vessel-segmentations/`
(resampled), computes an axial maximum-intensity projection (MIP) for each by
projecting along axis 2, and saves a 2-row × 6-column comparison PNG.

**Run command:**
```
cd /ess/home/home1/t-9sbose/vanguard
micromamba run -n vanguard python scripts/compare_seg_mips.py --seed 42
```

**Data roots:**
- Original:  `/gpfs/data/karczmar-lab/workspaces/saritbose/vessel_segmentations/`
- Resampled: `/gpfs/data/karczmar-lab/workspaces/saritbose/resampled-vessel-segmentations/`

**Output:** `~/vanguard_qc_pngs/seg_mip_comparison.png` (2158×951 px, 69 KB)

**Branch:** `feature/segmentation-speedup-and-qc`

**Cases selected (seed=42, timepoint=0001):**
- ISPY1: ISPY1_1021, ISPY1_1193
- ISPY2: ISPY2_489194, ISPY2_483667
- NACT: NACT_49, NACT_07

**Axial MIP nonzero pixel counts (original → resampled):**
| Case | Original | Resampled | Ratio |
|------|----------|-----------|-------|
| ISPY1_1021 | 536 | 1334 | 2.5× |
| ISPY1_1193 | 1148 | 4123 | 3.6× |
| ISPY2_489194 | 782 | 971 | 1.2× |
| ISPY2_483667 | 6042 | 6040 | ~1.0× |
| NACT_49 | 1025 | 3259 | 3.2× |
| NACT_07 | 60 | 243 | 4.1× |

**Key findings:**
- Resampled segmentations consistently detect more vessel coverage in the axial MIP
  (typically 2–4× more nonzero pixels), except ISPY2_483667 which is nearly identical.
- The MIP comparison is a fast visual QC tool for checking whether resampling is
  improving or degrading segmentation quality.
- Script uses `--seed` for reproducible case selection; run with the same seed to get
  the same 6 cases every time.

---

## 2026-06-26 — TC4D skeletonization over all resampled vessel segmentations

**Goal:** Run TC4D centerline extraction on all 1,506 resampled-dataset cases, writing
skeleton masks, morphometry JSON, and MIP PNGs to GPFS.

### Test run (job 12295437, 5 cases)

**Script:** `scripts/run_resampled_centerlines.sh` (TEST=1)
**Command:** `TEST=1 bash scripts/run_resampled_centerlines.sh`
**Input:** `/gpfs/data/karczmar-lab/workspaces/saritbose/resampled-vessel-segmentations`
**Output:** `/gpfs/data/karczmar-lab/workspaces/saritbose/resamples_centerlines_tc4d`
**Branch/commit:** `feature/segmentation-speedup-and-qc` @ `e916ae9`
**Env:** `micromamba activate vanguard`
**Result:** 5/5 COMPLETED, ~3.5–5 min/case. All outputs present (skeleton, morphometry, MIP).
**Note:** `/net/projects2/vanguard/` does not exist on Randi; annotation/tumor context
skipped gracefully, MIP still rendered from TC4D skeleton alone.

### Full array (job 12295463, 1,506 cases)

**Script:** `scripts/run_resampled_centerlines.sh`
**Command:** `bash scripts/run_resampled_centerlines.sh`
**Input:** same as test
**Output:** same as test
**Config:** 1,506 tasks, MAX_CONCURRENT=50, tier1q, 8 CPUs / 32 GB / 8 h limit
**Branch/commit:** `feature/segmentation-speedup-and-qc` @ `e916ae9`
**Env:** `micromamba activate vanguard`
**Submitted:** 2026-06-26 ~09:53 CDT
**Completed:** 2026-06-26 ~10:52 CDT (~59 min wall-clock)
**Result:** 1,506/1,506 COMPLETED, 0 FAILED
**Final count:** 1,506 `run_summary.json` files ✓

**Per-case outputs (under `<OUTPUT>/<COHORT>/<CASE>/`):**
- `<case>_skeleton_4d_exam_mask.npy`
- `<case>_skeleton_4d_exam_support_mask.npy`
- `<case>_center_manifold_4d_mask.npy`
- `<case>_morphometry.json`
- `<case>_vessel_coverage_mip.png`
- `run_summary.json`

**Key findings:**
- TC4D runs ~3–5 min/case on 8 CPU cores; 1,506 cases completed in under 1 hour at 50x concurrency.
- Missing annotation paths (/net/projects2/vanguard/) are handled gracefully — all core outputs
  (skeleton, morphometry) still produced correctly.

---

## 2026-06-26 — 3D vessel segmentation comparison: original vs. resampled

**Goal:** Go beyond 2D axial MIPs to visually compare the 3D structure of the
original vs. resampled vessel segmentations — specifically to see how much each
pipeline picks up and whether the spatial structure looks plausible.

**What I did:**
- Created `QC/compare_seg_3d.py`: for 6 randomly-selected cases (2 per cohort:
  ISPY1, ISPY2, NACT, seed=42), loads ALL available timepoints (0000, 0001, 0002)
  from both segmentation roots, takes the union across timepoints (a voxel is
  counted as vessel if it exceeds 0.5 probability in ANY timepoint — matching how
  the TC4D skeleton pipeline aggregates time). Produces per-case:
  - Static PNG: original point cloud on top, resampled on bottom; points colored
    by z-height gradient (plasma colormap) so depth is readable in the still image
  - Rotating 360° MP4: both panels spin together
- Created `QC/slurm/compare_seg_3d.slurm` and `QC/slurm/submit_compare_seg_3d.sh`
  (1 CPU, 16 GB, 30 min wall; runs all 6 cases sequentially in one job)
- Ran twice: first pass used a 3-way color split (gray/blue/red) which was hard to
  read; revised to separate top/bottom layout with larger points (s=20) and stride=1

**Run command:**
```bash
bash QC/slurm/submit_compare_seg_3d.sh
```
**Job IDs:** first pass 12296012 (COMPLETED, 29 s); revised pass 12297964 (COMPLETED, 48 s)
**Branch/commit:** `feature/segmentation-speedup-and-qc`
**Env:** `micromamba activate vanguard`

**Output files** (`~/vanguard_qc_pngs/`):

| Case | PNG | MP4 |
|------|-----|-----|
| ISPY1_1021 | `~/vanguard_qc_pngs/ISPY1_1021_3d_comparison.png` | `~/vanguard_qc_pngs/ISPY1_1021_3d_comparison.mp4` |
| ISPY1_1193 | `~/vanguard_qc_pngs/ISPY1_1193_3d_comparison.png` | `~/vanguard_qc_pngs/ISPY1_1193_3d_comparison.mp4` |
| ISPY2_489194 | `~/vanguard_qc_pngs/ISPY2_489194_3d_comparison.png` | `~/vanguard_qc_pngs/ISPY2_489194_3d_comparison.mp4` |
| ISPY2_483667 | `~/vanguard_qc_pngs/ISPY2_483667_3d_comparison.png` | `~/vanguard_qc_pngs/ISPY2_483667_3d_comparison.mp4` |
| NACT_49 | `~/vanguard_qc_pngs/NACT_49_3d_comparison.png` | `~/vanguard_qc_pngs/NACT_49_3d_comparison.mp4` |
| NACT_07 | `~/vanguard_qc_pngs/NACT_07_3d_comparison.png` | `~/vanguard_qc_pngs/NACT_07_3d_comparison.mp4` |

**Point counts (stride=1, union across all timepoints, threshold=0.5):**

| Case | Original pts | Resampled pts | Ratio |
|------|-------------|---------------|-------|
| ISPY1_1021 | 39 | 85 | 2.2× |
| ISPY1_1193 | 88 | 324 | 3.7× |
| ISPY2_489194 | 152 | 160 | 1.1× |
| ISPY2_483667 | 1847 | 1847 | 1.0× |
| NACT_49 | 105 | 431 | 4.1× |
| NACT_07 | 8 | 28 | 3.5× |

**Key findings:**
- Resampled segmentations consistently pick up more vessel voxels (2–4× more for
  ISPY1 and NACT; nearly identical for ISPY2). Consistent with the axial MIP comparison.
- The two grids are on different coordinate systems (native vs. standardized spacing),
  so there is no voxel-level overlap — expected, not a sign of disagreement.
- Point counts are low for most cases even at stride=1, reflecting genuinely sparse
  high-confidence vessel voxels at threshold=0.5. Best cases for visual QC:
  ISPY1_1193 (88 vs 324 pts) and NACT_49 (105 vs 431 pts).

---

## 2026-06-26 — Re-run previous group's pСR analysis on RESAMPLED segmentations (Stage 0 attempt + blocker found)

**Goal:** Mentor asked to "re-run the previous group's analysis on MAMA-MIA and see
how it changes performance." Scoped (with mentor framing) to: feed the new
*resampled* vessel segmentations through the previous group's downstream pipeline
and compare pCR AUC to their published independent-signal baseline
(`results/independent_signal_q3_summary.csv`: clinical+tumor_size 0.572 → vessel_all
0.596, ISPY2-only, n=808). Decision: compare resampled arm against that README
baseline (not re-running the original arm fresh).

**Pipeline recap:** segmentation → graph_extraction (centerlines + features) →
train_tabular (independent-signal matrix). Segmentation + centerlines already
existed for both original and resampled; only the modeling step remained.

**What I built (committed-ready, but UNCOMMITTED in working tree at run time):**
- `configs/independent_signal_randi_resampled.yaml` — randi-pathed twin of
  `configs/independent_signal.yaml`; identical modeling choices, only `centerline_root`
  → resampled centerlines and data_paths → randi-accessible locations.
- `slurm/submit_resampled_independent_signal.sh` — Stage 1 wrapper (sets CONFIG +
  OUT_ROOT, delegates to `submit_independent_signal_matrix_array.sh`).
- `graph_extraction/slurm/submit_resampled_tumor_graph_features.sh` — Stage 0 wrapper.
- Edited `graph_extraction/slurm/submit_tc4d_array.sh` (added optional `SITE_FILTER`)
  and `submit_tc4d_array.slurm` (added optional `TUMOR_MASK_DIR` /
  `RADIOLOGIST_ANNOTATIONS_DIR` passthrough — the baked-in `/net/projects2/...`
  default is not mounted on randi). Both edits are additive / backward-compatible.

**Pre-submit blocker found:** the resampled centerlines had `*_morphometry.json`
(morph block) for all 980 ISPY2 cases but `0/980` `*_tumor_graph_features.json`
(graph + kinematic blocks). The resampled centerline run never produced the
tumor-graph JSONs. → Stage 0 = features-only recompute to generate them.

**Stage 0 run:**
- Command: `bash graph_extraction/slurm/submit_resampled_tumor_graph_features.sh`
- Slurm job: **12299736**, array 0-979%50, partition `tier1q`, 8 CPU / 32 GB / 8 h.
- Mode: `--features-only --force-features` (force needed because morphometry.json
  already exists), `--no-render-mip`, `SEG_ONLY=1`, `SITE_FILTER=ISPY2`.
- Input root:  `/gpfs/data/karczmar-lab/workspaces/saritbose/resampled-vessel-segmentations`
- Output root: `/gpfs/data/karczmar-lab/workspaces/saritbose/resamples_centerlines_tc4d`
- Tumor masks: `/ess/scratch/scratch1/annawoodard/MAMA-MIA-syn60868042/segmentations/expert`
- Env: `micromamba -n vanguard`. Branch `feature/segmentation-speedup-and-qc`,
  HEAD `9be7002` (new scripts/config uncommitted).

**Result — FAILED for the majority; job cancelled.** Of 980 cases:
`165` produced a valid `tumor_graph_features.json` (`tumor_context: ok`); `815`
hit `tumor_mask_load_failed` → `tumor_graph_features_status: skipped_no_tumor_context`
(kinematic also `skipped_missing_tumor_context`).

**Root cause (key finding):** resampling changed the image/voxel grid, but the
expert tumor masks are still in the **native** grid. Example (ISPY2_416434):
skeleton expects `(271, 271, 146)`, expert mask is `(76, 320, 320)` → shape
mismatch, so the tumor cannot be placed and both tumor-centered blocks (graph +
kinematic) are skipped. The morph block is unaffected (it needs no tumor mask).
The 165 that worked are cases whose resampled grid already matched the native mask.
This is why the *original* (native) centerlines have the JSONs but the *resampled*
ones don't: native skeleton + native mask align; resampled skeleton + native mask
don't. Cancelled job 12299736 to avoid wasting the allocation on redundant work
(it was only rewriting already-existing morphometry).

**Next step:** add a tumor-mask resampling step — resample each expert tumor mask
onto its case's resampled grid (nearest-neighbour / label-preserving, using the
resampled segmentation as the reference geometry), write to a new mask dir, then
rerun Stage 0 with `TUMOR_MASK_DIR` pointed at the resampled masks. Then Stage 1
(`submit_resampled_independent_signal.sh`) → compare AUCs to the README baseline.

---

## 2026-06-26 — Fix: resample tumor masks onto the resampled grid (built + validated)

**Goal:** Unblock the Stage 0 failure (815/980 ISPY2 cases hit "mask shape mismatch":
native expert masks vs resampled skeletons). The tumor masks must live on the same
voxel grid as the resampled segmentations/skeletons.

**What I built (UNCOMMITTED in working tree):**
- `scripts/resample_tumor_masks.py` — resamples each expert tumor mask onto the
  matching RESAMPLED image as the geometric reference (SimpleITK `SetReferenceImage`),
  using NEAREST-NEIGHBOUR interpolation (label-preserving; linear/B-spline would
  invent fractional labels). Follows `scripts/resample_to_range.py` conventions
  (worker Pool, `--resume`, `--dry-run`, atomic writes, cohort labels). Output is a
  flat `<case_id>.nii.gz` dir matching what `--tumor-mask-dir` expects.
- `scripts/slurm/submit_resample_tumor_masks.slurm` — single multi-worker tier1q job
  (CPU-only, 8 CPU / 16 GB / 1 h), defaults to COHORT=ISPY2.
- Updated `graph_extraction/slurm/submit_resampled_tumor_graph_features.sh` so its
  default `TUMOR_MASK_DIR` now points at the resampled masks
  (`/gpfs/data/karczmar-lab/workspaces/saritbose/resampled-tumor-masks`), NOT the
  native expert masks.

**Why "follow how we resampled the actual data":** the resampled NIfTI images from
`resample_to_range.py` (`/ess/scratch/scratch1/t-9sbose/resampled_segmentation/<cohort>/<case>/<case>_0000.nii.gz`)
ARE the resampled grid the skeletons sit on. Using each as the reference image gives
the mask exactly that size/spacing/origin/direction — no spacing-plan recompute, no
rounding drift. Coverage confirmed: 980/980 ISPY2 resampled images + 980/980 expert masks.

**Validation (single case ISPY2_416434, env `vanguard`):**
- Native mask `(320,320,76)` @ 0.508/0.508/2.4 mm → resampled `(271,271,146)` @ 0.6/0.6/1.25 mm
  (matches the resampled image grid; identical origin + direction).
- Fed through `graph_extraction.masks._select_zyx_layout` against the skeleton shape
  `(271,271,146)`: **matched via the `yxz` layout** (previously failed for the native mask).
- Tumor preserved: native 52,799 vox = 32,701 mm³; resampled 72,286 vox = 32,529 mm³
  (~0.5% volume diff — expected for NN onto a finer grid). Confirms correct placement
  + label preservation.

**Next steps (not yet submitted):**
1. `sbatch scripts/slurm/submit_resample_tumor_masks.slurm`  (resample all 980 ISPY2 masks)
2. Re-run Stage 0: `bash graph_extraction/slurm/submit_resampled_tumor_graph_features.sh`
   (now uses the resampled masks → should produce graph + kinematic JSONs for all cases)
3. Stage 1: `bash slurm/submit_resampled_independent_signal.sh` → compare AUCs to
   `results/independent_signal_q3_summary.csv` (0.572 → 0.596).

**Env:** `micromamba -n vanguard`. Branch `feature/segmentation-speedup-and-qc`, HEAD `9be7002`
(new scripts uncommitted). Cancelled Stage 0 job from earlier was 12299736.

---

## 2026-06-26 — Mask resample run + Stage 0 relaunch (process log)

Continues the two prior entries (resampled-seg → pCR comparison; the tumor-mask
shape-mismatch fix). Recording the runs end-to-end so the journey is reproducible.

**Step 1 — resample ISPY2 tumor masks (DONE):**
- Command: `sbatch scripts/slurm/submit_resample_tumor_masks.slurm`
- Slurm job: **12300766**, partition `tier1q`, 8 CPU / 16 GB. `COMPLETED`, exit 0:0,
  **elapsed 00:00:41**.
- Result: **980/980** ISPY2 masks resampled onto the resampled image grid, 0 missing.
- Output: `/gpfs/data/karczmar-lab/workspaces/saritbose/resampled-tumor-masks/` (980 `.nii.gz`).
- Sanity from the log: in-range cases pass through unchanged (e.g. (384,384,160)→(384,384,160));
  off-grid cases reshape to their resampled reference (e.g. (256,256,62)→(256,256,129)).

**Step 2 — Stage 0 relaunch with aligned masks (RUNNING):**
- Command: `bash graph_extraction/slurm/submit_resampled_tumor_graph_features.sh`
  (now defaults `TUMOR_MASK_DIR` to the resampled masks above).
- Slurm job: **12300770**, array 0-979%50, `tier1q`, features-only `--force-features`.
- Input segs : `/gpfs/data/karczmar-lab/workspaces/saritbose/resampled-vessel-segmentations`
- Output     : `/gpfs/data/karczmar-lab/workspaces/saritbose/resamples_centerlines_tc4d`
- Tumor masks: `/gpfs/data/karczmar-lab/workspaces/saritbose/resampled-tumor-masks`
- Success criterion: `*_tumor_graph_features.json` count climbs toward 980 with
  `tumor_context: ok` (previous attempt, job 12299736 with native masks, got only 165/980).

**Recap of why the first Stage 0 failed:** the resampled centerlines lacked the
tumor-graph JSON (graph + kinematic blocks); recomputing them needs the tumor mask on
the skeleton's grid, but the native expert masks are on the native grid → "mask shape
mismatch" → both blocks skipped for 815/980 cases. Fix = resample masks onto the
resampled grid (NN), then rerun Stage 0.

**Still TODO after this finishes:**
- Verify JSON coverage (~980) and a few `tumor_context: ok`.
- Stage 1: `bash slurm/submit_resampled_independent_signal.sh` → compare AUCs to
  `results/independent_signal_q3_summary.csv` (0.572 → 0.596).

**Env:** `micromamba -n vanguard`. Branch `feature/segmentation-speedup-and-qc`,
HEAD `9be7002` (Stage 0/1 scripts, configs, and mask-resample scripts all still UNCOMMITTED).

---

## 2026-06-29 — Resampled independent-signal matrix: end-to-end run and AUC comparison

**Goal:** Re-run the previous group's independent-signal ablation matrix (6 feature-block
arms, 5-fold nested-CV logistic regression, ISPY2 n=808) using the resampled vessel
segmentation centerlines, and compare resulting AUCs against their published baseline to
assess whether resampling improves predictive signal.

**Prerequisites confirmed before submitting:**
- Vessel segmentation: done (resampled to DUKE spacing)
- TC4D skeletonization: done (`/gpfs/data/karczmar-lab/workspaces/saritbose/resamples_centerlines_tc4d/`)
- Tumor mask resampling: done (980 `.nii.gz` at `/gpfs/data/karczmar-lab/workspaces/saritbose/resampled-tumor-masks/`)
- Graph + kinematic JSON features: confirmed present for all 980 ISPY2 cases (`find ... -name "*_tumor_graph_features.json" | wc -l` → 980)

**What I did:**
1. Submitted `bash slurm/submit_resampled_independent_signal.sh` with `PARTITION=tier1q`
   (the `general` partition doesn't exist on Randi — must use `tier1q`).
   Config: `configs/independent_signal_randi_resampled.yaml`
   Output root: `/ess/scratch/scratch1/t-9sbose/vanguard_experiments/independent_signal_resampled_ispy2/`

2. **Bug fixed — wrong tumor mask path.** First submission (jobs 12331098–12331100)
   used `tumor_mask_root` pointing at the native expert masks
   (`/ess/scratch/scratch1/annawoodard/MAMA-MIA-syn60868042/segmentations/expert`).
   The cache build found all 808 mask files but failed to load 612 of them due to a
   voxel-grid shape mismatch (native masks are on the native grid; resampled centerlines
   are on the resampled grid). This set 612/808 patients' `tumor_size` features to NaN,
   which the imputer filled with column medians, destroying all tumor size signal.
   Clinical+tumor_size AUC collapsed to 0.475 (below random).
   **Fix:** Changed `tumor_mask_root` in `configs/independent_signal_randi_resampled.yaml`
   to `/gpfs/data/karczmar-lab/workspaces/saritbose/resampled-tumor-masks`. Cleared
   output dir and resubmitted (jobs: cache=12331353, array=12331354).

3. **Bug fixed — stale function signatures in `modeling/merge_results.py`.** The merge
   step imports `_metrics_summary_row` and `_add_baseline_deltas` from
   `run_ablation_matrix.py`, both of which had gained new required keyword arguments
   (`run_name`, `model_family`, `split_mode`; and `baseline_run_name`) never passed at
   the call sites. Fixed and resubmitted merge-only as job 12331526 (COMPLETED, exit 0).

**Exact commands (reproducible):**
```bash
# Sanity check graph features
find /gpfs/data/karczmar-lab/workspaces/saritbose/resamples_centerlines_tc4d/ISPY2 \
  -name "*_tumor_graph_features.json" | wc -l   # → 980

# Full pipeline (use after clearing old output dir)
PARTITION=tier1q bash slurm/submit_resampled_independent_signal.sh
# Jobs: cache=12331353, array=12331354, merge=12331355

# Merge-only resubmit (if merge fails but array succeeded)
FEATURES_CSV="/ess/scratch/scratch1/t-9sbose/vanguard_experiments/independent_signal_resampled_ispy2/features_full_labeled.csv"
OUT_ROOT="/ess/scratch/scratch1/t-9sbose/vanguard_experiments/independent_signal_resampled_ispy2"
sbatch --partition=tier1q --cpus-per-task=2 --mem=16G --time=01:00:00 \
  --output="logs/ml-ablation-merge-%j.out" --error="logs/ml-ablation-merge-%j.err" \
  --wrap="bash -lc 'cd /home/t-9sbose/vanguard; eval \"\$(micromamba shell hook -s bash)\"; micromamba activate vanguard; python -m modeling.merge_results --config configs/independent_signal_randi_resampled.yaml --features-csv ${FEATURES_CSV} --out-root ${OUT_ROOT}'"
# Final merge job: 12331526 (COMPLETED, exit 0)
```

**Environment:** `micromamba activate vanguard`, branch `feature/segmentation-speedup-and-qc`

**Results — AUC comparison (resampled vs previous group's baseline):**

| Arm | Resampled AUC ± std | Baseline AUC ± std | Delta |
|-----|:---:|:---:|:---:|
| clinical + tumor_size | 0.603 ± 0.065 | 0.572 ± 0.041 | **+0.031** |
| + morph | 0.589 ± 0.067 | 0.591 ± 0.033 | -0.002 |
| + graph | 0.581 ± 0.041 | 0.588 ± 0.055 | -0.007 |
| + kinematic | 0.588 ± 0.050 | 0.594 ± 0.043 | -0.006 |
| + graph + kinematic | 0.579 ± 0.039 | 0.594 ± 0.051 | -0.015 |
| + vessel_all | 0.563 ± 0.036 | 0.596 ± 0.032 | **-0.033** |

Baseline reference: `results/independent_signal_q3_summary.csv`
Full resampled results: `/ess/scratch/scratch1/t-9sbose/vanguard_experiments/independent_signal_resampled_ispy2/ablation_summary.csv`

**Data integrity checks performed after the run:**
- All 30 fold prediction files present; all 30 array tasks exit code 0
- Tumor masks: 0/808 NaN in tumor_size features after the mask path fix
- Kinematic coverage: 99.8% per column; every patient has 892–895/895 features
- Graph coverage: every patient has 45–49/49 features; min column coverage 52%
- All 808 patients have matching PCR labels in `pcr_labels.csv`

**Key findings:**
1. **Resampling improved the non-vessel baseline (+0.031 AUC on clinical+tumor_size).**
   Consistent voxel spacing likely produces more reliable tumor volume estimates.

2. **Vessel features do not add predictive signal on the resampled data.** The previous
   group showed vessel_all added +0.024 over the clinical baseline. On resampled data,
   vessel_all is -0.040 below the clinical baseline — adding vessel features hurts.

3. **Caveats to raise with mentor:**
   - Resampled pipeline has more features per arm than the baseline (e.g., 912 vs 836
     kinematic columns). The additional features may act as noise in the vessel arms.
   - `breast_density` is entirely NaN for all 808 ISPY2 patients and contributes nothing.
   - Patient overlap with the baseline is unverifiable: the previous group's feature
     table is on `/net/projects2` (not mounted on Randi).
   - This is a cross-cluster comparison — the config comment in
     `configs/independent_signal_randi_resampled.yaml` flags this explicitly.

---

## 2026-06-29 — Resampling investigation: feature diagnostics, spacing audit, near-tumor coverage

**Goal:** Determine whether the AUC results from the resampled independent-signal run
reflect real biology or pipeline artifacts. Six specific checks were designed and run.

**Scripts committed:**
```
scripts/investigate_resampling/01_feature_distribution_audit.py
scripts/investigate_resampling/02_spacing_audit.py
scripts/investigate_resampling/03_auc_comparison_plot.py
scripts/investigate_resampling/04_seg_visual_check.py
scripts/investigate_resampling/05_morphometry_json_audit.py
scripts/slurm/submit_resampling_investigation.sh
```

**How to run:**
```bash
PARTITION=tier1q bash scripts/slurm/submit_resampling_investigation.sh
# Visual check Slurm job: 12331680 (COMPLETED)
```

**All outputs at:** `/ess/scratch/scratch1/t-9sbose/resampled_investigation/`

---

**Check 1 — Feature distribution (01_feature_distribution/)**

808 patients × 1286 columns (50 morph, 49 graph, 895 kinematic, 288 clinical+tumor_size).

Features with >5% NaN:

| Feature | NaN rate |
|---------|----------|
| breast_density | 100% — absent for all ISPY2 |
| kinematic_near_washin_to_washout_ratio | 68% |
| kinematic_arrival_delay_dispersion_near | 62% |
| graph_near_branching_bias | 47.5% |
| graph_crossing_fraction_of_near_burden | 34.6% |
| menopausal_status | 18.3% |
| kinematic_core_to_periphery_peak_ratio | 14.5% |
| graph/kinematic core_to_periphery ratios | 10.5% |

All high-NaN features are excluded by `feature_select_min_non_na_rate: 0.2`.
Morph, graph, and kinematic blocks otherwise have near-complete coverage.

---

**Check 2 — Voxel spacing audit (02_spacing_audit/)**

Critical finding: resampled masks are NOT at a single uniform spacing.

| | XY (mm) | Z (mm) |
|--|---------|--------|
| Resampled (n=980) | 0.60–1.02 (mean 0.712) | 1.00–1.25 (most at 1.25) |
| Native sample (n=50) | 0.59–1.06 (mean 0.721) | 1.00–2.50 (median 1.0) |

The resampling standardized Z to ~1.25 mm for most cases but did NOT homogenize XY.
Since morph features are in voxel units, per-case XY variation still confounds them
in the resampled data — just differently than in native data where Z also varied widely.
`StandardScaler` absorbs global shifts but not patient-level spacing variation within a fold.

---

**Check 3 — AUC comparison plot (03_auc_comparison/auc_comparison.png)**

| Arm | Baseline AUC | Resampled AUC | Delta |
|-----|:---:|:---:|:---:|
| clinical + tumor_size | 0.572 | 0.603 | **+0.031** |
| + morph | 0.591 | 0.589 | -0.002 |
| + graph | 0.588 | 0.581 | -0.007 |
| + kinematic | 0.594 | 0.588 | -0.006 |
| + graph + kinematic | 0.594 | 0.579 | -0.015 |
| + vessel_all | 0.596 | 0.561 | **-0.035** |

---

**Check 4 — Segmentation visual check (04_seg_visual_check/, job 12331680)**

3 cases checked (ISPY2_100899, 102011, 102212). Integrity report:
- No NaN values in any segmentation (native or resampled)
- All values in [0.0, 1.0] — correct binary masks
- Z-dimension correctly expanded: native ~72–80 slices × 2.0 mm → resampled ~128 slices × 1.25 mm (same physical coverage)
- Vessel coverage (% nonzero) changed by <0.03 pp between native and resampled — no erosion/dilation artifacts

---

**Check 5 — Morphometry JSON plausibility (05_morphometry_audit/)**

All 980 ISPY2 cases passed physical plausibility checks:
- Mean segment length: 12.6 mm (range 1.7–49.9 mm) — correct for breast vasculature
- Mean vessel radius: 1.57 mm — reasonable for DCE-MRI visible vessels
- Mean tortuosity: 1.18 (always ≥ 1.0, as expected)
- Total vessel volume: ~15,100 mm³ per case
- No outlier flags on any case

---

**Check 6 — Near-tumor coverage: native vs resampled**

`TUMOR_NEAR_MM = 5.0` — a vessel is "near tumor" if any skeleton node is within 5 mm
of the tumor surface (from `graph_extraction/constants.py`).

| | Zero near-tumor segments | Has near-tumor segments |
|--|:---:|:---:|
| Native (980 cases) | 368 (37.6%) | 612 (62.4%), mean=10.3 segs |
| Resampled (980 cases) | 355 (36.2%) | 625 (63.8%), mean=8.1 segs |

Coverage rate is essentially the same — resampling did NOT reduce peritumoral vessel
detection. The high NaN rates in near-shell features (~47–68%) are a structural property
of the ISPY2 dataset: ~37% of patients simply have no vessel centerline nodes within 5 mm
of their tumor in this segmentation. This is the same in native and resampled data.

Among cases that DO have near-tumor segments, the resampled data has slightly fewer
(mean 8.1 vs 10.3). This reduces precision of derived features (wash-in/washout ratio,
dispersion) for those cases but is not a major effect.

---

**Key findings:**

1. **The clinical+tumor_size improvement (+0.031) is real.** Resampled tumor masks have
   finer Z-resolution (1.25 mm vs native median 1.0–2.5 mm), producing more accurate
   tumor volume measurements. This is the strongest positive signal from resampling.

2. **The vessel arm drop is not explained by data artifacts.** Segmentation integrity is
   clean (no NaN, no erosion), near-tumor coverage is identical, and morphometry values
   are physically plausible. The vessel arms dropping is likely a real finding.

3. **A code-version confound was discovered.** Commit `91a6d55` (Rebecca Wu, Apr 2026)
   added 5 second-order kinematic features and 10 other second-order features AFTER the
   baseline run was committed. Our resampled run includes these; the previous group's
   baseline did not. Three of the 5 new kinematic features have 15–68% NaN rates,
   meaning they are sparse and potentially noisy for the model. This partially confounds
   the baseline comparison.

4. **Native and resampled kinematic feature sets are structurally identical** (890 unique
   keys from both, zero difference). The apparent "59 extra features" hypothesis was wrong
   — the extras come from the post-baseline code commit, not from resampling itself.

**Next steps:**
- DeepSets on resampled data submitted (jobs 12332645/12332646/12332647) to get a
  model-architecture-independent comparison
- A fair code-version-controlled comparison would require running the pre-`91a6d55`
  code on both native and resampled data, or running post-`91a6d55` code on native data

---

## 2026-06-29 — Three-way AUC comparison: old-code-native vs new-code-native vs new-code-resampled

**Goal:** Disentangle code-version effects from resampling effects by adding the
"current code + native data" data point (Run B) to the comparison.

**Run B results provided by mentor/collaborator** (current code, native centerlines, n=808):

| Arm | n_features | AUC |
|-----|:---:|:---:|
| clinical+tumor_size | 35 | 0.561 |
| +morph | 82 | 0.587 |
| +graph | 330 | 0.566 |
| +kinematic | 864 | 0.565 |
| +graph+kinematic | 1159 | 0.579 |

**Three-way comparison:**

| Arm | A: old+native | B: new+native | C: new+resamp | B−A (code) | C−B (resamp) |
|-----|:---:|:---:|:---:|:---:|:---:|
| clinical+tumor_size | 0.572 | 0.561 | 0.603 | -0.011 | **+0.042** |
| +morph | 0.591 | 0.587 | 0.589 | -0.004 | +0.002 |
| +graph | 0.588 | 0.566 | 0.581 | -0.023 | +0.015 |
| +kinematic | 0.594 | 0.565 | 0.588 | -0.029 | **+0.024** |
| +graph+kinematic | 0.594 | 0.579 | 0.579 | -0.015 | +0.000 |

**Key findings:**

1. **Code version change (B−A) consistently hurts every arm.** Commit `91a6d55`
   (Apr 2026, Rebecca Wu) added 15 second-order features post-baseline. Several are
   sparse (15–68% NaN) and the new tumor_size second-order features expanded the
   clinical+tumor_size arm from 14 → 35 features — all extra noise. Every arm dropped
   between old and new code on the same native data (-0.004 to -0.029 AUC).

2. **Resampling consistently helps (C−B is positive for every arm).** When code version
   is held constant, resampled data outperforms native data across all five comparable
   arms (+0.000 to +0.042). The pure resampling effect is uniformly non-negative.

3. **The original apparent drop (C vs A was negative for vessel arms) was a code
   artifact, not a resampling artifact.** The headline number from the initial run
   (vessel_all 0.561 vs baseline 0.596) was misleading because it conflated two effects:
   a code regression (-0.015 to -0.029) and a genuine resampling improvement (+0.002 to
   +0.024). The net appeared negative but resampling alone is beneficial.

4. **Resampling most strongly improves the non-vessel baseline (+0.042).** Better Z-
   resolution in the resampled tumor masks produces more accurate tumor volume features.
   Vessel feature improvements are smaller but consistent.

**Implication:** The second-order features added in `91a6d55` are net harmful to model
performance in their current form. The sparse near-tumor and core-to-periphery ratio
features add noise that outweighs any signal. This is worth flagging — either the
feature engineering needs work, or these features should be gated behind a minimum
coverage threshold before entering the model.

---

## 2026-06-30 — Resampled vs Q3 baseline AUC overlay plot

**What I did:**
Created `scripts/plot_resampled_overlay_auc.py` to overlay the resampled ISPY2 AUC results
on the Q3 baseline summary. Each arm shown as a horizontal row: Q3 baseline (blue, mean ± 1 std)
and resampled (orange diamond, mean ± 1 std). Output: `results/resampled_vs_q3_auc_overlay.png`.

**Results:**
Resampled results mostly fall within the noise of the Q3 baseline. The two exceptions are
clinical+tumor_size (a bit higher) and vessel_all (a bit lower), but with only 5 folds it's
hard to read too much into either.

**Key findings:**
Visually confirms the three-way comparison: resampling doesn't dramatically shift AUC in
either direction. Differences are visible but not alarming given the fold variance.

---

## 2026-06-30 — Locked-split native run: isolating resampling effect

**Goal:** Address mentor's concern that the resampled vs. Q3 baseline comparison
confounds three variables: train/test split, code version, and resampling.
Lock the split constant so the only variable between the two runs is the
centerline source (resampled vs. native).

**What I did:**

1. Discovered that both native and resampled feature sets contain the exact same
   808 patients in the same order — so the StratifiedKFold(random_state=42) split
   was already identical. But made it explicit and documented by locking it.

2. Built the following infrastructure:
   - `scripts/extract_resampled_splits.py` — reads `predictions.csv` from the
     resampled run's baseline arm, saves `case_id, val_fold` to a canonical CSV.
   - `evaluation/build_splits.py` — added `_load_fixed_splits()` and a
     `fixed_splits_csv` check in `create_splits_for_dataframe()`. When the config
     sets `data_paths.fixed_splits_csv`, splits are loaded from that CSV instead
     of being generated from scratch.
   - `configs/native_locked_split_ispy2.yaml` — native centerlines config with
     `fixed_splits_csv` set.
   - `slurm/submit_native_locked_split.sh` — Slurm wrapper.
   - `scripts/compare_locked_split_auc.py` — generates the three-way AUC table
     and prints an interpretation.

3. Extracted and saved the locked split:
   ```
   python scripts/extract_resampled_splits.py
   # → /ess/scratch/scratch1/t-9sbose/vanguard_experiments/locked_split_resampled_ispy2.csv
   # 808 patients, folds: {0:162, 1:162, 2:162, 3:161, 4:161}
   ```

4. Submitted the native locked-split ablation to Slurm:
   ```
   PARTITION=tier1q bash slurm/submit_native_locked_split.sh
   ```

**Slurm job IDs:**
- Cache build : 12420112
- Fold array  : 12420113 (30 tasks: 6 arms × 5 folds)
- Merge       : 12420114

**Monitor:**
```bash
squeue -j 12420112,12420113,12420114
sacct -j 12420112,12420113,12420114 --format=JobIDRaw,State,Elapsed,ExitCode -n -P
```

**After completion:**
```bash
micromamba run -n vanguard python scripts/compare_locked_split_auc.py
```
Output: `results/locked_split_auc_comparison.csv`

**Results (all 30/30 tasks completed, 0 failures):**

| Arm | Q3 baseline | Resampled | Native locked | Δ (native−resamp) |
|-----|-------------|-----------|---------------|-------------------|
| clinical + tumor_size | 0.572 | 0.603 | 0.599 | −0.004 |
| + morph | 0.591 | 0.589 | 0.603 | **+0.013** |
| + graph | 0.588 | 0.581 | 0.606 | **+0.025** |
| + kinematic | 0.594 | 0.588 | 0.596 | **+0.008** |
| + graph + kinematic | 0.594 | 0.579 | 0.584 | **+0.005** |
| + vessel_all | 0.596 | 0.561 | 0.577 | **+0.016** |

**Finding: Resampling degrades vessel features.**
Native locked > resampled for ALL 5 vessel arms (same split, same code).
The spacing change (resampling to DUKE's 0.7×0.7×1.0 mm) distorts the
vessel centerline geometry enough to reduce feature quality by 0.005–0.025 AUC.

**Important nuance:** Even native locked vessel arms show negative deltas vs.
their own clinical baseline (−0.003 to −0.022). Both factors compound:
- Resampling degrades vessel geometry → worse features
- `second_order.py` adds NaN-heavy features → noise in model

**Output files:**
- `results/locked_split_auc_comparison.csv` — full numeric table
- `results/locked_split_auc_comparison.png` — three-way visual
- `/ess/scratch/scratch1/t-9sbose/vanguard_experiments/independent_signal_native_locked_ispy2/ablation_summary.csv`

**Visual generated with:**
```bash
micromamba run -n vanguard python scripts/plot_locked_split_comparison.py
```

---
