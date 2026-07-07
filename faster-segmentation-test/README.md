# faster-segmentation-test

Isolated workspace for a faster drop-in replacement of the vessel-segmentation
inference pipeline (`segmentation/batch_segmentation.py` →
`vanguard-blood-vessel-segmentation/predict.py`). **Nothing here modifies the
existing pipeline or the live run.** Outputs go to a separate directory.

## Why
The production pipeline takes ~3 days for the full cohort. The dominant cost is
the **vessel stage**, which runs `8 × 8 × 3 = 192` forward passes per volume at
`batch_size=1, num_workers=1` in FP32 — the GPU is almost idle between calls.

## Bottlenecks identified (from reading the current code)
| # | Bottleneck | Location | Planned fix |
|---|---|---|---|
| 1 | 192 serial forward passes/volume, `batch_size=1` | `model_utils.py:pred_and_save_masks_3d_divided` | Batch patches (size 16) per subject |
| 2 | `num_workers=1`, no prefetch | `model_utils.py` DataLoaders | `num_workers=3` |
| 3 | FP32 inference | both stages | `torch.cuda.amp.autocast` |
| 4 | 2× Python/subprocess + model reload per task | `batch_segmentation.py:run_batch_segmentation` | call inference in-process |
| 5 | Serial STEP-1 preprocessing | `batch_segmentation.py:process_file_batch` | `ThreadPoolExecutor` |

## Environment / cluster facts
- Cluster: **randi (CRI)**. Env: `vanguard` (torch 1.11.0, torchio 1.2.1, unet 0.8.1).
- Images (read-only): `/ess/scratch/scratch1/annawoodard/MAMA-MIA-syn60868042/images`
- Ground-truth `.npz` (52% run): `/ess/scratch/scratch1/t-9sbose/vessel_segmentations/`
- `.npz` format: one key `vessel`, shape `(X, Y, Z)`, float16 probabilities in [0, 1].
- **Safety:** login-node tests are CPU-only/tiny; all GPU runs go through SLURM.
  The live segmentation array job must not be disturbed.

## Files
- `predict_fast.py` — batched + AMP inference functions (importable, in-process).
- `batch_segmentation_fast.py` — fast batch driver (parallel preprocess, no subprocess).
- `validate_outputs.py` — compares new outputs vs ground-truth `.npz` (Dice, abs diff).
- `submit_fast_smoke.slurm` — 1–2 file GPU smoke test via SLURM.
- `tests/` — CPU logic/unit tests runnable on the login node.

## Progress
See `lab_notebook.md` for the chronological record. Feature status below is
updated only after a test passes.

- [x] **Step 1 — Batched vessel inference** ✅ CPU test: bit-identical to original (`max_abs_diff=0`), incl. cross-subject batch boundary. See `tests/test_batching_equiv.py`.
- [x] **Step 2 — AMP (autocast)** ✅ Wired in `predict_fast` (`use_amp`, no-op on CPU). Real effect validated on GPU in Step 4 (AMP cannot be exercised on CPU).
- [x] **Step 3 — driver + parallel preprocess + num_workers** ✅ `batch_segmentation_fast.py`; CPU test confirms parallel preprocessing is bit-identical to serial (`tests/test_preprocess_parallel.py`). In-process inference path validated on GPU in Step 4.
- [ ] Step 4 — GPU smoke test via SLURM (A100-40GB, `submit_fast_smoke.slurm`)
- [ ] Step 5 — Accuracy validation vs ground truth (`validate_outputs.py`)

### How to run the CPU test (login-node safe)
```bash
micromamba activate vanguard
python faster-segmentation-test/tests/test_batching_equiv.py
```
