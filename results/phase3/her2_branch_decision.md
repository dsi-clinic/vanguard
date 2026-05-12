# Phase 0 → Phase 2 branch decision (HER2 Deep Sets)

**Date:** 2026-05-12 (Phase 0d initial)
**Updated:** 2026-05-12 (Phase 0e results in)
**Plan:** [HER2 Deep Sets next steps](../../.cursor/plans/her2_deep_sets_next_steps_cfdd86f6.plan.md)
**Status:** Phase 0e returned. The clean binary in the original plan did not trip; the result sits between branches. Awaiting human-in-the-loop decision before committing to Phase 2a cluster time.

## Finding

On the **n=68 HER2 ∩ tabular intersection cohort** with the canonical
3-fold map (`data/fold_map_her2.csv`, seed 42):

| model                                  | pooled AUC | mean fold AUC | std fold AUC |
|----------------------------------------|-----------:|--------------:|-------------:|
| log(N_points) LR (one feature)         | 0.603      | 0.610         | 0.071        |
| DS Phase 3 winner `cos_T80` (mean of 2 seeds)        | —          | 0.604         | 0.071        |
| DS Phase 3 winner `h128_d02_lfocal` (mean of 2 seeds) | —         | 0.595         | 0.070        |
| DS Phase 3 winner `h256_d02_lfocal` (mean of 2 seeds) | —         | 0.584         | 0.014        |

A single-feature logistic regression on `log(num_points + 1)` is **at or
above** every existing Deep Sets Phase 3 winner on the same 68 HER2 cases.
The Deep Sets models on HER2 are operating as a tumor-scale detector
(the `mean_max_logcount` pooling already concatenates `log(N)` per case);
the per-point latent channels are not adding extractable signal beyond
the count itself.

## Branch decision

Of the Phase 0 → Phase 1 decision gates in the plan:

1. ~~"Any existing DS Phase 3 config has HER2 AUC ≥ 0.676 with cross-seed std ≤ 0.05"~~ — not met (best mean is 0.604).
2. **"log(N_points) LR matches DS HER2 within 0.02 AUC: declare 'model learned tumor scale', document, skip to Phase 2"** — **met** (gap is +0.006 to -0.020).
3. Default: proceed to Phase 1 — superseded by gate 2.

**Path taken:** **Branch 2** — Phase 1 HER2-only training and Phase 2b
HER2-only attention are cancelled. Phase 2a attention pooling on the
**full cohort (n=980)** is the only remaining lever that could move the
HER2 metric: it asks whether the bottleneck is the mean/max pooling
discarding per-point structure that exists but is being averaged out.

## Comparator caveat (the open question Phase 0e was supposed to close)

The 0.676 XGBoost HER2 AUC in
`results/model_family_robustness_ispy2_subtype_summary.csv` is from a
**full-cohort 5-fold XGB** whose HER2-stratum AUC was read post hoc —
a different patient set (n=808 total) and a different fold scheme than
our n=68 intersection 3-fold.

Phase 0e (XGBoost vessel_all on the **same n=68 cohort** with the
**same fold map**, **HER2-only training**) was the apples-to-apples
comparator. **Result (slurm job 852128):**

| metric          | value |
|-----------------|------:|
| pooled AUC      | **0.645** |
| mean fold AUC   | 0.661 |
| std fold AUC    | 0.124 |
| per-fold AUC    | 0.683, 0.802, 0.500 |

This sits **between** the two decision branches:

- XGB **does** clear log(N) and DS on the same cohort (0.645 vs 0.603
  vs 0.604) — extractable per-feature signal exists beyond tumor scale.
- But the gap is **small** (~0.04 pooled), the **fold std is enormous**
  (0.124 — one fold at 0.50), and the 95% CI on any AUC at n=68 is
  roughly ±0.12. The "XGB > log(N)" lift is **not** statistically clean.
- HER2-only training underperforms full-cohort training on HER2 (0.645
  vs 0.676) — most of that delta is training-size penalty (~46
  vs ~632 training cases per fold).

## Reframed question for Phase 2a

The original phrasing — "is the architecture the only bottleneck?" — no
longer fits the data. The sharper question is:

> Can attention pooling on the **full cohort (n=980)** recover the
> modest per-feature signal that XGB extracts on the tabular side but
> mean/max pooling currently averages out? The plausible lift is
> **0.02-0.04**, not 0.10.

Arguments for submitting Phase 2a anyway:
- Full-cohort n=980 has no cohort-size penalty, so the measured lift is
  the *true* architectural lift.
- Six variants × ~30 min each is cheap cluster time.
- Mean/max pooling provably discards per-point variance; there is a
  clear mechanism the experiment tests.

Arguments against:
- Even a successful Phase 2a moves DS from ~0.60 to ~0.65 on HER2,
  **not** past the full-cohort XGB bar of 0.676.
- The headline gap to XGB will likely remain.
- At n=68, the fold std overwhelms any plausible architectural lift in
  the HER2 readout itself.

**This is a human-in-the-loop decision** — the agent does not auto-submit
Phase 2a.

## Dependencies and ordering

- **Phase 0e (DONE):** slurm job 852128 completed in <1 min.
  `experiments/her2_phase0/xgb_vessel_all/{predictions.csv,metrics.json}`
  and `logs/her2-phase0-xgb-852128.out` are the artifacts.
- **Local scaffolding (DONE, no cluster needed):** `AttentionPool` in
  `deepsets_model.py` + 16 tests in `tests/test_attention_pool.py`,
  `evaluation/late_fusion.py` + 6 tests in `tests/test_late_fusion.py`,
  `configs/deepsets_sweep_her2_attention.yaml` (6 variants). All 52
  affected tests pass.
- **Phase 2a (pending human decision):** Submit with
  `SWEEP_CONFIG=configs/deepsets_sweep_her2_attention.yaml BASE_MANIFEST=experiments/deepsets_ispy2_pointfeat_baseline/deepsets_manifest.csv bash scripts/sweep_deepsets_train.sh`.

## Artifacts produced in Phase 0

- `scripts/aggregate_her2_diagnostics.py` — walks `metrics.json` for HER2 stratum AUC/AP.
- `scripts/build_her2_fold_map.py` — canonical fold-map generator.
- `scripts/her2_logn_lr_baseline.py` — one-feature log(N) LR.
- `scripts/her2_xgb_baseline.py` — vessel_all XGB (HER2-only training, n=68).
- `scripts/build_her2_tracker.py`, `scripts/append_her2_xgb_to_tracker.py` — tracker maintenance.
- `slurm/submit_her2_phase0_xgb.slurm` — slurm wrapper (job 852128 succeeded).
- `data/her2_intersection_case_ids.csv` — n=808 cohort with subtype/label.
- `data/fold_map_her2.csv` — 68×{case_id, fold, n_splits=3, random_state=42}.
- `experiments/her2_phase0/logn_lr/` — Phase 0d predictions.csv + metrics.json.
- `experiments/her2_phase0/xgb_vessel_all/` — Phase 0e predictions.csv + metrics.json.
- `results/her2_phase0_existing_runs.csv` — long-format HER2 metrics across phase3 winners.
- `results/her2_deepsets_tracker.csv` — 17 rows (Phase 0d + Phase 3 prior + Phase 0e).

## Phase 2 scaffolding (no compute yet)

- `deepsets_model.py` — `AttentionPool` module; `attention` and `attention_logcount` in `POOLING_CHOICES` and `_pooling_width`.
- `train_deepsets.py` — `attention_hidden_dim` config wired into the model constructor.
- `config.py` — default `attention_hidden_dim: 32`.
- `tests/test_attention_pool.py` — 16 tests (permutation invariance within-case and across-case-order, shape contract, softmax sums to 1, gradient flow).
- `evaluation/late_fusion.py` + `tests/test_late_fusion.py` — 6 tests covering fold-map alignment, case-id-set mismatch, missing-column raise, LR-overfit flag.
- `configs/deepsets_sweep_her2_attention.yaml` — 6 variants ({attention, attention_logcount} × {16, 32, 64} attention_hidden_dim) on full cohort.
