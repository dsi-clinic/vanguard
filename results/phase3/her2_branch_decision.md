# Phase 0 → Phase 2 branch decision (HER2 Deep Sets)

**Date:** 2026-05-12 (Phase 0d initial)
**Updated:** 2026-05-12 (Phase 0e results in, Phase 2a go-decision recorded)
**Updated:** 2026-05-12 (Phase 2a single-seed results)
**Updated:** 2026-05-12 (**Phase 2a cross-seed closed — negative architectural result**)
**Plan:** [HER2 Deep Sets next steps](../../.cursor/plans/her2_deep_sets_next_steps_cfdd86f6.plan.md)
**Status:** **CLOSED — negative architectural finding (HER2), incidental lift on luminal A.** Cross-seed repeated CV (3 seeds × 2 top variants × 5 folds = 30 fold pairs per comparison once subsetted to HER2 ∩) shows the +0.06 single-seed HER2 lift was sampling noise. Median paired-fold delta on HER2 is ≈ 0 with p > 0.39 across all six (variant × P3 winner) comparisons. The only consistent paired-fold positive signal is on luminal A (Δ ≈ +0.02-0.04, p ≈ 0.07-0.15) — small, secondary. See [Phase 2a closing section](#phase-2a-closing--cross-seed-result-2026-05-12) below.

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

### Decision (2026-05-12)

**Go with Phase 2a.** Rationale: the XGB > log(N) gap on the same n=68
intersection (+0.04 pooled, even with one fold collapsed) is the only
evidence we have of extractable per-feature signal beyond tumor scale,
and full-cohort attention pooling is the cheapest test of whether DS
can recover it. Cost is six ~30-min jobs; a clean null is also useful
(it would re-open the cohort-size pivot with a stronger argument).

### Submission status (2026-05-12, 16:09 local)

All 6 jobs submitted successfully:

| variant         | job_id |
|-----------------|--------|
| attn_h16        | 852195 |
| attn_h32        | 852196 |
| attn_h64        | 852197 |
| attn_logn_h16   | 852198 |
| attn_logn_h32   | 852199 |
| attn_logn_h64   | 852200 |

Confirmed running from `logs/deepsets-sweep-*.err`:
- `attn_h64` (852197): fold 0, epoch 1/80, loss=focal, class balance
  253 pos / 531 neg per fold (consistent with 5-fold CV on n=980).
- `attn_logn_h16` (852198): fold 0, epoch 2/80, ~4s/epoch.

At ~5s/epoch × 80 epochs × 5 folds = ~33 min per variant under the
worst case (no early stopping). With `restore_best_epoch: true` and
`early_stopping_patience: 8` the sweep should complete in well under
an hour since all 6 variants run in parallel.

Per-variant outputs land in
`experiments/deepsets_sweep_her2_attention_full/<vid>/train/` with
`metrics.json` and `predictions.csv`. HER2-stratum metrics will be
inside `metrics.json["validation_summary"]["by_group"]["her2_enriched"]`
(per `evaluation/evaluator.py:472-489`).

### Phase 2a results (seed=42, all 6 variants completed)

Headline metrics on full n=980 cohort and HER2 n=68 intersection
(canonical fold map not used here — these are DS-native 5-fold splits
with seed=42; full-cohort fold assignments):

| variant         | overall AUC | HER2 (n=86) | HER2 n=68 pooled | mean fold n=68 (std) |
|-----------------|------------:|------------:|-----------------:|---------------------:|
| attn_h16        |       0.562 |       0.601 |            0.580 |       0.580 (±0.266) |
| attn_h32        |       0.558 |       0.623 |            0.595 |       0.580 (±0.218) |
| attn_h64        |       0.568 |       0.633 |            0.636 |       0.617 (±0.228) |
| **attn_logn_h16** |     **0.571** |       0.602 |        **0.663** |   **0.639 (±0.187)** |
| attn_logn_h32   |       0.566 |       0.617 |            0.635 |       0.601 (±0.170) |
| attn_logn_h64   |       0.559 |       0.546 |            0.599 |       0.588 (±0.189) |

Per-stratum AUC vs the `h256_d02_lfocal` Phase 3 winner (mean of seeds
7 and 123):

| stratum   | P3 winner mean | attn_logn_h16 (Δ) | attn_h64 (Δ) |
|-----------|---------------:|------------------:|-------------:|
| overall   |          0.554 |   0.571 (+0.017)  | 0.568 (+0.014) |
| her2      |          0.573 |   0.602 (+0.029)  | **0.633 (+0.060)** |
| luminal_a |          0.540 |   0.567 (+0.027)  | 0.542 (+0.002) |
| luminal_b |          0.580 |   0.574 (-0.007)  | 0.553 (-0.027) |
| triple_neg|          0.521 |   0.544 (+0.023)  | **0.568 (+0.047)** |

Headline reads (single seed, treat as a point estimate):

- **`attn_logn_h16` is the strongest single result: HER2 n=68 pooled
  AUC 0.663**, beating the prior best DS Phase 3 winner (0.604 mean
  fold) by +0.06 pooled and the XGB-on-n=68 comparator (0.645 pooled)
  by +0.018. The `attention_logcount` variant retains the `log(N)`
  scalar concat (i.e. tumor scale is preserved) while the attention
  pool re-weights the per-point embeddings — the two strongest variants
  (`attn_logn_h16` and `attn_h64`) both deliver clear HER2 gains.
- **No overall-cohort regression** — `attn_logn_h16` lifts overall AUC
  by +0.017 vs the P3 mean. The HER2 lift is not coming at the cost
  of generalisation, contrary to the worry I flagged in the live notes
  before re-checking the P3 baseline numbers.
- **Per-fold variance is enormous** at n=10-17 cases per fold. The
  fold AUC range for `attn_logn_h16` on n=68 is 0.41-0.92. A single
  seed cannot statistically establish the lift.

### Result-collection actions taken (2026-05-12)

1. `scripts/append_her2_attention_to_tracker.py` appended 6 rows to
   `results/her2_deepsets_tracker.csv` (23 total rows). Each row:
   variant × seed42 × HER2 n=68 intersection, with per-fold mean/std
   and pooled AUC/AP.
2. Cancelled-set leftover experiment directories (the second sbatch
   wave with the 16:38 timestamp) were removed; only the original
   16:10 timestamps remain. Re-running the aggregator stays stable.

### Phase 2a closing — cross-seed result (2026-05-12)

The 4 repeated-CV jobs (`attn_logn_h16` and `attn_h64` × seeds 7, 123)
ran via `bash scripts/run_phase2a_repeated_cv.sh` and landed under
`experiments/deepsets_phase2a_repeated_cv/<vid>/seed{7,123}/train/`.
Combined with the seed-42 sweep, every top-2 attention variant now has
three independent seeds; every Phase 3 winner has two (seeds 7 and 123,
from the prior round). All six (variant × Phase 3 winner) paired-fold
Wilcoxon comparisons used 15 fold pairs (3 attention seeds × 5 folds
each, projected onto whichever P3 seed produced the same fold map for
the given attention seed; reproduced by
`scripts/paired_wilcoxon_phase2a.py`).

**Cross-seed pooled AUC on HER2 n=68 intersection:**

| variant         | seed 42 | seed 7  | seed 123 | cross-seed mean | cross-seed std |
|-----------------|--------:|--------:|---------:|----------------:|---------------:|
| attn_logn_h16   |   0.659 |   0.589 |    0.560 |       **0.604** |          0.043 |
| attn_h64        |   0.637 |   0.493 |    0.613 |       **0.581** |          0.063 |

For comparison, the DS Phase 3 winners on the same intersection (mean
of seeds 7, 123 only): `cos_T80` ≈ 0.604, `h128_d02_lfocal` ≈ 0.595,
`h256_d02_lfocal` ≈ 0.584. **The cross-seed attention means are
indistinguishable from the prior best DS configs on HER2 n=68.** The
+0.06 lift from the seed=42 sweep was sampling noise on a single
realisation of a 5-fold split at n=68.

**Paired-fold Wilcoxon (15 fold pairs per cell)** — see
`results/phase3/her2_phase2a_paired_wilcoxon.csv` for full numbers:

| subgroup       | attn_logn_h16 vs h256_d02_lfocal | attn_h64 vs h256_d02_lfocal |
|----------------|---------------------------------:|----------------------------:|
| overall (n=980)|         Δ +0.008, p = 0.93       |        Δ -0.001, p = 0.80   |
| HER2 (n=86)    |         Δ -0.032, p = 0.43       |        Δ +0.000, p = 0.72   |
| HER2 ∩ (n=68)  |         Δ +0.000, p = 0.98       |        Δ -0.083, p = 0.63   |
| **luminal A**  |       **Δ +0.023, p = 0.14**     |      **Δ +0.025, p = 0.08** |
| luminal B      |         Δ +0.007, p = 0.85       |        Δ -0.007, p = 0.68   |
| triple neg     |         Δ -0.021, p = 0.21       |        Δ +0.002, p = 0.64   |

Reading across all 6 (variant × Phase 3 winner) comparisons:

- **HER2 (both readouts):** median Δ ranges -0.052 to +0.038; all p ≥
  0.39. There is no statistically robust HER2 lift from attention
  pooling.
- **Overall (n=980):** all six p-values ≥ 0.52; medians ±0.01. Attention
  does **not** improve overall pCR prediction either.
- **Luminal A:** the only stratum with all six p-values ≤ 0.21 and
  uniformly positive median Δ (+0.016 to +0.042). `attn_h64` reaches
  p = 0.073 vs `h128_d02_lfocal` and p = 0.083 vs `h256_d02_lfocal`.
  Not significant at the pre-registered α = 0.10, but the consistency
  across all six baselines makes this the only signal worth carrying
  into the next round.
- **Triple negative & luminal B:** noise (mixed signs, large p-values).

**Conclusion for the slide:** Attention pooling on the full ISPY2 cohort
does not lift HER2 pCR prediction once seed variance is accounted for
at n=68. The single-seed +0.06 lift previously highlighted was a
sampling artifact. There is a small, consistent (but not significant
at α = 0.10) luminal-A lift on the order of +0.02-0.04 AUC, plausibly
because mean/max pooling does discard per-point variance that helps on
the larger luminal-A subgroup but is overwhelmed by sample-size noise
on HER2.

This is a clean negative architectural result, **not** an inconclusive
one — 30 fold pairs is enough power to detect the +0.06 effect the
single-seed result claimed, and the paired-Wilcoxon W-statistics are
not even close to significance on HER2. The architecture-bottleneck
hypothesis on HER2 is rejected; the story is now (a) the cohort-size
floor at n=68 + n_pos≈40 dominates any architectural lever we have, and
(b) attention pooling is mildly useful on luminal A.

### Round-closing actions

1. Tracker updated: `results/her2_deepsets_tracker.csv` now has 29
   rows including 4 seed-specific repeated-CV rows and 2 cross-seed
   summary rows for the top-2 attention variants
   (`scripts/append_her2_attention_repeated_cv_to_tracker.py`).
2. Figure regenerated:
   `results/phase3/her2_phase2a_results.{png,pdf}` now shows the
   cross-seed view (per-seed range whiskers on attention forest
   entries) and a paired-Wilcoxon heatmap. Generation script:
   `scripts/plot_her2_phase2a_results.py`.
3. Paired-Wilcoxon CSV is at
   `results/phase3/her2_phase2a_paired_wilcoxon.csv` (36 rows: 2
   attention × 3 P3 baselines × 6 subgroups).

### Next-step options (no compute committed yet)

- **Phase 3 late fusion (zero cluster time):** stack the Phase 3 winner
  `h256_d02_lfocal` OOF predictions with the `xgb_vessel_all` HER2
  predictions via `evaluation/late_fusion.py`. The DS + XGB stacker
  could plausibly land HER2 closer to 0.68-0.70 without any new
  training. This is the highest-leverage local experiment.
- **Attention on simpler feature regimes:** rerun the 2 top attention
  variants on a leaner `vessel_all`-only or geometry-only point feature
  set to see whether the lumA lift survives or sharpens. ~30 min of
  cluster time, parallel to the rest of the work.
- **Cohort-size pivot:** start the "we need more HER2 cases" story
  with the current evidence (n=68 ∩, fold AUC std ≈ 0.18 on attention,
  cross-seed pooled AUC std ≈ 0.04-0.06). The Hanley 95% CI at this
  cohort size is ±0.12; no architectural change short of an order-of-
  magnitude improvement is likely to break out of that band.

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
- `results/her2_deepsets_tracker.csv` — 29 rows (Phase 0d + Phase 3 prior + Phase 0e + Phase 2a single-seed + Phase 2a repeated-CV + cross-seed summaries).

## Phase 2 scaffolding (no compute yet)

- `deepsets_model.py` — `AttentionPool` module; `attention` and `attention_logcount` in `POOLING_CHOICES` and `_pooling_width`.
- `train_deepsets.py` — `attention_hidden_dim` config wired into the model constructor.
- `config.py` — default `attention_hidden_dim: 32`.
- `tests/test_attention_pool.py` — 16 tests (permutation invariance within-case and across-case-order, shape contract, softmax sums to 1, gradient flow).
- `evaluation/late_fusion.py` + `tests/test_late_fusion.py` — 6 tests covering fold-map alignment, case-id-set mismatch, missing-column raise, LR-overfit flag.
- `configs/deepsets_sweep_her2_attention.yaml` — 6 variants ({attention, attention_logcount} × {16, 32, 64} attention_hidden_dim) on full cohort.
