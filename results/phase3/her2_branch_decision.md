# Phase 0 → Phase 2 branch decision (HER2 Deep Sets)

**Date:** 2026-05-12 (Phase 0d initial)
**Updated:** 2026-05-12 (Phase 0e results in, Phase 2a go-decision recorded)
**Plan:** [HER2 Deep Sets next steps](../../.cursor/plans/her2_deep_sets_next_steps_cfdd86f6.plan.md)
**Status:** **Decision recorded — proceeding with Phase 2a.** The Phase 0e result split the two pre-registered branches; user chose the architecture-push reading (small XGB lift over log(N) is consistent with extractable per-feature signal that mean/max pooling may be averaging out). Phase 2a submitted as a 6-variant attention sweep on the full n=980 cohort.

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

### Pending: Phase 2a repeated CV (next slurm wave, 4 jobs)

Paired-fold testing requires matched fold assignments. The Phase 3
winners (cos_T80 / h128_d02_lfocal / h256_d02_lfocal) live at seeds 7
and 123. So the natural next step is to run the top 2 attention
variants at seeds 7 and 123 (seed=42 already done):

- `attn_logn_h16` × {7, 123}
- `attn_h64` × {7, 123}

4 jobs, same architecture and hyperparameters as Phase 2a, only
`model_params.random_state` overridden. Submit with:

```bash
bash scripts/run_phase2a_repeated_cv.sh
```

That driver clones each variant's seed-42 `runtime_config.yaml`,
overrides `random_state`, and submits via `slurm/deepsets_job.slurm`.
Outputs land at
`experiments/deepsets_phase2a_repeated_cv/<vid>/seed{7,123}/train/`.

Once those return, the analysis is:

1. Per-variant cross-seed mean and std (3 seeds: 42, 7, 123).
2. Paired-fold Wilcoxon of `attn_logn_h16` vs `h256_d02_lfocal` on
   HER2 n=68, seed-matched (10 fold pairs across seeds 7+123) — gate
   at `p < 0.10`.
3. Same paired test against `cos_T80` (the prior best HER2 single
   number on n=68: 0.604 mean fold).
4. If `attn_logn_h16` survives the paired test and the cross-seed
   std is ≤ 0.05, the round closes with a real "attention pooling
   recovers HER2 signal on full cohort" claim.
5. If it fails, the +0.06 single-seed lift is dismissed as noise and
   we either (a) try Phase 3 late fusion of the attention variant
   with the XGB comparator (`evaluation/late_fusion.py` is ready) or
   (b) pivot to the cohort-size story with the stronger evidence.

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
