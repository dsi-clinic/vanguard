# Plan: Improving GNN Baseline Performance with More Advanced Modeling

**Scope:** the vessel-centerline **GNN** pCR classifier only (`gnn/model.py`,
`gnn/train.py`). Not the tabular or Deep Sets tracks. Derived from `NOTES.md`
("Look into more sophisticated modeling approaches" + "This Week" items) and the
diagnosis recorded in `LAB_NOTEBOOK.md` (2026-07-15 entries).

**Author intent (from the task):** come up with a plan to improve baseline model
performance using more advanced modeling techniques, and *record every decision
carefully*. This file is that record. Each numbered **Decision (Dn)** states what
we chose, why, and what we rejected — so a future agent can audit or revise a
choice without re-deriving it.

Status legend: `[ ]` not started · `[~]` in progress · `[x]` done ·
`[-]` dropped (with reason).

---

## 1. Where the baseline actually is (grounded facts, not assumptions)

Read directly from the code and notebook on 2026-07-20:

**Current architecture (`gnn/model.py`):**
- `GCNClassifier`: 2× `GCNConv` → ReLU → dropout, then **global mean pool**, then
  a single linear head → 1 logit/graph. `hidden_dim=32`, `num_layers=2`,
  `dropout=0.2`.
- `EdgeGNNClassifier`: the junction-mode counterpart (`EdgeConditionedConv`),
  same mean-pool + linear-head readout.
- Optional graph-level covariates concatenated onto the pooled embedding before
  the head (`graph_dim>0`).

**Current training (`gnn/train.py`, confirmed by reading lines 440–483):**
- Optimizer: plain `Adam`, `lr=0.001`. **No weight decay, no LR scheduler, no
  gradient clipping.**
- Loss: plain `BCEWithLogitsLoss`. **No `pos_weight`** (pCR is the minority
  class, fold pCR rates 0.09–0.48 per the notebook).
- **Fixed 25 epochs, no early stopping.** The model used for prediction is the
  **final-epoch** model (line 480, right after the loop) — there is **no
  best-validation-epoch selection**. A fold that peaks at epoch 10 and decays is
  reported at its decayed epoch-25 state.
- Per-fold node-feature standardization (fit on train split only) — good, keep.
- 5-fold CV, `evaluation/` framework, standard `metrics.json`/`predictions.csv`.

**Current features (`configs/gnn.yaml`):** `["peak_time", "radius"]` (voxel mode).
Richer `seg_*` geometry, degree, washout, bifurcation-angle, and clinical
covariates exist in the loader but were **not** used in the latest experiments.

**Current performance (`LAB_NOTEBOOK.md`, 2026-07-15):**
- GNN val AUC ≈ **0.51**, near chance, **unstable across folds** (fold-only
  std ≈ 0.038). Fold ranking is consistent across seeds (Spearman 0.9–1.0):
  fold 0 always worst (~0.45), fold 2 always best (~0.55).
- Tabular XGBoost on 33 hand-summarized columns (mean/std/q10/q50/q90 of the 6
  node features + counts): **0.540 ± 0.018** — modestly *above* the GNN, at
  ~half the fold spread. Logistic regression 0.499 (chance).
- Per-fold AUC does **not** track between GNN and XGBoost (GNN's worst fold is
  XGBoost's best) → some fold difficulty is representation-dependent, not purely
  a cohort confound.
- Cohort is DUKE-dominated (94% after single-breast harmonization). ISPY1 sits
  *below* chance (~0.44).

---

## 2. Diagnosis: why "more advanced modeling" is the right lever (and its limits)

Three distinct problems are tangled together. Advanced modeling addresses some
but not all — naming which is which is the most important decision here.

1. **Under-trained / mis-selected models (optimization hygiene).** Fixed epochs +
   final-epoch reporting + no imbalance handling means we are almost certainly
   *not* reporting each model at its best, and the loss is dominated by the
   majority (non-pCR) class. This is cheap to fix and could account for part of
   the instability. **Advanced modeling helps here.**

2. **Lossy representation / readout.** XGBoost beats the GNN using *quantiles* of
   the same node features, while the GNN sees only the **mean** (global mean
   pool). Distributional tails (max enhancement, quantiles) are exactly what
   mean-pooling destroys, and the notebook already flags mean-pool signal-washout
   as a concern. Attention/multi-statistic readout and a stronger conv operator
   target this directly. **Advanced modeling helps here — highest-value lever.**

3. **Possibly-weak intrinsic signal + cohort confound.** ISPY1 below chance, a
   DUKE-dominated cohort, and seed-stable fold ranking all point to a
   cohort/acquisition component that **no architecture change can fix**. This
   bounds the expected upside and is *the reason the plan is gated* — we do not
   spend the expensive-technique budget until a cheap change shows the signal is
   there to be captured.

**Honest ceiling statement:** at ~0.51–0.54 AUC we cannot assume a large gain is
available. The plan is therefore designed to *fail cheaply and informatively* —
each tier has a pre-registered stop rule, matching the lab's existing practice
(the single-breast experiment's pre-registered stop rule, `LAB_NOTEBOOK.md`).

---

## 3. Guiding principles (constraints this plan must respect)

From `AGENTS.md`, `agents.local.md`, and `~/.claude/CLAUDE.md`:

- **Simplicity first** (`agents.local.md`): start with the cheapest change that
  could move the needle; add complexity only after it earns its place.
- **Fail fast, few fallbacks.** No defensive try/except, no "try it multiple
  ways" chains. One interface per choice; break loudly on mismatch.
- **Reuse before writing.** Check the shared package first; match existing style.
  `GCNConv` is already imported and reused across `model.py` and `pretrain/`.
- **No heavy compute on the head node.** Every training run goes through Slurm
  (`gnn/slurm/`, `tier1q`, account `karczmar-lab`).
- **Reproducible + audited.** Every new results dir gets a `README.md` (command,
  config, git commit, inputs) *before* the result is reported; every session
  appends to `LAB_NOTEBOOK.md`.
- **Don't silently change cohort/splits.** Fold definitions are frozen; changing
  them is an explicit, separate decision (see D1 and Open Question OQ-1).

---

## 4. Fixed evaluation protocol (decide once, hold constant across all tiers)

This is the single most important set of decisions: without it, any AUC change
is uninterpretable. These are locked for the whole plan.

- **Decision D0.1 — Primary metric: paired ΔAUC vs. the baseline, per (seed×fold).**
  Report mean ΔAUC with the paired distribution, not just two aggregate numbers.
  *Why:* fold variance (±0.038) dwarfs plausible effect sizes, so an unpaired
  comparison is underpowered. This is exactly the design the single-breast
  experiment used (paired delta-AUC by seed×fold, n=15). *Rejected:* comparing
  aggregate mean AUCs alone — too noisy at this effect size.

- **Decision D0.2 — Frozen folds, shared across arms.** Every arm uses the *same*
  split definition as the current baseline (`split_mode`, `random_state=42`, same
  `n_splits=5`). No arm may change folds and architecture at once.
  *Why:* the notebook already caught the "frozen-fold check proves cohort parity,
  not representation isolation" subtlety; we keep folds identical so a ΔAUC is
  attributable to the *model*, not the split. *Rejected:* re-freezing folds
  stratified by dataset — that is a real idea but a **separate** experiment
  (OQ-1), not bundled in here.

- **Decision D0.3 — ≥5 seeds per arm; separate fold variance from run variance.**
  Report the GNN's fold-only std *and* run-only std separately.
  *Why:* the notebook showed tabular models were deterministic (pure fold std)
  while the GNN mixes fold + training-run noise; conflating them misled the first
  std comparison. *Rejected:* 3 seeds (what prior runs used) — too few to trust a
  small ΔAUC.

- **Decision D0.4 — Report OOF AUC broken out by `dataset` (and ISPY2 laterality).**
  A gain that only appears on DUKE (94% of cohort) is a cohort artifact, not a
  method win. Reuse the existing per-dataset OOF breakdown already in
  `analysis/`. *Rejected:* pooled-only reporting — hides the confound.

- **Decision D0.5 — Keep the leakage canary and confound plots every run.**
  `pcr_dummy` learnability check + `prediction_vs_num_nodes.png` stay on, so a
  spurious "win" driven by graph size or a wiring bug is caught immediately.

- **Decision D0.6 — Reference baseline is frozen and named.** The comparison
  point is **arm-1 voxel mode, `["peak_time","radius"]`, current
  `GCNClassifier`** (the exact config behind the 0.51 number). Every ΔAUC in
  this plan is against *that*, on the frozen folds. *Why:* a moving baseline
  makes tiers incomparable.

---

## 5. Tiered plan (cheap → expensive, each gated by the previous)

Ordering rationale: Tiers 0–1 are cheap, touch only `train.py`/`model.py`, and
attack diagnosis-problems #1 and #2 (the fixable ones). Tiers 2–3 are expensive
(new pretraining data, temporal encoders) and only make sense if a cheaper tier
shows the signal is capturable. **Do not start a tier until the prior tier's gate
passes.**

### Tier 0 — Optimization hygiene (cheapest; directly from `NOTES.md` TODO)

`NOTES.md` explicitly lists: early stopping, clipping, LR scheduling. Add
imbalance handling and best-epoch selection alongside, because they are the same
size of change and address diagnosis-problem #1.

- **[x] D1.1 — Best-validation-epoch model selection.** *Done* (`restore_best_epoch`).
  **Refinement recorded:** the selection signal is validation **loss**, not val
  AUC as originally written — this matches `deepsets/train.py` (keeping the two
  tracks consistent) and val loss is smoother than AUC on these small imbalanced
  folds. The correctness win (not reporting the arbitrary final epoch) holds
  either way. *Rejected:* early-stopping-only without checkpointing the best epoch.
- **[x] D1.2 — Early stopping with patience.** *Done* (`early_stopping_patience`;
  `epochs` ceiling raised to 100 in `gnn_tier0.yaml`). Verified firing on the
  smoke cache (fold stopped at epoch 5, best epoch 1).
- **[x] D1.3 — `pos_weight` in `BCEWithLogitsLoss`.** *Done* via `loss:
  weighted_bce` (`pos_weight = n_neg/n_pos` from the fold train split). A
  zero-positive fold falls back to `pos_weight=1.0` **and logs its class
  balance** — a degenerate split surfaced, not silently defaulted. `focal` is
  also wired (`loss: focal`) for later if `pos_weight` under-delivers.
- **[x] D1.4 — Weight decay + gradient clipping.** *Done* (`weight_decay`,
  `max_grad_norm`). `gnn_tier0.yaml` uses `1e-4` / `1.0`.
- **[x] D1.5 — LR scheduler.** *Done* (`lr_scheduler: plateau` in
  `gnn_tier0.yaml`; `cosine` also available). Composes with early stopping.

**Implementation note (fail-fast):** all Tier-0 knobs reuse the existing shared
`model_params` config keys that `deepsets/train.py` already consumes — the GNN
simply had never wired them in. `build_loss_fn` / `FocalWithLogitsLoss` are small
local copies of the Deep Sets versions (reimplemented, not cross-imported, to
avoid coupling the two model tracks). Every existing GNN config's behavior is
preserved: the shared defaults (`weight_decay=1e-4`, `loss=weighted_bce`) differ
from the historical GNN runs, so `configs/gnn.yaml` and `gnn_smoke.yaml` now
**pin** the historical values explicitly (D0.6) rather than silently inheriting
the new defaults. No fallback chains.

**Status: code + configs landed on branch `feat/gnn-tier0-training`, validated
end-to-end on the 8-case smoke cache.** The real 5-fold run (baseline
`gnn.yaml` vs `gnn_tier0.yaml`, ≥5 seeds, D0 protocol, via Slurm) is the next
action and is **blocked** on a data-path issue found during verification:
`pcr_labels.csv` has moved from `.../saritbose/pcr_labels.csv` to
`.../saritbose/metadata/pcr_labels.csv` (the committed configs and
`agents.local.md` still point at the old path). Resolve that path decision
before submitting (see §7 / Open Questions).

- **Gate G0 (pre-registered):** run baseline + Tier-0 stack under the D0 protocol.
  **Proceed to Tier 1 only if** mean paired ΔAUC > 0 with the paired distribution
  not straddling 0 *or* fold-only std drops meaningfully (stability win counts).
  If Tier 0 does nothing on either axis, that is itself a finding (the problem is
  representation/signal, not optimization) — go **straight to Tier 1's readout
  change**, which is the representation lever, and record the null.

### Tier 1 — Architecture: readout and convolution (the highest-value lever)

Attacks diagnosis-problem #2 (mean-pool washes out the distributional signal
XGBoost exploits). Do readout **before** fancier convs — it is cheaper and the
evidence (quantiles beat mean) points straight at it.

- **[ ] D2.1 — Multi-statistic readout: concat `[mean ‖ max ‖ sum]` pool.**
  Replace the lone `global_mean_pool` with concatenated mean+max(+sum) pools into
  the head. *Why:* directly restores the tail/extreme information (peak
  enhancement, quantiles) that beat the mean in the tabular baseline; ~10 lines,
  no new layer types. **This is the first thing to try in Tier 1.** *Rejected as
  the opener:* attention pooling (D2.2) — more parameters, try only if
  multi-stat readout under-delivers.
- **[ ] D2.2 — Attention readout (`GlobalAttention` / `Set2Set`).** A learned,
  node-weighted pooling. *Why:* if some vessels/segments carry the pCR signal and
  most are noise, attention can up-weight them — and the attention weights double
  as **saliency maps** (a separate `NOTES.md` TODO). *Gated behind D2.1:* only if
  multi-stat readout shows a signal worth refining.
- **[ ] D2.3 — Stronger conv operator: `GraphSAGE` or `GATConv`.** Swap `GCNConv`
  for one alternative (pick **one**, not a sweep-of-everything — start with
  `GraphSAGE` for robustness with max-aggregation; `GATv2Conv` if we want
  attention + saliency and the graphs are small enough). *Why:* `GCNConv`'s
  symmetric-normalized mean aggregation is the weakest common operator and, like
  mean-pool, is averaging-biased. *Rejected:* GIN (built for graph-isomorphism /
  molecular tasks; less obviously suited to continuous per-node kinetic signal);
  running GCN/SAGE/GAT/GIN as a 4-way sweep (violates simplicity-first — commit to
  one, justify, and only branch if it fails).
- **[ ] D2.4 — Depth + over-smoothing control: JumpingKnowledge or residuals,
  and/or `BatchNorm`/`LayerNorm` between convs.** *Why:* only if we want >2 layers;
  2-layer GNNs under-reach on long vessel paths, but naive deepening over-smooths.
  *Gated:* only pursue if a wider receptive field is hypothesized to matter (e.g.
  segment/junction modes where paths are longer). Keep at 2 layers otherwise.

- **Also settle the features confound here (D2.5).** Before crediting architecture
  for any gain, run **one** arm that gives the *current* `GCNClassifier` the
  richer `seg_*`/geometry/degree feature set that the latest experiments omitted.
  *Why:* the notebook explicitly flags that the richer feature set was never
  tested and could change the picture; if features alone close the gap, we should
  know before attributing it to architecture. This is a feature arm, not an
  architecture arm — keep it labeled as such.

- **Gate G1 (pre-registered):** best Tier-1 arm vs. best Tier-0 model, D0 protocol.
  **Proceed to Tier 2 only if** the best architecture reaches an absolute OOF AUC
  that (a) clears the XGBoost tabular baseline (0.540) *and* (b) shows a positive
  paired ΔAUC that holds on non-DUKE datasets (D0.4), not DUKE-only. If Tier 1
  cannot beat a quantile-on-features XGBoost, the GNN is not yet earning its
  complexity and Tier 2/3 are premature — record that verdict.

### Tier 2 — Self-supervised pretraining (already prototyped; high ceiling)

`NOTES.md` "This Week: Pretraining task"; a contrast-forecasting pilot already
exists (`gnn/pretrain/`, `project_contrast_forecasting_pilot` memory). This is
where the currently-discarded **temporal** DCE signal re-enters the model.

- **[ ] D3.1 — Pretrain the encoder on node-level contrast forecasting (SSL),
  then fine-tune the classifier head.** Use `ContrastForecastGNN`
  (`gnn/pretrain/model.py`) as the shared encoder: predict each node's future
  enhancement frames (label-free), then transfer the `GCNConv` stack into
  `GCNClassifier` and fine-tune on pCR. *Why:* pCR labels are scarce and noisy
  (~1500 cases, imbalanced); a label-free objective over the full 4D signal is the
  standard way to get a stronger encoder. The forecasting pilot already beat
  trivial baselines (MAE 0.022 vs last-frame 0.225) on the placeholder data.
- **[ ] D3.2 — Keep the "does the graph matter" ablation.** The pretrain package
  already ships `PerNodeForecaster` (graph-free, same temporal head). Carry that
  ablation into fine-tuning: if the graph-free pretrained encoder fine-tunes as
  well as the GNN one, the message-passing is not what's helping — a falsifier we
  want kept live (matches the design doc §7.ii).
- **Open decisions inherited from the pilot** (`project_contrast_forecasting_pilot`
  memory, unresolved): **Q1** variable-T handling across cohorts; **Q2** target
  normalization for the forecasting loss. These must be settled *before* a
  real-cohort pretrain run — flagged here so they are not silently defaulted.

- **Gate G2:** fine-tuned-from-pretrained vs. best Tier-1 from-scratch, D0
  protocol. Proceed to invest further in pretraining only if it beats
  from-scratch on the paired ΔAUC, on non-DUKE data.

### Tier 3 — Explicit temporal encoding in the classifier (most speculative)

`NOTES.md` "This Week: Encode the temporal structure from contrast." Currently the
temporal axis is collapsed to scalars (`peak_time`, `washin_slope`, …). A per-node
temporal encoder (GRU/TCN over frames) inside the classifier — reusing the
`_TemporalForecastHead`-style encoder from `pretrain/` — would keep the dynamics
the loader is told to preserve (`AGENTS.md`: "Do not discard dynamic/kinematic
information").

- **[ ] D4.1 — Per-node GRU/TCN over raw frames → GNN, end-to-end for pCR.**
  Only pursue if Tier 2 shows the temporal signal carries pCR information (Tier 2
  is the cheaper test of the same hypothesis, since it reuses the temporal
  encoder without a full end-to-end classifier). *Rejected as an earlier tier:*
  it is the most code and the most compute for a hypothesis Tier 2 screens first.

---

## 6. Cross-cutting decisions

- **Decision D5 — One change per arm.** Never combine a readout change, a conv
  change, and a loss change in one arm and call the delta "the architecture."
  Ablate one axis at a time against the frozen baseline. *Why:* the notebook's
  own methodological correction (junction arm changed representation *and*
  architecture, muddying attribution) is the mistake this rule prevents.
- **Decision D6 — Config-driven, no new parallel scripts.** Every knob above is a
  `model_params`/config field consumed by the existing `gnn/train.py`, matching
  the repo's "extend the CLI/config, don't fork one-off scripts" preference
  (`prefer_config_over_new_scripts` memory). New model *classes* go in
  `gnn/model.py`; new training *behavior* goes in `train.py` behind config.
- **Decision D7 — Graph mode held fixed at voxel for Tiers 0–1.** Establish the
  modeling wins on the already-validated voxel baseline first; only revisit
  segment/junction modes (D2.4's longer paths) once a voxel-mode win exists.
  *Why:* changing mode + architecture at once re-introduces the confound D5 bans.
- **Decision D8 — Every arm writes a results `README.md` + `LAB_NOTEBOOK.md`
  entry** with the exact command, config snapshot, git commit, and the paired
  ΔAUC table, per repo policy — *before* the result is reported anywhere.

---

## 7. Open questions to resolve with the human (not silently defaulted)

- **OQ-1 — Re-freeze folds stratified jointly by dataset?** The notebook found
  fold pCR rates swing 0.09–0.48; joint dataset-stratification might cut fold
  variance. But it changes the frozen split, so it is a **separate, explicit**
  decision and would reset the D0.6 baseline. Recommend running it as its own
  A/B *after* Tier 0, not folding it into the modeling tiers.
- **OQ-2 — Is 0.54 (XGBoost) the bar, or is there a target AUC from the clinical
  side?** The gate G1 uses 0.540 as the "earn your complexity" threshold; if the
  team has a different clinically-meaningful bar, it changes the stop rules.
- **OQ-3 — Pretraining data source (Q1/Q2 above).** Duke placeholder vs. the full
  cohort, variable-T handling, and target normalization must be pinned before any
  real Tier-2 run.

---

## 8. First concrete step

Implement Tier 0 (D1.1–D1.5) as config fields in `gnn/train.py`, with **best-val-
epoch selection defaulted on** and everything else defaulted to reproduce the
current baseline exactly. Run baseline + Tier-0 on the frozen folds under the D0
protocol via Slurm. Evaluate against Gate G0. This is the cheapest change with a
plausible correctness fix (final-epoch reporting) already identified in the code.

*Nothing in this plan runs compute on the head node; every training arm is a
Slurm job under `tier1q`/`karczmar-lab`.*
