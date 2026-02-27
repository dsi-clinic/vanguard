# Radiomics Baseline Lab Notebook

## Scope
- Project: `radiomics_baseline`
- Current focus: imaging-only signal discovery on **ISPY2** first.
- Exclusions for this phase: no subtype covariate models, no shape-feature optimization.

## Objective
- Establish a reliable imaging-only predictive signal for pCR before attempting cross-site generalization.

## Current Status (as of 2026-02-25)
- Focused ISPY2-only sweep completed (`slurm job 729092`).
- Train/test cohort after site filter:
  - Train: `n=680` (`pcr=0:461`, `pcr=1:219`)
  - Test: `n=300` (`pcr=0:203`, `pcr=1:97`)

## Experiments Run

### E1. Focused ISPY2 imaging-only training sweep
- Setup:
  - Features: cached peri5 multiphase extraction
  - Model: logistic elastic-net + grid search
  - Preprocessing: correlation pruning + feature selection
  - Validation: 5-fold CV
  - Filter: `site=ISPY2`
- Cases and outcomes:

| Rank | Experiment | CV AUC mean | CV AUC std | Test AUC | Features used |
|---|---|---:|---:|---:|---:|
| 1 | `ispy2_imgonly_mrmr50` | 0.5895 | 0.0470 | 0.6422 | 50 |
| 2 | `ispy2_imgonly_baseline` | 0.5880 | 0.0471 | 0.6422 | 50 |
| 3 | `ispy2_imgonly_kbest20` | 0.5868 | 0.0490 | 0.6409 | 20 |
| 4 | `ispy2_imgonly_mrmr20` | 0.5839 | 0.0476 | 0.6237 | 20 |

### E2. Pipeline integrity checks (bug/leakage audit)
- Verified feature selection is inside fold-specific fit path (no CV leakage from k-best/mRMR).
- Added/confirmed fail-fast checks for duplicate patient IDs and missing label joins.
- Ensured training-only sweeps reuse cached extraction outputs where possible.

## Outcomes / Takeaways
- Imaging-only signal is present but weak/moderate (`~0.64` test AUC on ISPY2).
- mRMR(50) did not materially outperform baseline k-best(50); effect size is small.
- Variance across CV folds is non-trivial (`std ~0.047–0.049`), so ranking by single test AUC alone is unreliable.
- Main bottleneck appears upstream of classifier tuning: feature quality/representation likely dominates.

## Hypotheses
1. Temporal DCE information is under-utilized in current feature representation.
2. Quantization/texture settings (`binWidth`, 2D vs 3D extraction) may be suppressing MRI texture signal.
3. Residual size-related confounding remains in non-shape features and dilutes biologic signal.
4. Low-quality cases (phase/mask issues) are adding enough noise to flatten model gains.

## Ideas and Next Experiments (prioritized)

### P1. Add stronger temporal feature representation
- Hypothesis: richer temporal descriptors will lift imaging-only signal.
- Experiment:
  - Use all available post-contrast phases where present (`0001..0005`).
  - Include subtraction kinetics (`wash_in`, `wash_out`) and optional `t_peak_voxel` maps.
- Success criterion:
  - Improvement in CV AUC mean with stable or lower CV std vs current best (`0.5895 ± 0.0470`).

### P2. Small PyRadiomics parameter sensitivity sweep
- Hypothesis: current discretization/extraction mode is suboptimal for this MRI domain.
- Experiment:
  - Sweep `binWidth` in `{5, 10, 25}`.
  - Sweep `force2D` in `{true, false}` (fixed axis initially).
  - Keep model family fixed to isolate extraction effects.
- Success criterion:
  - Consistent CV improvement over baseline across at least two settings.

### P3. Deconfound size signal without using shape features
- Hypothesis: non-shape features still carry size proxy effects.
- Experiment:
  - Residualize non-shape features against tumor volume using train-fold-only fitting.
- Success criterion:
  - Higher CV AUC mean and/or lower subtype disparity at equal model complexity.

### P4. Extraction QC gates
- Hypothesis: a subset of low-quality segmentations/phases is degrading aggregate performance.
- Experiment:
  - Add hard gates for tiny masks, missing/invalid phases, and poor mask-image overlap.
  - Report dropped-case counts and reasons.
- Success criterion:
  - Reduced CV std and improved mean AUC without over-pruning sample size.

### P5. Model family check after upstream cleanup
- Hypothesis: nonlinear interactions may remain after feature cleanup.
- Experiment:
  - Compare tuned gradient-boosted trees against logistic baseline on same curated features.
- Success criterion:
  - Meaningful lift in CV mean with calibration and test AUC not deteriorating.

## Decision Rules for Progression
- Keep ISPY2-only until one configuration shows a clear, repeatable improvement over current best CV mean with comparable/lower std.
- Only after that, evaluate cross-site generalization and site-shift behavior.

## Notes
- This notebook is intended as the running record for hypotheses, interventions, and outcomes.
- Update this file after each submitted sweep with:
  - config deltas,
  - ranked metrics (`AUC_test`, `AUC_CV mean ± std`),
  - concrete takeaways and next decision.

## Per-Run Entry Template

### Run YYYY-MM-DD — <short name>
- Job ID(s): `<job_id>`
- Scope: `<ISPY2-only / cross-site / etc.>`
- Goal: `<what this run is trying to prove or falsify>`

#### Config Deltas
- Base config: `<path or experiment name>`
- Changed settings:
  - `<key>: <old> -> <new>`
  - `<key>: <old> -> <new>`

#### Result Table (ranked)
| Rank | Experiment | CV AUC mean | CV AUC std | Test AUC | n_features |
|---|---|---:|---:|---:|---:|
| 1 | `<name>` | `<0.xxxx>` | `<0.xxxx>` | `<0.xxxx>` | `<n>` |
| 2 | `<name>` | `<0.xxxx>` | `<0.xxxx>` | `<0.xxxx>` | `<n>` |

#### Outcome / Takeaways
- `<key finding 1>`
- `<key finding 2>`
- Decision: `<promote / repeat / drop>`

#### Next Action
- `<single next experiment with concrete change>`

## Planned Entry (Pre-Submission)

### Run 2026-02-26 — P1 temporal-enrichment ISPY2 imaging-only
- Job ID(s): `729263` (active), `729261` (superseded: added fixed-window kinetics), `729259` (superseded: removed optional t_peak confound), `729255` (superseded before phase-aggregation fix)
- Scope: `ISPY2-only`
- Goal: test whether richer DCE temporal representation improves imaging-only signal beyond current best (`CV 0.5895 ± 0.0470`, `Test 0.6422`).

#### Config Deltas
- Base config: `peri5_multiphase_logreg` / focused ISPY2 imaging-only train sweep
- Changed settings:
  - `extract.image_patterns` (raw): use fixed post-contrast phases `0001..0003` only.
  - `extract.aggregate_phase_features`: enabled with `phase_aggregate_stats=mean` so raw-phase features are collapsed to phase-blind summaries.
  - `kinetic generation`: use fixed-window maps computed from post phases `1,2,3` only (`--fixed-post-phase-indices 1,2,3`) to remove phase-count dependence in `AUC/slope_out/wash_out`.
  - `extract.image_patterns` (maps): use fixed-window kinetic + subtraction maps from dedicated directory.
  - `train.site_filter`: fixed `ISPY2`.
  - `train.include_subtype`: fixed `false`.
  - `shape handling`: no shape-focused variants in this run.

#### Result Table (ranked)
| Rank | Experiment | CV AUC mean | CV AUC std | Test AUC | n_features |
|---|---|---:|---:|---:|---:|
| 1 | `<pending>` | `<pending>` | `<pending>` | `<pending>` | `<pending>` |
| 2 | `<pending>` | `<pending>` | `<pending>` | `<pending>` | `<pending>` |

#### Outcome / Takeaways
- Pending run completion.
- Decision rule:
  - `promote` if CV mean improves with comparable or lower std and test AUC does not degrade materially.
  - `drop` if no improvement over baseline after accounting for CV std.

#### Next Action
- Submit focused temporal-enrichment sweep on burst QoS (`03:30:00`) using cached extraction/checkpoint reuse.

## Night-Watch Cycles

### Cycle 2026-02-25 23:38 CST
#### Results
- Queue state (amy radiomics): `running=1`, `pending=0` at cycle start.
- Active job: `729263` (`rad_ispy2_p1`) running on burst/general.
- Log evidence:
  - Fixed-window kinetic generation confirmed: `--fixed-post-phase-indices 1,2,3`.
  - Progress reached ~`360/980` patients in map generation during this cycle.
  - No extraction/training crash observed in current run yet.
- Historical summary files (`experiment_results.csv`, `experiment_averages.csv`) unchanged; they still reflect legacy broad sweeps and are not the phase-blind P1 readout.

#### Hypothesis Update
- Prior concern confirmed and addressed: variable timepoint count can leak through raw phase features and through kinetics derived from variable late phases.
- Updated working hypothesis:
  - Phase-count-blind construction (raw phase aggregation + fixed-window kinetics) is required before judging whether temporal signal improves ISPY2 imaging-only performance.

#### Next Experiments
- Immediate: let `729263` complete and evaluate:
  - `ispy2_p1_phaseblind_all_kbest50`
  - `ispy2_p1_phaseblind_all_mrmr50`
  - `ispy2_p1_phaseblind_rawonly_kbest50`
  - `ispy2_p1_phaseblind_kinsubonly_kbest50`
- Decision gate after completion:
  - promote if CV mean improves over baseline (`0.5895`) with comparable/lower std (`~0.047`) and no meaningful test AUC drop.
- If no lift:
  - run P2 extraction-parameter sweep (`binWidth` and `force2D`) under the same phase-blind constraints.

#### Actions Taken
- Read latest evidence:
  - `experiment_results.csv`
  - `experiment_averages.csv`
  - `squeue`/`sacct` state
  - active Slurm logs for `729263`
- Implemented and activated confound controls:
  - raw phase aggregation in extraction (`--aggregate-phase-features`)
  - fixed-window kinetic map generation (`--fixed-post-phase-indices 1,2,3`)
- Superseded prior intermediate runs and relaunched clean run:
  - cancelled `729255`, `729259`, `729261`
  - active run: `729263`

### Cycle 2026-02-25 23:46:59 CST
#### Results
- `squeue/sacct`: `729263` still `RUNNING` (no new completed training metrics yet).
- Latest run log confirms:
  - fixed-window kinetic generation completed successfully (`980 success, 0 errors`),
  - required fixed-window kinetic/subtraction maps validated for all ISPY2 patients,
  - extraction started with checkpointing (`train checkpoint rows currently 23/680`).
- Existing completed ISPY2 imaging-only benchmark remains:
  - `ispy2_imgonly_mrmr50`: CV AUC `0.5895 ± 0.0470`, Test AUC `0.6422`.

#### Insights Learned
1. Feature-selection gains are below noise floor:
   - `mrmr50` vs `baseline` CV delta = `+0.0015`, which is only `0.03x` fold std (`0.0470`).
   - Causal implication: selector choice is not the primary driver; current signal ceiling is dominated by representation/data quality, not k-best vs mRMR tuning.
2. Test ranking spread is tiny relative to CV uncertainty:
   - Top-3 test AUC spread (`baseline`, `kbest20`, `mrmr50`) is `0.0013`, while CV std is ~`0.047–0.049`.
   - Causal implication: small test differences are unstable; decisions should be made on CV mean with std, not single test ordering.
3. Best model shows a large test-vs-CV gap:
   - `mrmr50`: Test AUC `0.6422` vs CV mean `0.5895` (gap `+0.0527`).
   - Causal implication: single split likely carries favorable sampling noise; we should expect lower generalizable performance unless representation improves.

#### Hypothesis Update
- Top hypothesis remains: performance is limited by temporal feature representation quality and acquisition-related confounding, not by classifier/selector micro-tuning.
- Current `729263` run is the first clean test of that hypothesis under strict phase-count blinding (fixed raw phases + phase aggregation + fixed-window kinetics).

#### Next Experiments
- PASS (this cycle): no new submission until `729263` completes.
  - Reason: all free decision value is in the in-flight, confound-controlled run; launching additional variants now would consume slots before the key hypothesis readout is available.
- Immediately after `729263` completion:
  - If CV mean materially improves with comparable/lower std, promote the winning phase-blind representation and rerun one confirmation replicate.
  - If no lift, run ISPY2-only P2 sweep (binWidth and force2D) under the same phase-blind constraints.

#### Actions Taken
- Read latest evidence:
  - `experiment_results.csv`
  - `experiment_averages.csv`
  - `squeue` / `sacct` state
  - recent Slurm logs for job `729263`
- Computed quantitative effect-size/uncertainty comparisons from completed ISPY2 runs for this cycle’s decision logic.
- Updated this notebook cycle entry before proposing any new experiment submissions.

### Cycle 2026-02-25 23:55:01 CST
#### Results
- Active run status:
  - `729263` (`rad_ispy2_p1`) is still `RUNNING` on `general/burst`.
  - Fixed-window kinetic generation is complete and clean: `980 success, 0 errors`.
  - Extraction has started with checkpointing and is progressing (`train checkpoint rows` increasing; no crash signal in log).
- No new completed phase-blind training metrics yet (`ispy2_p1_phaseblind_sweep_summary.csv` not present yet).
- Current best completed ISPY2 imaging-only reference remains:
  - `ispy2_imgonly_mrmr50`: Test AUC `0.6422`, CV AUC `0.5895 ± 0.0470`.
- Standard slot-floor status after this cycle's submissions:
  - `8` active `general/qos=normal` ISPY2 radiomics jobs (`729286`–`729293`).

#### Insights Learned
1. Selector deltas remain negligible versus uncertainty:
   - `mrmr50` vs `baseline` CV gain = `+0.0015`, only `~0.03x` CV std.
   - Causal implication: changing selector alone is not moving true signal; feature representation is the bottleneck.
2. Apparent test differences are much smaller than fold variance:
   - top ISPY2 test-AUC spread among strongest runs is ~`0.001–0.002`, while CV std is ~`0.047–0.049`.
   - Causal implication: single-split test ordering is unstable; we need larger representation shifts, not micro-tuning.
3. The test-vs-CV gap is still large in the best prior run:
   - `0.6422 - 0.5895 = +0.0527`.
   - Causal implication: likely optimistic split effect; improving generalizable signal requires deconfounded temporal representation.

#### Hypothesis Update
- Top hypothesis is unchanged and now sharpened:
  - the dominant limiter is extraction/representation (temporal encoding + acquisition harmonization), not downstream model family or selector choice.
- Therefore, the highest-value next steps are ISPY2-only extraction-parameter experiments under the phase-blind constraints already implemented.

#### Next Experiments
- Restore standard-capacity floor with 8 ISPY2 phase-blind P2 jobs (general, qos=normal), each testing extraction settings directly tied to the top hypothesis:
  - `binWidth ∈ {5,10,15,25}` × `force2D ∈ {off,on}`.
- Keep model/training fixed (logistic elastic-net, same train settings) so outcome differences are attributable to extraction choices.
- Run these jobs independently of `729263` completion now that fixed-window kinetic assets already exist.

#### Actions Taken
- Read latest evidence this cycle:
  - `experiment_results.csv`
  - `experiment_averages.csv`
  - `squeue` / `sacct` state
  - recent `729263` logs
- Wrote this cycle’s quantitative insights and hypothesis update before submitting any new experiment jobs.
- Added standard non-burst P2 runner:
  - `scripts/slurm_general_ispy2_phaseblind_p2_single.sh`
- Submitted 8 ISPY2 phase-blind P2 jobs to restore standard floor:
  - initial set: `729269`–`729276` (dependency-gated),
  - replaced with ungated set for immediate scheduling: `729286`, `729287`, `729288`, `729289`, `729290`, `729291`, `729292`, `729293`.
- Added burst safety-net resume job for P1:
  - `729294` (`afterany:729263`) to auto-resume if `729263` times out mid-pipeline.

### Cycle 2026-02-26 00:25:01 CST
#### Results
- Queue/slot status:
  - Standard floor is satisfied (`8/8`) with ISPY2 P2 jobs active on `general/qos=normal`.
  - Running now: `729286`, `729287`, `729288`, `729289`, `729290`, `729291`, `729292`.
  - Pending now: `729293` (QOS job-limit gate).
  - Burst P1 remains active: `729263`; burst resume safety-net remains pending: `729294`.
- Run health:
  - No crashes/failures in active P2 jobs.
  - All running jobs reached extraction phase; checkpoint row counts are increasing for each setting.
  - No completed P2 training metrics yet (`completed_training=0`), so no new AUC readout this cycle.

#### Insights Learned
1. Latest completed effect-size signal remains tiny vs variance:
   - best completed selector delta (`mrmr50` vs baseline) is still only `+0.0015` CV AUC against `~0.047` fold std.
   - Causal implication: selector changes are unlikely to move performance materially without extraction-level improvements.
2. Legacy force2D effect is moderate but still sub-std:
   - from historical aggregates, `force2D=false` outperforms `true` by ~`0.013` mean test AUC.
   - Relative to current fold std (`~0.047`), this is ~`0.28x` std: plausible but uncertain.
   - Causal implication: force2D is worth testing, but only under controlled phase-blind extraction (which the current P2 sweep does).
3. Current cycle’s dominant uncertainty is unresolved because no P2 jobs have reached training outputs yet:
   - Causal implication: highest-value action is to keep slots fully utilized and avoid churn until first P2 metrics land.

#### Hypothesis Update
- Top hypothesis remains unchanged:
  - most recoverable performance is in extraction/representation choices (binWidth/force2D under phase-blind temporal construction), not in classifier/selector micro-tuning.
- Current 8-way P2 sweep is directly aligned with this hypothesis and should provide the first high-decision-value readout.

#### Next Experiments
- PASS on additional submissions this cycle.
  - Reason: standard slots are fully saturated (`8/8`) with the highest-priority ISPY2 P2 settings already running/pending; extra submissions would not improve near-term decision value.
- Immediate next step when first P2 metrics appear:
  - rank settings by `CV mean`, then `CV std`, then `AUC_test`;
  - trigger robustness replicate + compartment ablations for the top two settings.

#### Actions Taken
- Read latest evidence:
  - `experiment_results.csv`, `experiment_averages.csv`, `squeue`, `sacct`, recent P1/P2 logs.
- Verified all active P2 jobs are healthy and progressing (checkpoint growth, no failure signatures).
- Verified slot-floor compliance for standard queue remains satisfied at `8/8`.
NUDGE_SUBMIT_OK

### Cycle 2026-02-26 00:42:15 CST
#### Results
- Queue/slot status at cycle check:
  - Standard `general/qos=normal` active ISPY2 jobs: `8/8` (all running: `729286`–`729293`).
  - Burst jobs: `729263` running, `729294` pending on dependency (`afterany:729263`).
- Latest evidence files:
  - `experiment_results.csv` and `experiment_averages.csv` unchanged this cycle (no new completed training rows yet).
  - `/net/projects2/vanguard/annawoodard/radiomics_baseline/outputs/ispy2_imgonly_sweep_summary.csv` still shows best completed run: `ispy2_imgonly_mrmr50` with `Test AUC=0.6422`, `CV AUC=0.5895 ± 0.0470`.
- In-flight extraction progress (checkpoint rows):
  - P1 (`729263`): `327/680` train rows complete.
  - P2 train rows by setting: `73` to `182` of `680` complete across the 8 settings.
- Run health:
  - No active failures/crashes detected in `sacct/squeue`.
  - Stdout/err logs show ongoing extraction only; no failure signatures.

#### Insights Learned
1. Completed-model effect size remains below fold uncertainty:
   - `mrmr50` vs baseline CV gain is still `+0.0015`, versus CV std `~0.047`.
   - Causal implication: selector choice is not the dominant limiter; extraction representation remains the primary lever.
2. Early P2 throughput differences across extraction settings are small relative to mean throughput:
   - Current train-row rates are `5.85` to `6.64` rows/min (mean `~6.35`), i.e. spread `~0.79` rows/min (`~12%` of mean).
   - Causal implication: none of the tested `binWidth/force2D` settings appear operationally pathological; all are viable to completion for fair comparison.
3. P1 burst walltime risk is reduced but still non-zero:
   - At `66.1` min elapsed, P1 train ETA is `~71.4` min (from observed `4.95` rows/min), leaving `~72.5` min before the `3.5h` limit for test extraction + training.
   - Causal implication: `729263` may finish in one shot if downstream stages are fast; if not, the queued dependency resume (`729294`) should preserve progress via checkpoint cache.

#### Hypothesis Update
- Top hypothesis is unchanged: meaningful AUC lift (if present) will come from confound-controlled extraction/representation choices, not selector micro-tuning.
- This cycle strengthens confidence that all 8 P2 settings are progressing well enough to produce decision-grade comparisons soon.

#### Next Experiments
- PASS on new submissions this cycle.
  - Reason: standard capacity is already saturated (`8/8`) with the highest-value ISPY2 P2 sweep, and no completed P2 metrics are available yet to justify branching.
- Immediate trigger when first P2 metrics land:
  - rank by `CV AUC mean` then `CV AUC std`,
  - launch one replicate + compartment ablations (`raw-only` vs `kin/sub-only`) for the top setting.

#### Actions Taken
- Read required evidence this cycle:
  - `experiment_results.csv`, `experiment_averages.csv`, `squeue`, `sacct`, active Slurm logs.
- Verified queue state and slot-floor compliance (`8/8` standard active).
- Verified heavy outputs are under `/net/projects2/vanguard/annawoodard/radiomics_baseline` (not home).
- Quantified checkpoint progress and per-job throughput for P1/P2 to assess timeout/finish risk.

### Cycle 2026-02-26 00:44:29 CST
#### Results
- Queue/slot status:
  - Standard ISPY2 jobs on `general/qos=normal`: `8/8` active and running (`729286`–`729293`).
  - Burst jobs: `729263` running (`rad_ispy2_p1`), `729294` pending on dependency (`afterany:729263`).
- Latest evidence files read this cycle:
  - `experiment_results.csv` tail unchanged.
  - `experiment_averages.csv` tail unchanged.
- Completion state:
  - No new `metrics.json` for P1/P2 yet; all active runs are still in extraction.
- Checkpoint progress:
  - P1 extraction (`729263`): `341/680` train rows complete.
  - P2 extraction train rows by setting: `86` to `199` of `680` complete; test rows not started yet.
- Failure scan:
  - No `Traceback`, `[ERROR]`, OOM, or runtime-fatal signatures in active P1/P2 logs.

#### Insights learned
1. Completed-model performance delta is still far below uncertainty:
   - Best completed delta remains `mrmr50 - baseline = +0.0015` CV AUC with fold std `~0.047`.
   - Causal implication: selector tuning is not the main bottleneck; we should keep effort on extraction/representation.
2. In-flight P2 settings show low runtime variance, suggesting fair apples-to-apples comparison:
   - Current P2 throughput mean is `6.43 ± 0.20` train rows/min (min `5.97`, max `6.67`).
   - Relative spread is small (`~3.1%` CV for throughput).
   - Causal implication: none of the `binWidth/force2D` settings appears operationally pathological; final performance differences are likely methodological, not just runtime artifacts.
3. P1 remains slower but still plausible within burst walltime when combined with resume safety-net:
   - P1 throughput `4.98` rows/min vs P2 mean `6.43` rows/min (`~22.5%` slower).
   - P1 train ETA is `~68.1` min at current rate; dependency resume job (`729294`) protects progress if walltime truncates downstream stages.
   - Causal implication: no resubmission/cancellation needed now; current checkpoint strategy is functioning as intended.

#### Hypothesis update
- Top hypothesis is unchanged: meaningful AUC gain will come from confound-controlled feature representation choices (phase-blind temporal encoding / extraction params), not from further selector micro-tuning.
- Current 8-way ISPY2 P2 sweep is still the highest decision-value test of that hypothesis.

#### Next experiments
- PASS this cycle for new submissions.
  - Reason: standard slot floor is fully saturated (`8/8`) with highest-priority ISPY2 P2 runs, and no new completed metrics exist yet to justify branching.
- Trigger condition for next submit:
  - when first P2 `metrics.json` lands, rank by `CV mean`, then `CV std`, then `test AUC`, and immediately launch a focused replicate + compartment ablation for the top setting.

#### Actions taken
- Read required evidence for this cycle:
  - `experiment_results.csv`, `experiment_averages.csv`, `squeue`, `sacct`, active Slurm stdout/err logs.
- Quantified extraction progress and per-setting throughput/ETA from checkpoint artifacts.
- Verified no active crash signatures and no action-required failures.

### Cycle 2026-02-26 01:15:22 CST
#### Results
- Queue/slot status:
  - Standard floor is satisfied: `8/8` active ISPY2 jobs on `general/qos=normal` (`729286`–`729293`, all running).
  - Burst jobs: `729263` running, `729294` pending on dependency (`afterany:729263`).
- Latest evidence files:
  - `experiment_results.csv`: unchanged this cycle.
  - `experiment_averages.csv`: unchanged this cycle.
- Completion state:
  - No new P1/P2 `metrics.json` files yet; all active runs remain in extraction.
- Extraction checkpoint state:
  - P1 extraction (`ispy2_temporal_phaseblind_raw123_kinsubfixed123`): `500/680` train rows complete, `0/300` test rows complete.
  - P2 extraction train-row range across 8 settings: `275` to `356` of `680`; test rows still `0` for all settings.
- Log health:
  - No fatal signatures (`Traceback`, `[ERROR]`, OOM, runtime exceptions) in active P1/P2 logs.

#### Insights learned
1. Latest completed performance effect size remains much smaller than fold uncertainty:
   - Best completed CV delta is still `+0.0015` (`mrmr50` vs baseline) against `CV std ~0.047`.
   - Ratio is `~0.03x` one std.
   - Causal implication: selector differences are noise-scale; extracting stronger signal remains the main path to improvement.
2. P2 sweep runtime variance is low, so settings are being compared fairly:
   - Current throughput across P2 settings is `5.78 ± 0.12` train rows/min (`n=8`; min `5.56`, max `6.02`).
   - Estimated total extraction ETA is `113.9 ± 4.1` min from current point, with `~299–314` min walltime headroom per job.
   - Causal implication: P2 jobs are unlikely to diverge due to runtime instability/timeouts; resulting AUC differences should reflect parameter effects rather than operational artifacts.
3. P1 first burst run is likely extraction-bound near walltime, validating resume design:
   - P1 elapsed `99.8` min, headroom `110.2` min, but extraction ETA `~95.8` min at current rate (`5.01` rows/min).
   - Remaining margin after extraction is only `~14.4` min for finalization/training in this run.
   - Causal implication: a single 3.5h burst slice may not finish end-to-end, so dependency resume (`729294`) is necessary and should preserve work via checkpoint cache.

#### Hypothesis update
- Top hypothesis is unchanged: performance lift will come from confound-controlled feature representation/extraction (phase-blind temporal encoding + extraction params), not from additional selector/model micro-tuning.
- Current in-flight P2 sweep remains the highest-decision-value test of this hypothesis.

#### Next experiments
- PASS for new submissions this cycle.
  - Reason: standard capacity is fully utilized (`8/8`) with ISPY2-first high-value jobs, no failures require replacement, and no completed P2 metrics yet exist to justify branching.
- Immediate trigger for next action:
  - when first P2 metrics appear, rank by `CV AUC mean`, then `CV AUC std`, then `test AUC`, and launch replicate + compartment ablation for the top setting.

#### Actions taken
- Read required evidence this cycle:
  - `experiment_results.csv`, `experiment_averages.csv`, `squeue`, `sacct`, and active Slurm logs/checkpoints.
- Quantified run progress, throughput variance, ETA, and walltime headroom for P1/P2.
- Verified no active crashes and confirmed slot-floor compliance (`8/8` standard active).

### Cycle 2026-02-26 01:45:20 CST
#### Results
- Queue/slot status:
  - Standard floor remains satisfied: `8/8` active ISPY2 jobs on `general/qos=normal` (`729286`–`729293`, all running).
  - Burst jobs: `729263` running (`rad_ispy2_p1`), `729294` pending on dependency (`afterany:729263`).
- Latest evidence files:
  - `experiment_results.csv` unchanged this cycle.
  - `experiment_averages.csv` unchanged this cycle.
- Completion state:
  - No new P1/P2 `metrics.json` yet; all active jobs are still in extraction.
- Checkpoint progress:
  - P1 extraction (`ispy2_temporal_phaseblind_raw123_kinsubfixed123`): `623/980` complete (`622 train`, `0 test` checkpoint rows plus one in-flight row captured between probes).
  - P2 extraction progress by setting: `416` to `513` of `980` complete; all still in train split.
- Failure scan:
  - No fatal signatures in active logs (`Traceback`, `[ERROR]`, OOM, uncaught exceptions absent).

#### Insights learned
1. Completed-model effect size is still below uncertainty by an order of magnitude:
   - Best completed CV delta remains `+0.0015` (`mrmr50` vs baseline) vs fold std `~0.047`.
   - Effect/std ratio is `~0.03`.
   - Causal implication: selector changes remain noise-scale; extraction representation is still the dominant lever.
2. P2 sweep has become even tighter operationally, increasing trust in eventual comparative readout:
   - P2 throughput is `5.53 ± 0.08` rows/min (`n=8`; min `5.39`, max `5.66`).
   - Estimated extraction slack before walltime is strongly positive across all settings (`+178.1` to `+186.7` min).
   - Causal implication: P2 jobs should complete extraction comfortably and comparably; upcoming performance differences are more likely methodological than runtime confounded.
3. P1 first burst slice is now near extraction-walltime boundary:
   - P1 done `623/980`, rate `4.82` rows/min, extraction ETA `~74.1` min.
   - Remaining walltime headroom is `~80.6` min, leaving only `~6.5` min slack before walltime for non-extraction stages.
   - Causal implication: timeout before full end-to-end completion is plausible; dependency resume job (`729294`) is necessary and expected to be used.

#### Hypothesis update
- Core hypothesis unchanged: signal recovery depends on confound-controlled extraction/representation choices, not additional selector/model micro-tuning.
- Current ISPY2-first P2 sweep is still the highest decision-value experiment set in flight.

#### Next experiments
- PASS on new submissions this cycle.
  - Reason: standard floor is fully saturated (`8/8`) with highest-value ISPY2 runs, no crash/replacement need, and no completed new metrics yet to guide branching.
- Immediate trigger for follow-up submissions:
  - when first P2 `metrics.json` appears, rank by `CV AUC mean`, then `CV AUC std`, then `test AUC`, and launch replicate + raw-only/kin-sub-only ablations on the top setting.

#### Actions taken
- Read required evidence this cycle:
  - `experiment_results.csv`, `experiment_averages.csv`, `squeue`, `sacct`, active Slurm logs, and checkpoint artifacts.
- Quantified progress, throughput variance, ETA, and walltime slack for P1/P2.
- Verified no active job failures and confirmed slot-floor compliance.

### Cycle 2026-02-26 02:15:15 CST
#### Results
- Queue/slot status:
  - Standard floor remains met: `8/8` active ISPY2 jobs on `general/qos=normal` (`729286`–`729293`, all running).
  - Burst jobs: `729263` running (`rad_ispy2_p1`), `729294` pending on dependency (`afterany:729263`).
- Evidence files:
  - `experiment_results.csv` unchanged this cycle.
  - `experiment_averages.csv` unchanged this cycle.
- Completion state:
  - No new P1/P2 `metrics.json` files yet.
- Checkpoint progress:
  - P1 extraction (`ispy2_temporal_phaseblind_raw123_kinsubfixed123`): `678/980` done.
  - P2 extraction progress by setting: `567` to `646` of `980` done.
- Log health:
  - No fatal signatures in active logs (`Traceback`, `[ERROR]`, OOM, uncaught exceptions absent).

#### Insights learned
1. Completed-model effect size remains far below fold uncertainty:
   - Best completed CV delta still `+0.0015` (`mrmr50` vs baseline) versus `CV std ~0.047`.
   - Effect/std ratio remains `~0.03`.
   - Causal implication: downstream selector differences are not the bottleneck; extraction representation remains the best lever.
2. P2 jobs remain tightly matched and safely within walltime:
   - P2 throughput: `5.25 ± 0.09` rows/min (`n=8`; min `5.10`, max `5.39`).
   - P2 extraction slack before walltime: `+167.7` to `+178.1` min across settings.
   - Causal implication: P2 comparative results should be method-driven (not runtime-failure-driven), and all 8 settings are likely to produce complete outputs.
3. P1 first burst slice is now likely to timeout before finishing extraction:
   - P1 done `678/980`, rate `4.25` rows/min, extraction ETA `~71.0` min.
   - Remaining walltime headroom is `~50.7` min (`slack = -20.3` min).
   - Causal implication: `729263` likely ends before extraction completes; dependency resume `729294` is expected to carry the run across completion via checkpoint reuse.

#### Hypothesis update
- Core hypothesis unchanged: meaningful performance lift depends on confound-controlled feature representation/extraction, not further selector/model micro-tuning.
- The in-flight 8-way ISPY2 P2 sweep remains the highest-value test of that hypothesis.

#### Next experiments
- PASS on additional submissions this cycle.
  - Reason: standard capacity is fully saturated (`8/8`), no failed jobs require replacement, and no completed new metrics exist yet to justify branching.
- Immediate trigger for next actions:
  - when first P2 `metrics.json` appears, rank by `CV AUC mean`, then `CV AUC std`, then `test AUC`; submit replicate + compartment ablations for the top setting.

#### Actions taken
- Read required evidence this cycle:
  - `experiment_results.csv`, `experiment_averages.csv`, `squeue`, `sacct`, recent Slurm logs, and checkpoint artifacts.
- Quantified progress, throughput variance, extraction ETA, and walltime slack for P1 and all P2 settings.
- Verified no active job failures and maintained slot-floor compliance.

### Cycle 2026-02-26 02:45:21 CST
#### Results
- Queue/slot status:
  - Standard floor still satisfied: `8/8` active ISPY2 jobs on `general/qos=normal` (`729286`–`729293`, all running).
  - Burst jobs: `729263` running (`rad_ispy2_p1`), `729294` pending on dependency.
- Evidence files:
  - `experiment_results.csv` unchanged this cycle.
  - `experiment_averages.csv` unchanged this cycle.
- Completion state:
  - No new P1/P2 `metrics.json` files yet.
- Checkpoint progress:
  - P1 extraction: `720/980` done (`680` train + `40` test).
  - P2 extraction: all 8 settings at `678/980` done (train nearly complete, no test rows yet).
- Failure scan:
  - No fatal signatures in logs (`Traceback`, `[ERROR]`, OOM, uncaught exceptions absent).

#### Insights learned
1. Completed-model effect size remains much smaller than fold uncertainty:
   - Best completed CV delta remains `+0.0015` (`mrmr50` vs baseline) vs `CV std ~0.047`.
   - Effect/std ratio remains `~0.03`.
   - Causal implication: selector/model micro-tuning remains low leverage; extraction representation is still the dominant improvement axis.
2. P2 jobs are now synchronized at the same near-complete train checkpoint:
   - All 8 settings are exactly `678/980` complete with no extraction failures recorded.
   - Throughput estimate at this point is `4.66 ± 0.19` rows/min, with extraction slack still strongly positive (`+141.3` to `+163.6` min).
   - Causal implication: this looks like a shared late-train bottleneck (same remaining patients) rather than config-specific instability; final comparative readout should remain fair once they clear this tail.
3. P1 first burst slice is now certain to timeout before extraction completion:
   - P1 done `720/980`, extraction ETA `~68.6` min, walltime headroom `~20.0` min (`slack = -48.6` min).
   - Causal implication: dependency resume is mandatory for continuity; without additional chaining, end-to-end completion could still be vulnerable if second slice is interrupted.

#### Hypothesis update
- Core hypothesis unchanged: performance lift depends on confound-controlled extraction/representation choices, not additional selector/model micro-tuning.
- The in-flight P2 sweep remains the highest decision-value ISPY2 experiment set.

#### Next experiments
- No new branch experiments this cycle (await first completed P2 metrics).
- Operational continuity action:
  - submit one additional dependency-chained burst resume safety-net after job `729294` for the same P1 pipeline, to guarantee checkpoint-driven continuation if needed.

#### Actions taken
- Read required evidence this cycle:
  - `experiment_results.csv`, `experiment_averages.csv`, `squeue`, `sacct`, recent Slurm logs, and checkpoint artifacts.
- Quantified extraction progress, throughput variance, and walltime slack for P1/P2.
- Verified no active job failures and kept standard floor satisfied (`8/8`).
- Addendum (02:46 CST): submitted additional burst resume safety-net `729392` with dependency `afterany:729294` using `scripts/slurm_burst_ispy2_temporal_enriched_sweep.sh`.

### Cycle 2026-02-26 03:15:14 CST
#### Results
- Queue/slot status:
  - Standard floor remains satisfied: `8/8` active ISPY2 jobs on `general/qos=normal` (`729286`–`729293`, all running).
  - Burst chain state:
    - `729263` hit `TIMEOUT` at `03:30:10`.
    - `729294` started via dependency and is running.
    - `729392` (old safety-net) was replaced after script fix; new chained job is `729403` (`afterany:729294`, pending).
- Evidence files:
  - `experiment_results.csv` unchanged this cycle.
  - `experiment_averages.csv` unchanged this cycle.
  - `/net/projects2/vanguard/annawoodard/radiomics_baseline/outputs/ispy2_imgonly_sweep_summary.csv` unchanged (`mrmr50`: `CV 0.5895 ± 0.0470`, `Test 0.6422`).
- Completion state:
  - No new P1/P2 `metrics.json` files yet.
- Checkpoint state (latest):
  - P2 runs progressed into test extraction for several settings (e.g., `bin5_f2d1: train=680, test=62`; `bin5_f2d0: train=680, test=48`; `bin10_f2d1: train=680, test=45`), with others still completing train tail.
  - P1 resume run (`729294`) initially restarted with a fresh checkpoint manifest and low completed rows (`~61` at cycle check), instead of continuing from prior `~720`.

#### Insights learned
1. Completed-model effect size remains much smaller than uncertainty:
   - Best completed CV delta is still `+0.0015` (`mrmr50` vs baseline) against fold std `~0.047` (ratio `~0.03x`).
   - Causal implication: selector/model micro-tuning remains low-yield; extraction representation remains the main lever.
2. P2 jobs are now heterogeneous by split-phase but still operationally healthy:
   - Done counts range `678` to `742` of `980`; mean throughput `3.98 ± 0.07` rows/min.
   - Estimated extraction slack remains positive for all settings (`+104.9` to `+119.7` min).
   - Causal implication: no immediate timeout risk for P2; expected to complete extraction and proceed to training without replacement submissions.
3. Root-cause found for failed resume reuse in P1:
   - `729263` timed out after reaching deep extraction progress, but `729294` recreated checkpoint manifest (`created_at_utc=2026-02-26T09:07:14Z`) and restarted near zero rows.
   - Fingerprint payload includes `splits`/`params` path metadata (including `mtime_ns`), and run scripts rewrote generated split/params files on each submission.
   - Causal implication: rewrite-induced metadata drift invalidates fingerprint and clears cached checkpoint rows; true resumability requires stable generated file metadata when content is unchanged.

#### Hypothesis update
- Core hypothesis is unchanged for model performance: signal lift should come from confound-controlled extraction/representation choices.
- New operational sub-hypothesis confirmed: checkpoint reuse robustness depends on not rewriting unchanged generated split/params files.

#### Next experiments
- No new branch experiments this cycle (await first completed P2 metrics for decision branching).
- Continue current ISPY2 runs to completion with fixed cache-stable scripts and chained burst fallback:
  - active: `729294`.
  - pending fallback: `729403`.

#### Actions taken
- Read required evidence this cycle:
  - `experiment_results.csv`, `experiment_averages.csv`, `squeue`, `sacct`, recent logs, checkpoint manifests.
- Implemented fix for checkpoint invalidation due unnecessary file rewrites:
  - `scripts/slurm_burst_ispy2_temporal_enriched_sweep.sh`: write `splits_ispy2_only.csv` only if content changed.
  - `scripts/slurm_general_ispy2_phaseblind_p2_single.sh`: write `splits_ispy2_only.csv` and generated params only if content changed.
- Re-chained burst safety-net to use patched script snapshot:
  - cancelled `729392` and submitted `729403` with `--dependency=afterany:729294`.
- Verified standard floor remains satisfied and no active crash signatures in radiomics jobs.

### Cycle 2026-02-26 03:45:23 CST
#### Results
- Queue/slot status:
  - Standard floor remains satisfied: `8/8` active ISPY2 jobs on `general/qos=normal` (`729286`–`729293`, all running).
  - Burst chain status:
    - `729263`: `TIMEOUT` (completed as expected).
    - `729294`: running (`qos=burst`) as resume slice.
    - `729403`: pending dependency (`afterany:729294`) as patched fallback.
- Evidence files:
  - `experiment_results.csv` unchanged this cycle.
  - `experiment_averages.csv` unchanged this cycle.
  - `ispy2_imgonly_sweep_summary.csv` unchanged (`mrmr50`: `CV 0.5895 ± 0.0470`, `Test 0.6422`).
- Completion state:
  - No new P1/P2 `metrics.json` yet.
- Checkpoint progress:
  - P2 settings are all in test extraction phase now:
    - train rows complete: `680/680` for all 8 settings,
    - test rows complete range: `108` to `228` (of `300`).
  - P1 resume checkpoint currently: `241 train`, `0 test` (`241/980`) under the new manifest created by `729294`.
- Failure scan:
  - No fatal signatures found in radiomics P1/P2 logs (`Traceback`, `[ERROR]`, OOM, uncaught exceptions absent).

#### Insights learned
1. Completed-model effect size remains far below CV uncertainty:
   - Best completed CV delta is still `+0.0015` (`mrmr50` vs baseline) vs `CV std ~0.047` (`~0.03x std`).
   - Causal implication: selector/model adjustments remain low-leverage relative to representation-level changes.
2. P2 jobs are converging to completion with low runtime dispersion and ample slack:
   - Current throughput mean is `4.11 ± 0.10` rows/min across the 8 settings.
   - Remaining extraction ETA mean is `33.4 ± 10.5` min; minimum walltime slack is still `+110.8` min.
   - Causal implication: P2 extraction should finish without timeout-driven bias; next decision-quality signal should come from training metrics, not operational variance.
3. P1 resume slice is operationally healthier than the first slice but still dependent on chaining for robustness:
   - At current pace (`6.14` rows/min), extraction ETA is `~120.4` min with `~170.8` min headroom in `729294` (`+50.4` min extraction slack).
   - Because training still follows extraction and prior manifest reset already consumed continuity, fallback chaining remains necessary.
   - Causal implication: keep `729403` dependency fallback in place; do not branch new burst probes until P1/P2 produce metrics.

#### Hypothesis update
- Performance hypothesis remains unchanged: meaningful lift should come from confound-controlled extraction/representation choices rather than additional model/selector micro-tuning.
- Operationally, after the script fix, future resume continuity should improve because unchanged split/params files will no longer force fingerprint drift.

#### Next experiments
- PASS on additional submissions this cycle.
  - Reason: standard slots are saturated (`8/8`) with highest-value ISPY2 runs, no failure replacement is required, and no new completed metrics are available to guide branching.
- Trigger for next submit decisions:
  - once first P2 `metrics.json` appears, rank by `CV mean`, then `CV std`, then `test AUC`, and launch focused replicate + compartment ablations for the top setting.

#### Actions taken
- Read required evidence this cycle:
  - `experiment_results.csv`, `experiment_averages.csv`, `squeue`, `sacct`, radiomics logs, and checkpoint directories.
- Quantified extraction progress, throughput variance, ETA, and slack for P1/P2.
- Confirmed no crash signatures and maintained slot-floor compliance.

### Cycle 2026-02-26 04:15:17 CST
#### Results
- Queue/slot status:
  - Standard floor remains satisfied: `8/8` active ISPY2 jobs on `general/qos=normal` (`729286`–`729293`, all running).
  - Burst chain status:
    - `729294` running (`qos=burst`) as resume slice.
    - `729403` pending dependency (`afterany:729294`) as patched fallback.
- Evidence files:
  - `experiment_results.csv` unchanged this cycle.
  - `experiment_averages.csv` unchanged this cycle.
  - `ispy2_imgonly_sweep_summary.csv` unchanged (`mrmr50`: `CV 0.5895 ± 0.0470`, `Test 0.6422`).
- Completion state:
  - No new P1/P2 `metrics.json` files yet.
- Checkpoint progress:
  - P2 extraction is effectively complete for most settings and nearly complete for the remainder:
    - completed rows (`train+test`) range: `944` to `979` of `980`.
    - four settings are at `979/980`; others at `975`, `956`, `944`.
  - P1 resume checkpoint currently: `386 train`, `0 test` (`386/980`) under manifest created in `729294`.
- Failure scan:
  - No fatal signatures in radiomics P1/P2 logs (`Traceback`, `[ERROR]`, OOM, uncaught exceptions absent).

#### Insights learned
1. Completed-model effect size remains much smaller than uncertainty:
   - Best completed CV delta is still `+0.0015` (`mrmr50` vs baseline) vs `CV std ~0.047` (`~0.03x std`).
   - Causal implication: extra model/selector tweaks are still low-yield relative to extraction representation changes.
2. P2 runs are about to transition from extraction to training with large walltime margin:
   - Mean throughput is `4.13 ± 0.08` rows/min.
   - Mean extraction ETA is `~2.1 ± 3.1` min; minimum slack before the 6h limit is still `+118.9` min.
   - Causal implication: P2 jobs should produce training metrics in this runtime window; timeout-driven bias is unlikely.
3. P1 resume slice remains viable but still needs fallback insurance for end-to-end completion:
   - Current P1 resume estimate: extraction ETA `~106.5` min, headroom `~140.8` min (`+34.4` min extraction slack).
   - Causal implication: `729294` may complete extraction but training may still be tight; keeping `729403` dependency fallback is appropriate.

#### Hypothesis update
- Performance hypothesis unchanged: meaningful lift is more likely from confound-controlled extraction/representation choices than from classifier/selector micro-tuning.
- Near-term decision value is now in imminent P2 training metrics; prioritize ranking/replication decisions as soon as they land.

#### Next experiments
- PASS on additional submissions this cycle.
  - Reason: standard slots are fully saturated (`8/8`), no replacement is required, and no new completed metrics are available yet for branching.
- Immediate trigger for next cycle actions:
  - if P2 metrics appear, rank by `CV mean`, then `CV std`, then `test AUC`, and submit replicate + compartment ablations for the top setting.

#### Actions taken
- Read required evidence this cycle:
  - `experiment_results.csv`, `experiment_averages.csv`, `squeue`, `sacct`, radiomics logs, and checkpoint directories.
- Quantified extraction completion, throughput variance, ETA, and slack for P1/P2.
- Verified no active crash signatures and maintained slot-floor compliance (`8/8`).

### Cycle 2026-02-26 04:30:29 CST
#### Results
- Queue/slot status:
  - Standard active count dropped below floor to `7/8` because `729287` completed.
  - Running standard ISPY2 jobs currently: `729286`, `729288`, `729289`, `729290`, `729291`, `729292`, `729293`.
  - Burst chain: `729294` running; `729403` pending dependency fallback.
- Evidence files:
  - `experiment_results.csv` unchanged this cycle.
  - `experiment_averages.csv` unchanged this cycle.
  - `ispy2_imgonly_sweep_summary.csv` unchanged (`mrmr50: CV 0.5895 ± 0.0470, Test 0.6422`).
- New completed P2 metrics available:
  - `ispy2_p2_phaseblind_bin5_f2d1`: `CV 0.6212 ± 0.0370`, `Test 0.6258`.
  - `ispy2_p2_phaseblind_bin5_f2d0`: `CV 0.6195 ± 0.0494`, `Test 0.6053`.
- In-flight extraction state:
  - remaining P2 settings are near extraction completion (`944–979/980` done).
  - P1 resume slice checkpoint: `469/980` done at cycle analysis snapshot.
- Failure scan:
  - No fatal radiomics errors observed (`Traceback`, `[ERROR]`, OOM absent).

#### Insights learned
1. New phase-blind P2 settings improve CV materially over prior best completed ISPY2 imaging baseline, with lower uncertainty:
   - `bin5_f2d1` vs prior `mrmr50`: `CV +0.0318` (`0.6212 - 0.5895`), `CV std -0.0100` (`0.0370 vs 0.0470`).
   - Causal implication: controlled extraction setup is likely improving fold-level generalization stability.
2. However, test AUC does not yet exceed prior optimistic split result:
   - `bin5_f2d1` test `0.6258` vs prior `0.6422` (`-0.0164`).
   - Causal implication: prior top test may have been favorably sampled; stronger CV with lower std suggests we should prioritize reproducibility checks over single-test ranking.
3. At fixed `binWidth=5`, `force2d=true` beats `force2d=false` on both test and stability:
   - Test delta `+0.0206` (`0.6258 - 0.6053`), CV delta `+0.0018`, std improvement `-0.0124`.
   - Causal implication: under current phase-blind setup, `force2d=true` looks like a better default for next local parameter probes.

#### Hypothesis update
- Updated top hypothesis: in this phase-blind pipeline, the best next gains are likely from local extraction tuning around the winning `binWidth≈5, force2d=true` regime and confirmation replicates, not broader model-family changes.

#### Next experiments
- Immediate floor restoration + high-value local search submissions on standard queue:
  - `bin4_f2d1`
  - `bin6_f2d1`
- Rationale:
  - restores required `>=8` standard active jobs now,
  - directly tests sensitivity around the current best setting (`bin5_f2d1`).

#### Actions taken
- Read required evidence this cycle:
  - `experiment_results.csv`, `experiment_averages.csv`, `squeue`, `sacct`, logs, checkpoint directories.
- Quantified new completed P2 performance with CV mean/std and test deltas.
- Determined standard floor violation (`7/8`) and selected immediate replacement experiments.

### Cycle 2026-02-26 04:34:08 CST (04:30 Follow-up)
#### Results
- Standard slot-floor status after corrective submissions:
  - `std_active=8` on `general/qos=normal` (`rad_ispy2_p2`, running+pending), floor restored.
- Floor-restoration submissions made this cycle:
  - `729444` (`bin4_f2d1`),
  - `729445` (`bin6_f2d1`),
  - `729446` (`bin8_f2d1`).
- New completed P2 metrics confirmed:
  - `bin5_f2d1`: `CV 0.6212 ± 0.0370`, `Test 0.6258`.
  - `bin5_f2d0`: `CV 0.6195 ± 0.0494`, `Test 0.6053`.
  - `bin10_f2d1`: `CV 0.6212 ± 0.0370`, `Test 0.6258`.

#### Insights learned
1. `force2d=true` at `binWidth=5` improves both test AUC and fold stability relative to `force2d=false`:
   - test delta `+0.0206` (`0.6258 - 0.6053`),
   - CV std delta `-0.0124` (`0.0370 - 0.0494`).
   - Causal implication: `force2d=true` remains the stronger operating point for next probes.
2. The prior best CV lift over historical baseline remains meaningful while test remains lower:
   - `bin5_f2d1` vs historical `mrmr50`: `CV +0.0318`, `CV std -0.0100`, but test `-0.0164`.
   - Causal implication: we are likely reducing split optimism and improving stability, but need replication before claiming net gain.
3. Potential binWidth-effect bug/insensitivity detected for `force2d=true`:
   - `bin5_f2d1` and `bin10_f2d1` have identical feature file hashes (`train` and `test`) and identical metrics.
   - Causal implication: the current binWidth sweep may have limited decision value until we verify whether binWidth is actually impacting extracted features under this setup.

#### Hypothesis update
- Updated operational hypothesis: `force2d` is currently informative, but `binWidth` may be partially inert in this configuration; confirm this before spending additional slots on dense binWidth sweeps.

#### Next experiments
- Keep current submitted local probes (`bin4/6/8_f2d1`) running because they both restore floor and provide direct evidence on binWidth sensitivity.
- If hashes/metrics remain identical across these probes, pivot next standard jobs from binWidth sweeps to high-value alternatives (top-setting replicate and compartment ablations).

#### Actions taken
- Restored required standard floor from `7/8` to `8/8` with three immediate standard submissions (`729444`, `729445`, `729446`).
- Verified new completed metrics and computed CV/test deltas with std.
- Verified identical feature hashes between `bin5_f2d1` and `bin10_f2d1` to flag potential binWidth insensitivity.

### Cycle 2026-02-26 04:40:01 CST
#### Results
- Queue/slot status at cycle start and follow-up:
  - 04:40 tick showed standard active `6/8` (below floor).
  - Follow-up at 04:42 showed standard active dropped further to `5/8` after `729291` completed.
  - Active standard jobs now: `729292` (`bin25_f2d0`), `729293` (`bin25_f2d1`), `729444` (`bin4_f2d1`), `729445` (`bin6_f2d1`), `729446` (`bin8_f2d1`).
  - Burst chain unchanged: `729294` running (`p1`), `729403` pending dependency fallback.
- New completed P2 metric this cycle:
  - `ispy2_p2_phaseblind_bin15_f2d1`: `CV 0.6212 ± 0.0370`, `Test 0.6258`.
- Updated completed P2 set now includes six runs:
  - `bin5_f2d0`, `bin10_f2d0`, `bin15_f2d0` all at `CV 0.6195 ± 0.0494`, `Test 0.6053`.
  - `bin5_f2d1`, `bin10_f2d1`, `bin15_f2d1` all at `CV 0.6212 ± 0.0370`, `Test 0.6258`.
- Baseline CSVs unchanged this cycle:
  - `experiment_results.csv` unchanged.
  - `experiment_averages.csv` unchanged.

#### Insights learned
1. `force2d=true` remains directionally better than `force2d=false`, with improved stability:
   - Test AUC delta: `+0.0206` (`0.6258 - 0.6053`).
   - CV mean delta: `+0.0018` (`0.6212 - 0.6195`).
   - CV std delta: `-0.0124` (`0.0370 - 0.0494`).
   - Causal implication: geometry handling (2D vs 3D extraction) is a real lever in this pipeline; keep `force2d=true` as default for probes.
2. Across completed bins (`5,10,15`), binWidth effect size is exactly `0.0000` on test/CV/std for each `force2d` branch.
   - Within `force2d=true`: max-min on test, CV mean, CV std all `0.0000`.
   - Within `force2d=false`: max-min on test, CV mean, CV std all `0.0000`.
   - Causal implication: current binWidth sweep is likely low decision value unless extreme values break this invariance.
3. Observed between-branch delta is smaller than fold variability on CV mean but not on test:
   - CV delta (`+0.0018`) is only `~0.05x` of best-branch fold std (`0.0370`).
   - Test delta (`+0.0206`) is meaningful relative to prior branch spread.
   - Causal implication: prioritize experiments that can create larger representation shifts than local bin tweaks.

#### Hypothesis update
- Updated top hypothesis: under phase-blind ISPY2 extraction, moderate binWidth changes are effectively inert, while `force2d` meaningfully changes extracted signal quality/stability.
- Immediate high-value test is to probe extreme binWidth values under `force2d=true` to determine whether invariance is intrinsic (pipeline-level) or only local around `5-15`.

#### Next experiments
- Restore required standard floor from `5/8` to `8/8` by submitting three `general/qos=normal` jobs that directly test the top hypothesis:
  - `bin1_f2d1`
  - `bin50_f2d1`
  - `bin100_f2d1`
- Decision criterion:
  - if these also match existing metrics/hashes, stop spending slots on binWidth and pivot to representation ablations (raw-only vs kinetic/subtraction-only) in standard queue.

#### Actions taken
- Read required evidence this cycle:
  - `experiment_results.csv`, `experiment_averages.csv`, `squeue`, `sacct`, active P1/P2 logs, and checkpoint/metrics artifacts in `/net/projects2/vanguard/annawoodard/radiomics_baseline`.
- Confirmed new completion (`bin15_f2d1`) and recomputed effect sizes with CV mean/std and test deltas.
- Identified current standard floor violation (`5/8`) and selected three immediate hypothesis-testing replacements.

### Cycle 2026-02-26 04:44:30 CST (04:40 Follow-up)
#### Results
- Standard floor restoration completed:
  - Submitted `729451` (`bin1_f2d1`), `729452` (`bin50_f2d1`), `729453` (`bin100_f2d1`) on `general/qos=normal`.
  - Standard active count is now `8/8` (`rad_ispy2_p2`, running+pending).
- Submission validation:
  - each job reused unchanged ISPY2 split and kinetic cache checks,
  - each job wrote a new params file (`pyrad_params_bin{1,50,100}_f2d1.yaml`) and started extraction.

#### Insights learned
1. Extreme-bin probe set is now running and directly targets the current top uncertainty (binWidth invariance).
2. Queue policy compliance is restored immediately without sacrificing hypothesis relevance.

#### Hypothesis update
- Pending: if extreme bins remain identical to `5/10/15`, treat binWidth as operationally inert in this pipeline and pivot the next slots to representation ablations.

#### Next experiments
- No additional submissions this follow-up step; keep slots on current P2 set until first extreme-bin metric lands.

#### Actions taken
- Submitted three standard hypothesis-testing replacements (`729451-729453`).
- Re-checked queue state and confirmed standard floor restored (`8/8`).

### Cycle 2026-02-26 04:50:01 CST
#### Results
- Queue/slot status at cycle read (`04:50:30 CST`):
  - Standard active count = `6/8` (below required floor).
  - Active standard P2 jobs: `729444` (`bin4_f2d1`), `729445` (`bin6_f2d1`), `729446` (`bin8_f2d1`), `729451` (`bin1_f2d1`), `729452` (`bin50_f2d1`), `729453` (`bin100_f2d1`).
  - Burst chain unchanged: `729294` running (`p1`), `729403` pending dependency fallback.
- Evidence files checked this cycle:
  - `experiment_results.csv` unchanged (mtime still `2026-02-25 13:27:45 -0600`).
  - `experiment_averages.csv` unchanged (mtime still `2026-02-25 13:27:45 -0600`).
- New completed P2 metrics this cycle:
  - `ispy2_p2_phaseblind_bin25_f2d0`: `CV 0.6195 ± 0.0494`, `Test 0.6053`.
  - `ispy2_p2_phaseblind_bin25_f2d1`: `CV 0.6212 ± 0.0370`, `Test 0.6258`.
- Completed phase-blind P2 set now has 8 runs (`bin 5/10/15/25` x `force2d {0,1}`), all successful.

#### Insights learned
1. BinWidth remains quantitatively inert across all completed values (`5,10,15,25`) within each `force2d` branch.
   - `force2d=true`: max-min deltas are `0.0000` for `test`, `CV mean`, and `CV std`.
   - `force2d=false`: max-min deltas are `0.0000` for `test`, `CV mean`, and `CV std`.
   - Causal implication: moderate binWidth sweeps are currently not changing model behavior; decision value is low unless extreme values break invariance.
2. `force2d=true` advantage is now replicated across four bin widths with better test AUC and lower variance.
   - Test delta: `+0.0206` (`0.6258 - 0.6053`).
   - CV mean delta: `+0.0018` (`0.6212 - 0.6195`).
   - CV std delta: `-0.0124` (`0.0370 - 0.0494`).
   - Causal implication: keep `force2d=true` as default while searching for stronger levers.
3. Effect size vs uncertainty remains weak on CV mean even when test delta is visible.
   - CV delta `+0.0018` is only `~0.05x` of best-branch fold std (`0.0370`).
   - Causal implication: we still need larger representation shifts (not local bin tweaks) to drive reliable lift.

#### Hypothesis update
- Top hypothesis this cycle: binWidth sensitivity is largely suppressed in the current phase-blind extractor path (either true insensitivity or parameter not impacting extracted features), while `force2d` remains the only consistent extraction lever.
- Therefore, immediate high-value actions are:
  - keep stress-testing binWidth with wider extremes to falsify invariance quickly,
  - then pivot slots to representation ablations if invariance persists.

#### Next experiments
- Restore floor from `6/8` to `8/8` using two additional `general/qos=normal` ISPY2 P2 jobs focused on the active top hypothesis:
  - `bin2_f2d1`
  - `bin200_f2d1`
- Decision criterion:
  - if these still collapse to identical feature hashes/metrics, stop allocating additional standard slots to binWidth and reallocate to representation ablations.

#### Actions taken
- Read required cycle evidence:
  - `experiment_results.csv`, `experiment_averages.csv`, `squeue`, `sacct`, and recent P1/P2 logs.
- Confirmed new completions for `bin25_f2d0/f2d1` and recomputed effect sizes with uncertainty.
- Identified standard-floor violation (`6/8`) and selected two immediate corrective submissions.

### Cycle 2026-02-26 04:52:05 CST (04:50 Follow-up)
#### Results
- Standard floor restoration completed:
  - Submitted `729458` (`bin2_f2d1`) and `729459` (`bin200_f2d1`) on `general/qos=normal`.
  - Standard active count is now `8/8` (`rad_ispy2_p2`, running+pending).
- Submission validation:
  - both jobs reused unchanged ISPY2 split/kinetic caches,
  - both jobs wrote new params files and entered extraction.

#### Insights learned
1. Floor can drop rapidly as completed P2 jobs roll off; immediate replacement submission is required to keep `>=8` standard active.
2. New submissions are aligned with the top hypothesis (binWidth invariance stress-test) while preserving queue compliance.

#### Hypothesis update
- Unchanged from 04:50 cycle: decision pivot remains contingent on whether new extreme-bin runs (`1,2,50,100,200`) diverge from the current identical-metric regime.

#### Next experiments
- No additional submissions in this follow-up; hold slots for currently running stress-test set and reassess on first new metrics.

#### Actions taken
- Submitted two standard P2 jobs (`729458`, `729459`) and verified they started correctly.
- Re-checked queue and confirmed floor restored (`8/8`).

### Cycle 2026-02-26 05:20:01 CST
#### Results
- Queue/slot status (`05:20:15 CST`):
  - Standard active count is `8/8` (compliant).
  - Active standard P2 jobs: `729444` (`bin4_f2d1`), `729445` (`bin6_f2d1`), `729446` (`bin8_f2d1`), `729451` (`bin1_f2d1`), `729452` (`bin50_f2d1`), `729453` (`bin100_f2d1`), `729458` (`bin2_f2d1`), `729459` (`bin200_f2d1`).
  - Burst chain unchanged: `729294` running (`p1`), `729403` pending dependency fallback.
- Completed P2 metrics are unchanged from prior cycle count, now stable at 8 completed runs (`bin 5/10/15/25` x `force2d {0,1}`).
- Baseline CSVs unchanged this cycle:
  - `experiment_results.csv` unchanged.
  - `experiment_averages.csv` unchanged.
- Error scan over active radiomics logs found no fatal signatures (`Traceback`, `[ERROR]`, uncaught exceptions, OOM).

#### Insights learned
1. The binWidth-invariance pattern is now persistent across all completed moderate bins with no measurable effect size.
   - Within each branch (`force2d=false` and `force2d=true`), max-min deltas remain `0.0000` for test AUC, CV mean AUC, and CV std.
   - Causal implication: additional moderate binWidth runs are unlikely to change performance; extreme-bin falsification remains the right immediate test.
2. Running extreme-bin probes have healthy extraction throughput and large walltime slack.
   - Active standard extraction rates: mean `6.88 ± 1.08` rows/min.
   - Representative ETAs to finish extraction are `~67` to `~122` min with remaining walltime `~312` to `~331` min.
   - Causal implication: these jobs should complete extraction comfortably; timeout risk for P2 is low this cycle.
3. P1 burst run is close to the walltime boundary but still viable for extraction completion.
   - `729294`: `678/980` rows, rate `5.05` rows/min, extraction ETA `~59.8` min, remaining walltime `~75.8` min, slack `+16.0` min.
   - Causal implication: extraction may finish, but end-to-end completion remains tight; dependency fallback (`729403`) is still justified.

#### Hypothesis update
- Hypothesis unchanged: current extraction path is strongly sensitive to `force2d`, but largely insensitive to binWidth over tested moderate values; decision value now comes from whether extreme bins (`1,2,50,100,200`) break this invariance.

#### Next experiments
- PASS on additional submissions this cycle.
  - Reason: standard floor is already saturated (`8/8`) with directly relevant extreme-bin tests in progress; extra submissions would only queue-churn without increasing near-term decision value.
- Trigger for next cycle:
  - when first extreme-bin metric lands, compare feature hashes + (`test`, `CV mean`, `CV std`) against the current invariant regime and pivot immediately if still identical.

#### Actions taken
- Read required evidence this cycle:
  - `experiment_results.csv`, `experiment_averages.csv`, `squeue`, `sacct`, active radiomics logs, and extraction checkpoint artifacts.
- Computed active-job throughput/ETA/slack for P2 and P1.
- Performed fatal-error signature scan on active logs and confirmed no crash indicators.

### Cycle 2026-02-26 05:50:01 CST
#### Results
- Queue/slot status (`05:50:16 CST`):
  - Standard active count is `8/8` (compliant).
  - Active standard P2 jobs: `729444` (`bin4_f2d1`), `729445` (`bin6_f2d1`), `729446` (`bin8_f2d1`), `729451` (`bin1_f2d1`), `729452` (`bin50_f2d1`), `729453` (`bin100_f2d1`), `729458` (`bin2_f2d1`), `729459` (`bin200_f2d1`).
  - Burst chain unchanged: `729294` running (`p1`), `729403` pending dependency fallback.
- Completed P2 metrics remain unchanged this cycle (8 completed settings: `bin 5/10/15/25` x `force2d {0,1}`).
- Baseline CSV evidence unchanged:
  - `experiment_results.csv` unchanged.
  - `experiment_averages.csv` unchanged.
- Checkpoint progress snapshot:
  - P2 extremes/local probes progressing (`rows_cp`: `333` to `628` of `980` depending on run).
  - P1 remains at `678/980` with no new checkpoint rows since `05:20:26 CST`.

#### Insights learned
1. Completed-run binWidth effect size is still exactly zero across all moderate tested bins with uncertainty unchanged.
   - Within `force2d=true`: `test=0.6258`, `CV=0.6212 ± 0.0370` for `bin 5/10/15/25`.
   - Within `force2d=false`: `test=0.6053`, `CV=0.6195 ± 0.0494` for `bin 5/10/15/25`.
   - Causal implication: moderate binWidth appears operationally inert; only extreme-bin runs can falsify this.
2. Active P2 runs have healthy throughput and strong walltime headroom despite variance.
   - Mean extraction rate across active P2 jobs: `6.30 ± 1.11` rows/min.
   - Per-job extraction slack is strongly positive (`+184.7` to `+243.4` min).
   - Causal implication: current standard jobs are likely to finish extraction well before timeout; no urgent resubmission needed for P2.
3. P1 burst run now has negative extraction slack and likely times out before extraction completion.
   - `729294`: `678/980`, rate `4.13` rows/min, extraction ETA `~73.1` min, remaining walltime `~45.8` min, slack `-27.4` min.
   - Checkpoint row count unchanged for ~31 minutes while stderr continues emitting PyRadiomics warnings.
   - Causal implication: expect timeout on `729294`; rely on queued dependency fallback `729403` for continuation.

#### Hypothesis update
- Hypothesis unchanged: `force2d` is a consistent lever; binWidth signal is suppressed in the current extraction path unless extreme values break invariance.
- Operationally, near-term decision value remains in first completed extreme-bin metrics and their feature-hash divergence (or lack thereof).

#### Next experiments
- PASS on additional submissions this cycle.
  - Reason: standard queue is fully allocated (`8/8`) to high-decision-value ISPY2 extreme-bin tests, and burst fallback for P1 (`729403`) is already queued.
- Next-cycle trigger:
  - if `729294` times out, confirm `729403` starts and resumes from checkpoint;
  - if first extreme-bin metrics appear, compare (`test`, `CV mean`, `CV std`) and feature hashes against invariant baseline to decide immediate pivot.

#### Actions taken
- Read required evidence this cycle:
  - `experiment_results.csv`, `experiment_averages.csv`, `squeue`, `sacct`, active logs, and checkpoint artifacts.
- Quantified active-job throughput/ETA/slack for both P2 and P1.
- Confirmed no fatal error signatures in active radiomics logs.

### Cycle 2026-02-26 06:20:01 CST
#### Results
- Queue/slot status (`06:20:16 CST`):
  - Standard active count is `8/8` (compliant).
  - Active standard P2 jobs: `729444` (`bin4_f2d1`), `729445` (`bin6_f2d1`), `729446` (`bin8_f2d1`), `729451` (`bin1_f2d1`), `729452` (`bin50_f2d1`), `729453` (`bin100_f2d1`), `729458` (`bin2_f2d1`), `729459` (`bin200_f2d1`).
  - Burst chain unchanged: `729294` running (`p1`), `729403` pending dependency fallback.
- Completed P2 metrics unchanged this cycle:
  - still 8 completed settings (`bin 5/10/15/25` x `force2d {0,1}`), with no new `metrics.json` from extreme-bin runs yet.
- Baseline CSV evidence unchanged:
  - `experiment_results.csv` unchanged.
  - `experiment_averages.csv` unchanged.
- Checkpoint progress this cycle:
  - Active P2 extraction rows now range `495` to `678` of `980`.
  - P1 checkpoint advanced to `789/980` (`680 train + 109 test`) but still no final feature files.

#### Insights learned
1. BinWidth invariance in completed moderate settings remains exact with unchanged uncertainty.
   - `force2d=true`: `test=0.6258`, `CV=0.6212 ± 0.0370` for `bin 5/10/15/25`.
   - `force2d=false`: `test=0.6053`, `CV=0.6195 ± 0.0494` for `bin 5/10/15/25`.
   - Causal implication: until extreme-bin metrics land, there is no evidence that binWidth is a useful optimization axis.
2. Active P2 jobs retain large positive extraction slack despite moderate throughput variance.
   - Mean active extraction rate: `5.77 ± 0.54` rows/min.
   - Per-job extraction slack: `+176.6` to `+220.1` min.
   - Causal implication: P2 timeout risk is low; current queue allocation is stable and should yield decision-making metrics.
3. P1 is still likely to timeout before extraction completes even though checkpointing is progressing.
   - `729294`: `789/980`, rate `4.06` rows/min, extraction ETA `~47.0` min, remaining walltime `~15.8` min, slack `-31.2` min.
   - Causal implication: expect timeout and automatic handoff to dependency job `729403`; checkpointing should preserve completed work.

#### Hypothesis update
- Hypothesis unchanged: `force2d` has consistent signal; binWidth remains suppressed/inert in completed moderate runs, so next decision value depends on extreme-bin outcomes (`1,2,50,100,200`).

#### Next experiments
- PASS on additional submissions this cycle.
  - Reason: standard floor is already saturated (`8/8`) with highest-value ISPY2 extreme-bin probes, and no new metrics are available yet to justify a branch change.
- Next-cycle triggers:
  - confirm `729403` starts/resumes after `729294` finishes or times out,
  - immediately rank first completed extreme-bin metrics by (`CV mean`, `CV std`, `test`) and compare feature hashes to invariant baseline.

#### Actions taken
- Read required evidence this cycle:
  - `experiment_results.csv`, `experiment_averages.csv`, `squeue`, `sacct`, recent logs, and checkpoint artifacts.
- Quantified active-job throughput/ETA/slack for P2 and P1.
- Verified no fatal error signatures in active radiomics logs.

### Cycle 2026-02-26 06:50:02 CST
#### Results
- Queue/slot status (`06:50:15 CST`):
  - Standard active count is `8/8` (compliant).
  - Active standard P2 jobs: `729444` (`bin4_f2d1`), `729445` (`bin6_f2d1`), `729446` (`bin8_f2d1`), `729451` (`bin1_f2d1`), `729452` (`bin50_f2d1`), `729453` (`bin100_f2d1`), `729458` (`bin2_f2d1`), `729459` (`bin200_f2d1`).
  - Burst: `729403` running (`p1`); no burst pending jobs.
- P1 transition this cycle:
  - `729294` ended with `TIMEOUT` at `2026-02-26 06:37:16 CST`.
  - Dependency fallback `729403` started immediately and resumed extraction.
- Baseline CSV evidence unchanged:
  - `experiment_results.csv` unchanged.
  - `experiment_averages.csv` unchanged.
- Completed P2 metrics unchanged this cycle (still 8 completed settings from `bin 5/10/15/25` x `force2d {0,1}`).

#### Insights learned
1. Checkpoint handoff worked as intended after burst timeout, preserving and extending P1 progress.
   - Previous cycle end: `789/980` rows.
   - Current cycle: `959/980` rows (`680 train + 279 test`) under `729403`.
   - Causal implication: extraction checkpointing is functioning and effectively mitigates burst walltime timeout risk.
2. P2 extraction jobs remain well within walltime with large positive slack despite rate spread.
   - Active P2 extraction rate: `5.40 ± 0.51` rows/min.
   - Extraction slack across active P2 jobs: `+160.0` to `+204.2` minutes.
   - Causal implication: no immediate timeout/resubmission risk on standard queue; metrics should arrive without intervention.
3. No new performance evidence has yet challenged binWidth invariance.
   - Completed metrics remain exactly split by `force2d` branch with unchanged means/std.
   - Causal implication: keep waiting for extreme-bin outcomes before reallocating slots away from binWidth stress tests.

#### Hypothesis update
- Hypothesis unchanged: `force2d` drives the observed signal; binWidth remains inert in completed moderate bins, and extreme-bin results are required to falsify or confirm full invariance.
- Operational update: P1 timeout risk has shifted from high to low due successful checkpoint resume in `729403`.

#### Next experiments
- PASS on additional submissions this cycle.
  - Reason: standard floor is full (`8/8`) with highest-value ISPY2 probes, and no new completed extreme-bin metrics exist yet to justify a branch pivot.
- Next-cycle triggers:
  - if first extreme-bin metrics complete, rank by (`CV mean`, `CV std`, `test`) and compare feature hashes against invariant baseline.
  - if P1 extraction finalizes, monitor whether training starts/completes within current burst walltime and capture resulting P1 metrics.

#### Actions taken
- Read required evidence this cycle:
  - `experiment_results.csv`, `experiment_averages.csv`, `squeue`, `sacct`, recent P1/P2 logs, and checkpoint artifacts.
- Confirmed and logged `729294` timeout plus `729403` checkpoint-resume progression.
- Quantified active P2 throughput and walltime slack; confirmed standard floor remains compliant.

### Cycle 2026-02-26 07:25:01 CST
#### Results
- Queue/slot status (`07:25:29 CST`):
  - Standard active count dropped to `7/8` (below required floor) after `729446` completed.
  - Active standard P2 jobs: `729444` (`bin4_f2d1`), `729445` (`bin6_f2d1`), `729451` (`bin1_f2d1`), `729452` (`bin50_f2d1`), `729453` (`bin100_f2d1`), `729458` (`bin2_f2d1`), `729459` (`bin200_f2d1`).
  - Burst jobs are finished (`729403` completed; no burst jobs active).
- New completions this cycle:
  - `729446` completed (`ispy2_p2_phaseblind_bin8_f2d1`) with `test 0.6258`, `CV 0.6212 ± 0.0370`.
  - `729403` completed P1 temporal phase-blind sweep; produced 4 training results.
- Baseline CSV evidence unchanged:
  - `experiment_results.csv` unchanged.
  - `experiment_averages.csv` unchanged.

#### Insights learned
1. P2 performance remains effectively invariant across all completed `force2d=true` bins despite feature-hash differences.
   - `bin8_f2d1` has different final feature hashes (`train 685845af4938`, `test 6b5a6a03e54f`) from prior `f2d1` bins, yet identical metrics (`test 0.6258`, `CV 0.6212 ± 0.0370`).
   - Causal implication: extraction-level binWidth changes can alter raw features without changing downstream predictive performance under current modeling.
2. P1 completion adds mixed evidence on feature-family utility, with largest test lift from selector change but not CV lift.
   - `all_mrmr50` vs `all_kbest50`: test `+0.0328` (`0.6381 - 0.6053`), CV `-0.0001` (`0.6193 - 0.6195`), std `-0.0018`.
   - `kinsubonly_kbest50` vs `all_kbest50`: CV `+0.0037` (`0.6232 - 0.6195`), test `+0.0084`, std `+0.0010`.
   - Causal implication: selector choice (kbest vs mrmr) may improve held-out split outcomes without improving CV mean; needs direct confirmation in current P2 extraction regime.
3. Active extreme-bin P2 jobs still have strong extraction headroom.
   - Mean active extraction rate: `4.46 ± 0.67` rows/min.
   - Extraction slack remains positive (`+109.7` to `+198.0` min).
   - Causal implication: current runs are unlikely to timeout and should continue delivering decisive extreme-bin evidence.

#### Hypothesis update
- Updated operational hypothesis: binWidth changes are not the main bottleneck under current pipeline; higher decision value now comes from model-selection/feature-selection choices on the stabilized phase-blind extraction (especially testing mRMR in P2).

#### Next experiments
- Restore standard floor from `7/8` to `8/8` with one high-decision-value standard job:
  - P2 train-only rerun on completed `bin8_f2d1` extraction using `--feature-selection mrmr --k-best 50`.
- Rationale:
  - directly tests whether the P1 `mRMR` test-lift signal transfers to the active P2 extraction regime,
  - avoids re-extraction by reusing cache, so feedback arrives quickly while keeping standard floor compliant.

#### Actions taken
- Read required evidence this cycle:
  - `experiment_results.csv`, `experiment_averages.csv`, `squeue`, `sacct`, recent logs, and extraction/training outputs.
- Confirmed `729446` completion and `729403` P1 sweep completion.
- Quantified effect sizes (test/CV/std deltas) for P1/P2 and identified current floor violation (`7/8`).

### Cycle 2026-02-26 07:30:12 CST (07:25 Follow-up)
#### Results
- Standard floor restoration completed:
  - Submitted `729484` (`rad_ispy2_p2`, train-only `bin8_f2d1` with `mrmr50`).
  - Queue state after submission: `7` running + `1` pending standard P2 jobs (`std_active=8/8`).
  - Pending reason for `729484`: `QOSMaxJobsPerUserLimit` (will run as slots free).

#### Insights learned
1. Floor can be restored with high-decision-value pending work even when immediate start is blocked by QoS concurrency.
2. mRMR transfer test is now queued without extra extraction cost, so it can produce a selector-specific answer quickly once scheduled.

#### Hypothesis update
- Unchanged from 07:25: mRMR may lift held-out performance in this regime without CV gain; `729484` is the direct test.

#### Next experiments
- PASS on additional submissions in this follow-up step.
  - Reason: standard floor is restored (`8/8`) and slot limit is currently saturated.

#### Actions taken
- Submitted one standard train-only P2 mRMR job (`729484`) using cached `bin8_f2d1` features.
- Verified queue and floor compliance after submission.

### Cycle 2026-02-26 07:35:01 CST
#### Results
- Queue/slot status (`07:35:20 CST`):
  - Standard active count is `7/8` (below floor): `6` running + `1` pending.
  - Running standard P2 jobs: `729444` (`bin4_f2d1`), `729445` (`bin6_f2d1`), `729451` (`bin1_f2d1`), `729452` (`bin50_f2d1`), `729458` (`bin2_f2d1`), `729459` (`bin200_f2d1`).
  - Pending standard P2 job: `729484` (train-only `bin8_f2d1_mrmr50`, waiting on `QOSMaxJobsPerUserLimit`).
- New completion this cycle:
  - `729453` completed (`ispy2_p2_phaseblind_bin100_f2d1`) with `test 0.6258`, `CV 0.6212 ± 0.0370`.
- Baseline CSV evidence unchanged:
  - `experiment_results.csv` unchanged.
  - `experiment_averages.csv` unchanged.

#### Insights learned
1. Extreme-bin completion (`bin100_f2d1`) still matches the invariant `force2d=true` metric regime exactly.
   - `bin100_f2d1`: `test 0.6258`, `CV 0.6212 ± 0.0370`.
   - Delta vs prior `f2d1` settings (`5/8/10/15/25`): `0.0000` on test, CV mean, and CV std.
   - Causal implication: broadening binWidth farther has still not changed predictive performance under current P2 training setup.
2. Feature-level differences continue not to translate into performance differences.
   - `bin8_f2d1` and `bin100_f2d1` share the same final hashes (`train 685845af4938`, `test 6b5a6a03e54f`) and same metrics; earlier `f2d1` bins had different hashes but also same metrics.
   - Causal implication: classifier/selection stage may be collapsing extraction differences into equivalent predictive subsets.
3. Active P2 jobs remain healthy with strong extraction headroom.
   - Mean active extraction rate: `4.46 ± 0.67` rows/min.
   - Positive extraction slack remains (`+109.7` to `+198.0` min).
   - Causal implication: no timeout-driven intervention needed for active standard runs; we can spend the missing slot on a high-value train-only probe.

#### Hypothesis update
- Updated top hypothesis: in the current phase-blind ISPY2 regime, binWidth is not a productive optimization axis for model performance; higher decision value now comes from testing feature-family and selector choices on already-completed extractions.

#### Next experiments
- Restore standard floor from `7/8` to `8/8` by submitting one standard train-only job on cached extraction:
  - `ispy2_p2_phaseblind_bin100_f2d1_kinsubonly_kbest50` (exclude raw DCE channels, keep kinetic/subtraction features).
- Rationale:
  - directly tests whether the P1 kinsub-only CV lift (`+0.0037`) transfers to the P2 extraction regime,
  - zero extraction cost; fast decision feedback once scheduled.

#### Actions taken
- Read required evidence this cycle:
  - `experiment_results.csv`, `experiment_averages.csv`, `squeue`, `sacct`, recent logs, and P2 metrics/checkpoint artifacts.
- Confirmed new completion (`729453`) and quantified effect sizes/uncertainty.
- Identified floor violation (`7/8`) and selected one immediate high-value replacement job.

### Cycle 2026-02-26 07:37:05 CST (07:35 Follow-up)
#### Results
- Standard floor restoration completed:
  - Submitted `729487` (`rad_ispy2_p2`, train-only `bin100_f2d1_kinsubonly_kbest50`).
  - Queue now has `6` running + `2` pending standard P2 jobs (`std_active=8/8`).
  - Both pending jobs (`729484`, `729487`) are waiting on `QOSMaxJobsPerUserLimit`.

#### Insights learned
1. Floor compliance can be restored via high-value queued train-only probes even under user QoS job-count cap.
2. We now have two complementary no-extraction selector/feature-family probes queued on completed extractions (`mrmr50` and `kinsub-only`), improving decision value per slot release.

#### Hypothesis update
- Unchanged from 07:35 cycle: model-side selection/feature-family choices are currently higher-yield than additional binWidth sweeps.

#### Next experiments
- PASS on additional submissions in this follow-up step.
  - Reason: standard floor is restored (`8/8`) and further submissions would queue-churn under the same QoS cap.

#### Actions taken
- Submitted one standard train-only P2 kinsub-only job (`729487`) on cached `bin100_f2d1` features.
- Verified floor restoration to `8/8` active standard jobs after submission.

### Cycle 2026-02-26 08:05:02 CST
#### Results
- Queue/slot status (`08:05:16 CST`):
  - Standard active count dropped to `6/8` (below floor) after two quick train-only jobs completed.
  - Running standard P2 jobs: `729444` (`bin4_f2d1`), `729445` (`bin6_f2d1`), `729451` (`bin1_f2d1`), `729452` (`bin50_f2d1`), `729458` (`bin2_f2d1`), `729459` (`bin200_f2d1`).
  - No burst jobs active.
- New completed train-only P2 probes this cycle:
  - `729484` (`ispy2_p2_phaseblind_bin8_f2d1_mrmr50`): `test 0.6367`, `CV 0.6167 ± 0.0402`.
  - `729487` (`ispy2_p2_phaseblind_bin100_f2d1_kinsubonly_kbest50`): `test 0.6373`, `CV 0.6213 ± 0.0420`.
- Additional extraction completion carried forward:
  - `ispy2_p2_phaseblind_bin100_f2d1` finalized with unchanged baseline regime metrics (`test 0.6258`, `CV 0.6212 ± 0.0370`).
- Baseline CSV evidence unchanged:
  - `experiment_results.csv` unchanged.
  - `experiment_averages.csv` unchanged.

#### Insights learned
1. Both new model-side probes increased held-out test AUC by ~+0.011 over the current phase-blind baseline.
   - vs baseline `bin*_f2d1 kbest50` (`test 0.6258`, `CV 0.6212 ± 0.0370`):
   - `bin8_mrmr50`: test `+0.0109`, CV `-0.0045`, std `+0.0032`.
   - `bin100_kinsubonly_kbest50`: test `+0.0115`, CV `+0.0001`, std `+0.0050`.
   - Causal implication: selector/feature-family changes can lift single split test AUC, but may not improve CV mean stability.
2. kinsub-only transfer signal appears more CV-consistent than mRMR transfer in this P2 regime.
   - `kinsub-only` retains CV mean parity with baseline (`~+0.0001`) while mRMR drops CV (`-0.0045`).
   - Causal implication: kinsub-only may be the safer direction for robust generalization, while mRMR might be exploiting split-specific signal.
3. Remaining active P2 extraction jobs still have healthy walltime margin.
   - Mean active extraction rate: `4.10 ± 0.14` rows/min.
   - Extraction slack remains strongly positive (`+109.0` to `+135.5` min).
   - Causal implication: active extraction runs should finish without timeout; we can allocate missing slots to fast train-only diagnostics.

#### Hypothesis update
- Updated top hypothesis: the bottleneck is no longer extraction parameterization (binWidth), but model-side regularization/selection interacting with feature families; robust gains likely require combinations that preserve CV while improving test (kinsub-only appears promising, mRMR alone less stable).

#### Next experiments
- Restore floor from `6/8` to `8/8` with two high-decision-value standard train-only probes on cached `bin100_f2d1` extraction:
  - `ispy2_p2_phaseblind_bin100_f2d1_kinsubonly_mrmr50` (tests selector+feature-family interaction).
  - `ispy2_p2_phaseblind_bin100_f2d1_rawonly_kbest50` (tests whether raw-only reproduces higher test behavior seen in P1).
- Rationale:
  - both avoid extraction cost,
  - both directly discriminate whether observed test lift comes from kinetic/subtraction channel focus vs selector behavior vs raw-only drift.

#### Actions taken
- Read required evidence this cycle:
  - `experiment_results.csv`, `experiment_averages.csv`, `squeue`, `sacct`, recent logs, and current P2 metrics/checkpoint artifacts.
- Quantified new effect sizes with uncertainty for completed train-only probes.
- Identified floor violation (`6/8`) and selected two immediate replacement jobs.

### Cycle 2026-02-26 08:09:20 CST (08:05 Follow-up)
#### Results
- Standard floor restoration completed:
  - Submitted `729491` (`rad_ispy2_p2`, train-only `bin100_f2d1_kinsubonly_mrmr50`).
  - Submitted `729492` (`rad_ispy2_p2`, train-only `bin100_f2d1_rawonly_kbest50`).
  - Queue after submissions: `7` running + `1` pending standard P2 (`std_active=8/8`).
  - `729491` is already running; `729492` is pending on `QOSMaxJobsPerUserLimit`.

#### Insights learned
1. We restored floor while prioritizing high-decision-value no-extraction probes, minimizing turnaround time.
2. The kinsub+mRMR interaction test is already in progress, so we should get model-side interaction evidence in the next cycle window.

#### Hypothesis update
- Unchanged from 08:05: model-side selector/feature-family interaction appears to be the current highest-value optimization axis.

#### Next experiments
- PASS on additional submissions in this follow-up step.
  - Reason: floor is restored (`8/8`) and QoS cap is already binding pending starts.

#### Actions taken
- Submitted two standard train-only P2 jobs (`729491`, `729492`) on cached `bin100_f2d1` features.
- Verified post-submit floor compliance (`std_active=8/8`) and job start state.

### Cycle 2026-02-26 08:15:01 CST
#### Results
- Queue/slot status (`08:15:14 CST`):
  - Standard active count is `6/8` (below floor), with `6` running and `0` pending.
  - Running standard P2 jobs: `729444` (`bin4_f2d1`), `729445` (`bin6_f2d1`), `729451` (`bin1_f2d1`), `729452` (`bin50_f2d1`), `729458` (`bin2_f2d1`), `729459` (`bin200_f2d1`).
- New completed train-only jobs this cycle:
  - `729491` (`ispy2_p2_phaseblind_bin100_f2d1_kinsubonly_mrmr50`): `test 0.6329`, `CV 0.6200 ± 0.0444`.
  - `729492` (`ispy2_p2_phaseblind_bin100_f2d1_rawonly_kbest50`): `test 0.6182`, `CV 0.5865 ± 0.0347`.
- Baseline CSV evidence unchanged:
  - `experiment_results.csv` unchanged.
  - `experiment_averages.csv` unchanged.

#### Insights learned
1. Kinetic/subtraction-only feature family still yields the strongest test AUC in this local model-space, but mRMR weakens stability vs kbest.
   - `kinsub_kbest` (`test 0.6373`, `CV 0.6213 ± 0.0420`) vs `kinsub_mrmr` (`test 0.6329`, `CV 0.6200 ± 0.0444`):
   - test delta `-0.0045`, CV delta `-0.0013`, std `+0.0024` for mRMR.
   - Causal implication: for kinsub-only, kbest currently dominates mRMR on both held-out and CV stability.
2. Raw-only under current P2 extraction is likely a weak/unstable signal source.
   - `rawonly_kbest` vs baseline all-feature `kbest`: test `-0.0077` (`0.6182 - 0.6258`), CV `-0.0347` (`0.5865 - 0.6212`), std `-0.0023`.
   - Causal implication: removing kinetic/subtraction channels materially harms CV performance; raw channels alone are insufficient.
3. Active extraction jobs remain healthy and near completion with strong slack.
   - Mean active extraction rate: `4.16 ± 0.13` rows/min.
   - Extraction slack remains positive (`+112.8` to `+135.6` min).
   - Causal implication: no timeout intervention needed; open slots should be used for fast train-only disambiguation jobs.

#### Hypothesis update
- Updated top hypothesis: improvement opportunity is in model-side combinations on phase-blind extraction, and the next decision-critical gap is isolating selector effects in the remaining bin100 cells (`all_mrmr` and `rawonly_mrmr`) rather than further extraction sweeps.

#### Next experiments
- Restore floor from `6/8` to `8/8` with two train-only standard jobs on cached `bin100_f2d1` features:
  - `ispy2_p2_phaseblind_bin100_f2d1_all_mrmr50`.
  - `ispy2_p2_phaseblind_bin100_f2d1_rawonly_mrmr50`.
- Rationale:
  - completes the missing high-value selector/feature-family factorial cells for `bin100`,
  - gives direct attribution for whether mRMR helps in all-feature and raw-only regimes.

#### Actions taken
- Read required evidence this cycle:
  - `experiment_results.csv`, `experiment_averages.csv`, `squeue`, `sacct`, recent logs, and latest P2 metrics/checkpoint artifacts.
- Quantified effect sizes/uncertainty for newly completed train-only jobs (`729491`, `729492`).
- Identified floor violation (`6/8`) and selected two immediate high-value train-only replacements.

### Cycle 2026-02-26 08:17:10 CST (08:15 Follow-up)
#### Results
- Standard floor restoration completed:
  - Submitted `729494` (`rad_ispy2_p2`, train-only `bin100_f2d1_all_mrmr50`).
  - Submitted `729495` (`rad_ispy2_p2`, train-only `bin100_f2d1_rawonly_mrmr50`).
  - Queue after submission: `8` running + `0` pending standard P2 jobs (`std_active=8/8`).
- Both new jobs started immediately and are actively training.

#### Insights learned
1. `8/8` can be restored quickly with train-only jobs when extraction runs are still in progress.
2. Completing the missing `all/raw x mrmr` cells should resolve selector-vs-feature-family attribution in the current regime.

#### Hypothesis update
- Unchanged from 08:15 cycle: model-side selector/feature-family interaction remains the highest-value optimization axis.

#### Next experiments
- PASS on additional submissions in this follow-up step.
  - Reason: standard floor is restored (`8/8`) and all selected high-value diagnostic jobs are now active.

#### Actions taken
- Submitted two standard train-only P2 jobs (`729494`, `729495`) on cached `bin100_f2d1` features.
- Verified both jobs are running and floor compliance is restored.

### Cycle 2026-02-26 08:15:01 CST (Late Consolidation)
#### Results
- Queue/slot status at evidence read (`08:56 CST`):
  - Standard active count observed at `6/8` (`pending` jobs currently `JobHeldUser`).
- New completed train-only P2 jobs since earlier 08:15 note:
  - `729494` (`ispy2_p2_phaseblind_bin100_f2d1_all_mrmr50`): `test 0.6367`, `CV 0.6167 ± 0.0402`.
  - `729495` (`ispy2_p2_phaseblind_bin100_f2d1_rawonly_mrmr50`): `test 0.6182`, `CV 0.5863 ± 0.0347`.
- P2 metrics table now has 16 entries including all key bin100 model-side cells.
- Baseline CSV evidence unchanged (`experiment_results.csv`, `experiment_averages.csv`).

#### Insights learned
1. In all-feature regime, mRMR improves held-out test but degrades CV mean and increases variability.
   - `all_mrmr50` vs baseline `all_kbest50`: test `+0.0109` (`0.6367 - 0.6258`), CV `-0.0045` (`0.6167 - 0.6212`), std `+0.0032`.
   - Causal implication: mRMR signal may be split-specific rather than robustly generalizable in current P2 setup.
2. Raw-only remains consistently weak regardless of selector.
   - `rawonly_kbest50`: `test 0.6182`, `CV 0.5865 ± 0.0347`.
   - `rawonly_mrmr50`: `test 0.6182`, `CV 0.5863 ± 0.0347`.
   - Causal implication: selector choice does not rescue raw-only performance; kinetic/subtraction channels are carrying most usable signal.
3. Best observed test remains in kinetic/subtraction-focused models, with kbest dominating mRMR for stability.
   - `kinsubonly_kbest50`: `test 0.6373`, `CV 0.6213 ± 0.0420`.
   - `kinsubonly_mrmr50`: `test 0.6329`, `CV 0.6200 ± 0.0444`.
   - Causal implication: current best direction is kinsub-focused with kbest, then tuning model capacity/feature count rather than selector family.

#### Hypothesis update
- Updated top hypothesis: the practical optimization axis is within kinsub-only model capacity (e.g., `k-best` size), not binWidth nor mRMR selection.

#### Next experiments
- Restore floor from `6/8` to `8/8` with two high-value train-only standard jobs on cached `bin100_f2d1` extraction:
  - `ispy2_p2_phaseblind_bin100_f2d1_kinsubonly_kbest20`.
  - `ispy2_p2_phaseblind_bin100_f2d1_kinsubonly_kbest100`.
- Rationale:
  - directly probes capacity/regularization tradeoff around current best family (`kinsub + kbest`),
  - no extraction cost, fast turnaround for robust-vs-overfit discrimination.

#### Actions taken
- Read required evidence this cycle:
  - `experiment_results.csv`, `experiment_averages.csv`, `squeue`, `sacct`, recent logs, and all current P2 metrics artifacts.
- Quantified effect sizes and uncertainty for newly completed `all_mrmr` and `rawonly_mrmr` cells.
- Selected two immediate replacement train-only jobs to restore floor and maximize decision value.

### Cycle 2026-02-26 09:00:10 CST (08:15 Follow-up)
#### Results
- Standard floor restoration completed:
  - Submitted `729610` (`rad_ispy2_p2`, train-only `bin100_f2d1_kinsubonly_kbest20`).
  - Submitted `729611` (`rad_ispy2_p2`, train-only `bin100_f2d1_kinsubonly_kbest100`).
- Queue after submission:
  - `std_active=8/8` (6 pending held-user jobs + 2 running new train-only jobs).
  - legacy extraction jobs `729444/729445/729451/729452/729458/729459` are currently `PD` with reason `JobHeldUser`.
  - new jobs `729610/729611` started immediately and are running.

#### Insights learned
1. We can maintain the required floor even with held legacy jobs by adding active train-only diagnostics.
2. The kinsub `k-best` sweep (`k=20` vs `k=100`) is now executing and should directly resolve capacity sensitivity around the current best family.

#### Hypothesis update
- Unchanged from 08:15: kinsub + kbest is currently the strongest path; best next gain likely comes from selecting a better `k` value rather than selector swap.

#### Next experiments
- PASS on further submissions in this follow-up step.
  - Reason: floor restored (`8/8`) and two highest-value kinsub capacity probes are active.

#### Actions taken
- Submitted `729610` and `729611` and verified both are running.
- Verified standard floor compliance (`8/8`) after submission.

### Cycle 2026-02-27 11:26:40 CST (Post-Overnight Resume)
#### Results
- Current queue at cycle read had no active `rad_ispy2` jobs.
- Extraction status snapshot:
  - fully complete bins: `5,8,10,15,25,100` for `force2d=true` and `5,10,15,25` for `force2d=false`.
  - near-complete but not finalized: `bin1_f2d1 (956/980)`, `bin2_f2d1 (916/980)`, `bin4_f2d1 (979/980)`, `bin6_f2d1 (979/980)`, `bin50_f2d1 (958/980)`, `bin200_f2d1 (884/980)`.
- Latest completed train-only results (overnight tail):
  - `bin100 all_mrmr50`: `test 0.6367`, `CV 0.6167 ± 0.0402`.
  - `bin100 kinsub_kbest50`: `test 0.6373`, `CV 0.6213 ± 0.0420`.
  - `bin100 kinsub_mrmr50`: `test 0.6329`, `CV 0.6200 ± 0.0444`.
  - `bin100 raw_kbest50`: `test 0.6182`, `CV 0.5865 ± 0.0347`.
  - `bin100 raw_mrmr50`: `test 0.6182`, `CV 0.5863 ± 0.0347`.

#### Insights learned
1. Kinetic/subtraction-focused models remain strongest, and within that family `kbest` is better than `mrmr` on both test and CV stability.
   - `kinsub_kbest50` vs `kinsub_mrmr50`: test `+0.0045`, CV `+0.0013`, std `-0.0024`.
   - Causal implication: selector family should stay `kbest` while tuning capacity (`k`).
2. Raw-only is consistently weak and selector-insensitive.
   - `raw_kbest50` and `raw_mrmr50` are effectively identical (`test 0.6182`, `CV ~0.586`).
   - Causal implication: raw-only branch should be deprioritized for performance pursuit.
3. Six near-complete extraction runs are low-cost to finish and high-value to close bin-extreme uncertainty.
   - most are within `1-96` rows of finalization.
   - Causal implication: resuming these now gives fast closure on extraction-invariance while we run model-side probes in parallel.

#### Hypothesis update
- Updated hypothesis: binWidth variation remains low-yield for predictive performance; best opportunity is kinsub + kbest capacity tuning, while finishing near-complete extractions is primarily to close uncertainty, not because large performance shifts are expected.

#### Next experiments
- Keep `8` standard + `8` burst slots busy with this split:
  - Standard (8):
    - resume extraction runs: `bin1_f2d1`, `bin2_f2d1`, `bin4_f2d1`, `bin6_f2d1`, `bin50_f2d1`, `bin200_f2d1`.
    - train-only kinsub capacity probes: `bin100 kinsub kbest20`, `bin100 kinsub kbest100`.
  - Burst (8, <=4h):
    - train-only diagnostics: `bin100 kinsub kbest10/30/70/150`, `bin100 all kbest20/100/200`, `bin8 kinsub kbest50`.

#### Actions taken
- Prepared queue-filling submission plan prioritizing no-extraction train-only diagnostics plus low-cost extraction completion.
- Next step in this cycle: submit jobs and verify slot occupancy.

### Cycle 2026-02-27 11:36:40 CST (Post-Resume Follow-up)
#### Results
- Queue fill executed to target occupancy:
  - Standard `qos=normal`: `8/8` active (`running+pending`).
  - Burst `qos=burst`: `8/8` active (`running+pending`).
- Submitted jobs:
  - Standard extraction resumes: `730585`..`730590` (`bin1,2,4,6,50,200` with `force2d=1`).
  - Standard train-only: `730592` (`bin100 kinsub kbest100`), `730601` (`bin100 kinsub mrmr20`), `730605` (`bin100 all mrmr20`), `730606` (`bin100 kinsub mrmr100`).
  - Burst train-only: `730593`..`730600`, `730602`, `730603`, `730604` (kinsub/all/raw diagnostics as listed in plan).
- Correction made:
  - canceled duplicate standard job `730591` (`bin100 kinsub kbest20` already existed) and replaced with missing `730601`.

#### Insights learned
1. We can keep both queue classes fully occupied with a mixed strategy: long extraction resumes on standard plus fast train-only diagnostics on burst.
2. The pending/running mix changed over minutes due scheduler preemption/limits, but target occupancy was achieved after replenishment.

#### Hypothesis update
- Unchanged: kinsub + kbest remains strongest current branch; ongoing jobs are targeted to resolve `k` sensitivity and mRMR interaction effects.

#### Next experiments
- PASS on additional submissions right now.
  - Reason: both occupancy targets are satisfied (`8` standard + `8` burst active).

#### Actions taken
- Submitted 16 jobs to fill both slot classes and then submitted 3 additional corrective/replacement jobs as queue state shifted.
- Verified final occupancy: `std_active=8`, `burst_active=8`.

### Cycle 2026-02-27 11:45:03 CST (Active Queue + Corr Sweep)
#### Results
- Queue status after refill:
  - `std_active=8/8` (`7` running + `1` pending normal).
  - `burst_active=8/8` (`4` running + `4` pending burst).
- New completed correlation-threshold probes (ISPY2, phase-blind, image-only):
  - `ispy2_p2_phaseblind_bin100_f2d1_all_mrmr20_corr050`: `test 0.6519`, `CV 0.6175 ± 0.0383`.
  - `ispy2_p2_phaseblind_bin100_f2d1_all_mrmr20_corr060`: `test 0.6422`, `CV 0.6202 ± 0.0380`.
  - `ispy2_p2_phaseblind_bin100_f2d1_all_mrmr20_corr070`: `test 0.6395`, `CV 0.6260 ± 0.0438`.
  - `ispy2_p2_phaseblind_bin100_f2d1_all_mrmr20_corr080`: `test 0.6317`, `CV 0.6269 ± 0.0384`.
  - `ispy2_p2_phaseblind_bin100_f2d1_all_mrmr20_corr085`: `test 0.6291`, `CV 0.6195 ± 0.0371`.
  - `ispy2_p2_phaseblind_bin100_f2d1_all_mrmr20_corr095`: `test 0.6511`, `CV 0.6139 ± 0.0354`.
  - `ispy2_p2_phaseblind_bin100_f2d1_all_mrmr20_corr099`: `test 0.6410`, `CV 0.6192 ± 0.0388`.
  - `ispy2_p2_phaseblind_bin100_f2d1_kinsubonly_kbest40_corr095`: `test 0.6366`, `CV 0.6212 ± 0.0376`.
  - `ispy2_p2_phaseblind_bin100_f2d1_kinsubonly_kbest40_corr085`: `test 0.6211`, `CV 0.6177 ± 0.0421`.
  - `ispy2_p2_phaseblind_bin100_f2d1_kinsubonly_kbest40_corr080`: `test 0.6262`, `CV 0.6271 ± 0.0433`.
  - `ispy2_p2_phaseblind_bin100_f2d1_kinsubonly_kbest40_corr075`: `test 0.6145`, `CV 0.6201 ± 0.0418`.
  - `ispy2_p2_phaseblind_bin8_f2d1_kinsubonly_kbest50_corr095`: `test 0.6422`, `CV 0.6253 ± 0.0377`.

#### Insights learned
1. Correlation pruning is now a high-leverage knob with a clear CV-vs-test tradeoff.
   - In `all+mrmr20`, looser pruning (`corr=0.50/0.95`) maximizes single-split test AUC (~`0.651`), while mid pruning (`corr=0.70/0.80`) gives better CV mean (`0.6260-0.6269`).
   - Causal implication: a meaningful part of prior variance likely came from correlation-pruning aggressiveness, not just selector family.
2. For `kinsub+kbest40`, `corr=0.80` gives the strongest CV mean (`0.6271`) but lower test than `corr=0.95`.
   - Causal implication: stability-optimal and test-optimal settings are diverging; we should rank by CV first, then test as tiebreaker.
3. Bin8 remains competitive under corrected kinsub settings.
   - `bin8 kinsub kbest50 corr095` (`test 0.6422`, `CV 0.6253 ± 0.0377`) is still among strongest imaging-only rows.
   - Causal implication: binWidth sensitivity is not eliminated; interaction with pruning/selection still matters.

#### Hypothesis update
- Updated hypothesis: current weak-signal narrative is outdated for ISPY2 imaging-only. There is moderate signal, but model ranking is highly sensitive to correlation-pruning regime; robust optimization should target high-CV settings (`~0.625-0.627`) and treat high test-only spikes (`~0.651`) as potentially split-specific unless replicated.

#### Next experiments
- Continue running pending jobs already submitted this cycle:
  - `all_mrmr20_corr097` (normal queue pending),
  - RF diagnostics (`all_rf_mrmr20_corr090`, `all_rf_kbest40_corr070/085`, `kinsub_rf_kbest40_corr095`) to test nonlinear capacity and whether CV improves without inflating variance.
- Keep queue floor at `8 normal + 8 burst` by topping up with non-duplicate train-only probes whenever burst jobs complete.

#### Actions taken
- Refilled queue repeatedly to maintain requested slot utilization under fast burst completions.
- Root-caused and corrected a submission-script bug that mis-labeled some `kinsub` jobs:
  - Missing line-continuation caused the second `--exclude-feature-regex` to execute as a shell command after training.
  - Renamed completed mislabeled outputs to truthful names:
    - `...kinsubonly_kbest40_corr095` -> `...all_kbest40_corr095`
    - `...kinsubonly_kbest40_corr085` -> `...all_kbest40_corr085`
    - `...kinsubonly_kbest40_corr080` -> `...all_kbest40_corr080`
    - `...kinsubonly_kbest40_corr075` -> `...all_kbest40_corr075`
  - Canceled affected in-flight mislabeled jobs and resubmitted corrected kinsub jobs.
- No push operations performed (local-only workflow preserved).

### Cycle 2026-02-27 11:46:50 CST (Fast Follow-up)
#### Results
- Additional burst probes completed quickly after prior cycle:
  - `ispy2_p2_phaseblind_bin100_f2d1_all_kbest40_corr060`: `test 0.6565`, `CV 0.6092 ± 0.0398`.
  - `ispy2_p2_phaseblind_bin100_f2d1_all_kbest40_corr070`: `test 0.6471`, `CV 0.6167 ± 0.0421`.
  - `ispy2_p2_phaseblind_bin100_f2d1_all_kbest40_corr099`: `test 0.6477`, `CV 0.6137 ± 0.0376`.
  - `ispy2_p2_phaseblind_bin100_f2d1_kinsubonly_kbest20_corr095`: `test 0.6479`, `CV 0.6156 ± 0.0375`.
  - `ispy2_p2_phaseblind_bin100_f2d1_kinsubonly_kbest60_corr095`: `test 0.6322`, `CV 0.6254 ± 0.0421`.
- Queue remains at requested occupancy (`8 normal + 8 burst` active at check time).

#### Insights learned
1. Very high held-out test spikes (`~0.648-0.656`) continue to coincide with lower CV means (`~0.609-0.617`).
   - Causal implication: those settings are likely less robust despite attractive single-split test values.
2. Best CV cluster remains around `0.625-0.627` with moderate test (`~0.626-0.642`).
   - Causal implication: robust signal exists, but objective choice (CV robustness vs test peak) must be explicit.

#### Hypothesis update
- Unchanged: prioritize CV-ranked settings for robustness; treat extreme test-only gains as provisional until replicated.

#### Next experiments
- Let currently running RF burst jobs finish; compare whether nonlinear models can improve CV mean without increasing variance.
- Keep topping up burst with non-duplicate train-only probes if jobs drain before RF finishes.

#### Actions taken
- Added corrective/top-up submissions to maintain `8/8` burst under rapid job turnover.
- Logged follow-up results and robustness interpretation.
