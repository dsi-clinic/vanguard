# UChicago vessel-GNN pCR on the cleaned pretreatment cohort (N=82)

Re-estimates the vessel-only pCR gate on the one-exam-per-patient pretreatment
cohort, replacing the legacy 181-exam manifest that mixed in post-treatment
follow-up images. Answers "how does the clean cohort change our performance?"
and "what is the HER2-specific performance?".

**Headline: the above-chance signal does not survive the cohort clean-up.** Both
headline models now have 95% CIs that include 0.5. On the legacy cohort they did
not. See "How to read this" before quoting any number.

## Result

Pooled OOF AUC over all 82 cases, bootstrap 95% CI (2000 case resamples),
seed-averaged over 5 seeds. `gate_auc_ci.csv`, `gate_forest.png`.

| model | this cohort (N=82) | legacy (N=179) |
|---|---|---|
| tabular 7-means logreg | **0.557 [0.427, 0.681]** | 0.614 [0.528, 0.693] |
| GNN baseline (voxel, 7-feat, floored) | **0.528 [0.399, 0.658]** | 0.616 [0.528, 0.699] |
| tabular 38-feat xgboost | 0.511 [0.377, 0.633] | 0.585 [0.499, 0.670] |
| tabular 38-feat logreg | 0.429 [0.304, 0.557] | 0.538 [0.450, 0.625] |

Gate (paired bootstrap, GNN − strongest tabular), `gate_paired_delta.csv`:

    dAUC = -0.028  95% CI [-0.108, +0.051]   P[dAUC>0] = 0.25   GATE NOT PASSED

The graph still does not beat the tabular bar — unchanged from the legacy
cohort, and now the bar itself is not distinguishable from chance.

## HER2-specific performance

Stratification of the *same* pooled OOF predictions, not a refit. `subtype_breakdown.csv`,
`subtype_forest.png`.

| subgroup | n | pCR/non | GNN baseline | tabular 7-means |
|---|---|---|---|---|
| HER2-positive | 31 | 15 / 16 | 0.487 [0.279, 0.702] | 0.583 [0.364, 0.796] |
| HER2-negative | 43 | 18 / 25 | 0.469 [0.287, 0.644] | 0.456 [0.283, 0.624] |
| HER2 unavailable | 8 | 0 / 8 | undefined | undefined |

Finer subtype axis: triple-negative (n=18) 0.623 / 0.636, HR+/HER2− (n=25) 0.278 /
0.246, HER2+ as above. **None of these subgroups separates from chance**; every CI
spans 0.5 and is ~0.4 AUC wide. At n=18-43 that is the expected width, so these
numbers rank the subgroups only in the weakest sense and must not be reported as
"the model works better on X".

## How to read this

1. **N halved (179 -> 82), so CIs widened by design.** Half-width goes from
   ~0.085 to ~0.13. A drop from 0.616 to 0.528 is *within* what resampling noise
   alone could produce, so this run does not by itself prove the legacy signal
   was an artifact.
2. **But it also no longer supports the earlier claim.** The Phase 1/2
   conclusion recorded in LAB_NOTEBOOK 2026-07-27 was "vessel kinetics predict
   pCR at ~0.61, CIs clear 0.5". On the clean cohort the CIs do not clear 0.5.
   That claim is now unsupported, whatever the cause.
3. **The two candidate causes are not yet separated.** Either (a) losing 97
   exams cost power, or (b) the legacy 0.61 was partly carried by post-treatment
   images, where a shrunk tumor co-occurs with the patient's pCR label and makes
   the task easier. These have different consequences and the run below
   distinguishes them — it has NOT been run yet.

**Recommended next check (cheap, reads only existing CSVs):** score the
legacy-trained model's saved OOF predictions restricted to these same 82 cases.
If legacy-on-82 is ~0.53, the drop is cohort composition (explanation b) and the
old number was inflated. If it is ~0.61, the drop is retraining on less data
(explanation a). This needs the legacy per-seed `predictions.csv` under
`/gpfs/data/karczmar-lab/workspaces/spencervenancio/experiments/gnn_uchicago_tier_sweep_floored`,
which exist.

## Cohort

`/gpfs/data/karczmar-lab/vanguard/dce2d_internal_ultrafast_pretreatment_cohort_v1/dce2d_internal_ultrafast_manifest.csv`
— 83 exams / 83 patients, 50 non-pCR / 33 pCR; 32 HER2+, 25 HR+/HER2−, 18
triple-negative, 8 subtype unavailable. A strict subset of the legacy 181-exam
manifest (83 overlap, 0 new, 98 dropped); folds and labels preserved exactly.

**Effective N is 82, not 83.** Exam
`uch_nac_1_3_978925353542321868351609024718208384487357325840` has no v5
centerline: its DCE is split across 8 HR series with no UFAST, so the paired
preprocessing pipeline could not run it. It is listed in the cohort's own
`paired_preprocessing_exclusions.csv` (`split_series_not_runnable`). It is
HER2-positive and non-pCR, which is why the built cohort is 49/33 and HER2+ is
31 rather than the manifest's 50/33 and 32.

Because the cache is built with an explicit `--cases` whitelist (see below),
this exam is filtered at *discovery* and so does **not** appear in
`processed/dropped_cases.json`. Its absence shows up as `num_graphs=82` against
an 83-row labels file.

## Two deliberate edge-case handlers

Flagged per the repo's fail-fast policy; see also `AUDITING_RESULTS.md`.

1. **Single-class subgroups report `auc=NaN`, `status=undefined_single_class`**
   rather than crashing or being skipped. All 8 subtype-unavailable patients are
   non-pCR, so AUC is genuinely undefined there. Note this missingness is
   perfectly correlated with outcome — worth raising with Anna, since it means
   subtype is not missing at random.
2. **Degenerate bootstrap draws are dropped and counted** in
   `n_boot_degenerate`. In small subgroups some resamples contain one class and
   have no AUC.

## Reproduce

Git commit `55f18d990f5fa60afb63d7853f3054bf48b10d75`, branch
`feat/pretreatment-cohort-83`, worktree `/ess/home/home1/t-9svena/vanguard-pretreat`.
Config: `configs/uchicago_gnn_tier_baseline_floored_pretreat.yaml` (identical to
`uchicago_gnn_tier_baseline_floored.yaml` except labels, cache, and the explicit
83-id `gnn_cases` whitelist — so old-vs-new is a cohort change, not a model change).

```bash
# 1. labels + folds from the cohort manifest  (head node, seconds)
python -m gnn.export_uchicago_labels \
  --dataset-root /gpfs/data/karczmar-lab/vanguard/dce2d_internal_ultrafast_pretreatment_cohort_v1 \
  --out /gpfs/data/karczmar-lab/workspaces/spencervenancio/uchicago/uchicago_labels_folded_pretreat_v1.csv

# 2. graph cache, restricted to the 83 cohort ids   (job 13437906, 4m37s)
CASES=$(tail -n +2 /gpfs/data/karczmar-lab/workspaces/spencervenancio/uchicago/uchicago_labels_folded_pretreat_v1.csv | cut -d, -f1 | paste -sd,) \
ROOT=/ess/scratch/scratch1/t-9svena/uchicago/preprocessing_out_v5/centerlines \
DCE_ROOT=/ess/scratch/scratch1/t-9svena/uchicago/preprocessing_out_v5/dce \
LABELS_PATH=/gpfs/data/karczmar-lab/workspaces/spencervenancio/uchicago/uchicago_labels_folded_pretreat_v1.csv \
CACHE_DIR=/ess/scratch/scratch1/t-9svena/uchicago/gnn_cache_voxel_richfeat_floored_pretreat83 \
NODE_MODE=voxel \
NODE_FEATURES=peak_time,peak_enhancement,time_to_enhancement,washin_slope,washout_slope,auc_positive,radius \
KINETIC_BASELINE_FLOOR_FRAC=0.05 \
sbatch --time=02:00:00 gnn/slurm/submit_gnn_build.slurm

# 3. training (5 seeds) + tabular bar, then the chained analysis
TRAIN=$(sbatch --parsable gnn/slurm/submit_gnn_uchicago_pretreat83.slurm)          # 13437995
TAB=$(CONFIG=configs/uchicago_gnn_tier_baseline_floored_pretreat.yaml \
      OUT_DIR=experiments/uchicago_pretreat83 \
      sbatch --parsable gnn/slurm/submit_tabular_bar_uchicago_floored.slurm)       # 13437996
sbatch --dependency=afterok:${TRAIN}:${TAB} gnn/slurm/submit_pretreat83_analysis.slurm  # 13437997
```

Inputs: cache
`/ess/scratch/scratch1/t-9svena/uchicago/gnn_cache_voxel_richfeat_floored_pretreat83`
(82 graphs, `kinetic_baseline_floor_frac=0.05`, `feature_source =
raw_dce_protocol_baseline_physical_time_all_modes_v4` — identical to the legacy
floored cache, so the two runs differ only in which cases are in them).
Per-seed training runs:
`/ess/scratch/scratch1/t-9svena/uchicago/experiments/gnn_pretreat83/baseline_seed{0..4}/`.

Two earlier submissions failed and were superseded; both were configuration
faults, not data faults:
- **13437831** — built against the centerline root without a whitelist, so it
  discovered all 179 legacy exams, dropped 97 as unlabeled (54%) and correctly
  tripped `max_missing_label_frac=0.1`. Fixed by whitelisting rather than by
  raising the threshold, which would have disarmed the guard for every later run.
- **13437969 / 13437974** — cache recorded the 83-case whitelist while the config
  still said `gnn_cases: null`, so the cache-manifest check refused the load.
  Fixed by writing the whitelist into the config, and by making
  `tabular/gnn_feature_baseline.py` pass `cases` through (it previously ignored a
  configured whitelist, which would have silently summarized a different case
  set than the GNN trained on).

## Files

| file | what |
|---|---|
| `gate_auc_ci.csv` | pooled OOF AUC + bootstrap CI per model |
| `gate_paired_delta.csv` | paired dAUC gate, GNN vs strongest tabular |
| `gate_forest.png` | forest plot of the above |
| `oof_predictions.csv` | per-case OOF probability per model — the input to the subgroup table |
| `subtype_breakdown.csv` | per-subgroup AUC + CI, HER2 and subtype axes |
| `subtype_forest.png` | forest plot of the HER2 axis |
| `tabular_baseline_features.csv` | 38 per-case vessel summary features |
| `tabular_baseline_results.json` | per-fold and pooled tabular AUCs |
