# Issue #151 — top-K features vs baseline

This workflow measures whether a **small set of globally ranked vessel
columns** adds signal on top of a **clinical + tumor size** baseline.

## Components

- `analysis/feature_ranking.py` — univariate AUC, XGBoost gain, and L2-LR score
  rankers (shared tidy schema).
- `analysis/top_k_arms.py` — builds baseline + top-K ablation arms with
  `explicit_model_columns`.
- `modeling/run_top_features_eval.py` — loads the feature superset, runs the ranker,
  writes `feature_ranking_global.csv`, then calls `run_ablation_matrix`.
- `configs/top_features_eval.yaml` — worked example matching Issue #118 cohort
  defaults (ISPY2, `bilateral_filter: false`).

## Leakage / interpretability default

The ranker is fit **once on all labeled rows** before cross-validation. This is
a deliberate **screening** default: it is fast, deterministic, and easy to
audit, but it means hold-out AUCs for the top-K arms are **optimistically biased**
relative to truly nested feature selection. Use the results to compare “a
handful of strong vessel columns vs baseline,” not for formal inference.

A stricter follow-up ranks inside each outer-fold training split only.

## Running

```bash
micromamba run -n vanguard python modeling/run_top_features_eval.py \
  --config configs/top_features_eval.yaml \
  --outdir experiments/issue151_run
```

Outputs live under `--outdir`: merged feature CSVs, per-arm run directories,
`ablation_summary.csv`, and `feature_ranking_global.csv`.

## Ablation arm schema

Arms may set `explicit_model_columns`: an ordered list of modeling columns
(after block filtering) kept for that run. Annotation columns (`case_id`,
`dataset`, …) and the label are always retained by `select_features` and must
**not** appear in `explicit_model_columns`.
