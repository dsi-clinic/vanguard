# Tabular model outputs

Output root for all **tabular** (non–Deep Sets) experiments from the `vanguard` repo
(`/home/t-9svena/vanguard`). The repo holds code/configs only; run artifacts live here.

Layout: `tabular/<run>/` — one directory per run.

## Contents (as of 2026-06-25)

| Run dir | What it is | Source |
|---------|-----------|--------|
| `all_datasets_run/` | 5-arm × 2-family (LR, XGBoost) ablation, DUKE+ISPY1+ISPY2+NACT (n≈1491). Slurm job 12166646. See its README for results. | `run_ablation_matrix.py --config configs/all_datasets_gpfs.yaml` |
| `xgboost_interaction/` | XGBoost interaction / permutation-importance / PDP analysis. | `slurm/submit_xgboost_interaction.slurm` |
| `results/` | Aggregated summary CSVs/plots (feature coverage audit, issue118 baseline arms, independent-signal Q3, model-family robustness). | various analysis scripts |

## How runs land here

- `config.py` default `base_outdir` is the workspace root; tabular config YAMLs set
  `base_outdir: /gpfs/data/karczmar-lab/workspaces/spencervenancio/tabular`.
- Tabular Slurm scripts default `OUTDIR`/`OUT_ROOT` to `…/spencervenancio/tabular/<run>`.

## History

`all_datasets_run/`, `xgboost_interaction/`, and `results/` were relocated here from the repo
(`experiments/`, `results/`) on 2026-06-25, and the repo's output defaults were repointed to this
workspace. See `vanguard/CLAUDE.md` → "Experiment Outputs".
