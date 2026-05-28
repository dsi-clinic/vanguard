# Clinical Graph XGBoost Ablation

This config runs the six-arm clinical/graph ablation with the XGBoost model
family.

Run locally from the repository root:

```bash
python run_ablation_matrix.py \
  --config configs/clinical_graph_ablation_xgb.yaml \
  --outdir experiments/clinical_graph_ablation_xgb
```

Or submit through Slurm:

```bash
CONFIG=configs/clinical_graph_ablation_xgb.yaml \
OUTDIR=experiments/clinical_graph_ablation_xgb \
sbatch slurm/submit_clinical_graph_ablation.slurm
```

The primary output for the reported table is:

- `experiments/clinical_graph_ablation_xgb/ablation_summary.csv`

When per-fold metrics are available, the run also writes:

- `experiments/clinical_graph_ablation_xgb/ablation_fold_auc.csv`
