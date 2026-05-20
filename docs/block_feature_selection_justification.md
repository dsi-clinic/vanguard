# Block-Aware Feature Selection: Quantitative Justification

## Problem

This project organizes vessel and clinical features into five canonical blocks:

| Block | Description | Approximate size |
|-------|-------------|-----------------|
| `clinical` | Non-imaging case-level metadata | ~19 features |
| `tumor_size` | Tumor size and peritumoral shell summaries | ~10–15 features |
| `morph` | Whole-network morphometry aggregates | ~40 features |
| `graph` | Tumor-centered structural vessel features | variable (50–200+) |
| `kinematic` | Tumor-centered dynamic vessel features | variable (100–300+) |

When feature selection is applied **globally** (e.g., rank all features by
univariate AUC and keep the top *k*), two failure modes arise:

1. **Large blocks dominate the budget.**  Because `graph` and `kinematic` can
   have 10–20× more columns than `clinical`, a disproportionate share of the
   top-*k* slots is filled by the largest blocks.  Small but informative blocks
   like `clinical` or `tumor_size` may lose all representation.

2. **Intra-block redundancy wastes slots.**  Features within a block tend to
   measure related quantities (e.g., vessel caliber at multiple radii), so they
   are highly correlated.  A global ranker may select many near-duplicate
   features from the same block instead of diverse features from different
   blocks.

## Diagnostics

The script `scripts/analyze_block_selection_bias.py` quantifies these effects
on the actual project feature table.  It produces three outputs:

### 1. Block size imbalance

A bar chart and table showing how many numeric features belong to each block.
The ratio between the largest and smallest block is typically 10–20×.

**Output:** `block_sizes.csv`, `block_sizes.png`

### 2. Intra- vs inter-block correlation

A heatmap of mean |Spearman ρ| within and between blocks.  If intra-block
correlation is substantially higher than inter-block correlation, features
within a block carry redundant information.  Selecting many features from the
same high-correlation block wastes the selection budget on duplicates.

**Output:** `block_correlation.csv`, `block_correlation_heatmap.png`

### 3. Global top-k survival simulation

Ranks all features by univariate AUC (against the pCR label) and selects the
top *k* globally.  Then repeats with a block-aware budget that guarantees
every block at least ⌊k / n_blocks⌋ slots.  A side-by-side bar chart shows
which blocks survive under each strategy.

The expected result is that global top-k eliminates or severely under-
represents small blocks like `clinical` and `tumor_size`, while the block-
aware approach preserves representation from every measurement family.

**Output:** `feature_auc_ranking.csv`, `selection_survival.csv`,
`selection_survival.png`

## Running the analysis

```bash
micromamba activate vanguard
python scripts/analyze_block_selection_bias.py \
    experiments/<run_name>/features_labeled.csv \
    -o experiments/block_selection_analysis \
    --top-k 64
```

## Why this matters for ablation studies

The ablation experiments in this repository (e.g.,
`configs/independent_signal.yaml`, `configs/clinical_graph_ablation.yaml`)
test whether adding a feature block improves pCR prediction.  If the feature
selector has already silently dropped a block's features due to global
ranking, the ablation cannot measure that block's contribution — the "with
block" and "without block" arms become identical.

Block-aware selection ensures that when a block is toggled on, its features
actually reach the model, making the ablation comparison meaningful.

## References

- `features/__init__.py` — canonical block definitions and column-to-block
  mapping (`feature_block_for_column`)
- `scripts/feature_selection.py` — existing global (marginal) selection
  pipeline with its stated limitations (lines 13–17)
- `config.py` — `feature_select_mode` options (`global_topk` vs
  `block_kinematic`)
