# Five-minute symposium talk

This Quarto reveal.js deck presents the vessel-GNN project to a nontechnical audience.
The storyline is based on `LAB_NOTEBOOK.md`, `docs/design/PLAN_weight_transfer.md`,
and `presentations/symposium_poster/`.

Build the HTML deck from this directory. The cluster's `/apps/default/bin/quarto`
launcher currently points to a missing YAML resource; the existing local Quarto
1.4.557 installation works:

```bash
/ess/home/home1/t-9svena/quarto-1.4.557/bin/quarto render slides.qmd
```

Speaker notes are embedded in `slides.qmd` and available in reveal.js presenter view.
The talk is written for approximately five minutes, including the title slide.

The central reported comparisons are:

- Simple seven-feature vessel averages: pooled out-of-fold AUC 0.614.
- Baseline voxel GNN: pooled out-of-fold AUC 0.616.
- GNN with all edges removed: pooled out-of-fold AUC 0.618.
- Protocol-audit bracket for the vessel signal: approximately 0.539–0.613.

The self-supervised forecasting slide is explicitly labeled as planned work, not a result.
