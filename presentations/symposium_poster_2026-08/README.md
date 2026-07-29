# Symposium poster — UChicago DSI Summer Lab, August 2026

First draft of the final research poster for the University of Chicago Data Science Institute
Summer Lab symposium. 36 in × 24 in, landscape, `tikzposter`. Author: Spencer Venancio
(University of Wisconsin–Madison); advisor Dr. Anna Woodard; work performed at UChicago DSI.

The poster is organised around one question in two parts — **(1) is there signal in the blood
vessels?** and **(2) does the graph representation add anything beyond that?** — and answers them
in two separate blocks, badged ① and ②.

## Build

```bash
./build.sh          # one build, ~4 s
./build.sh watch    # rebuild on every save until Ctrl-C
./build.sh png      # build, then write preview.png (VS Code previews PNGs natively)
```

`build.sh` sets the two paths below and calls `latexmk`, which runs `pdflatex` as many times as
the layout needs. To drive the compiler yourself instead:

```bash
export PATH=/ess/scratch/scratch1/t-9svena/envs/perl/bin:/ess/scratch/scratch1/t-9svena/texlive/2026/bin/x86_64-linux:$PATH
pdflatex poster.tex     # run twice; second pass settles the tikzposter layout
```

There is no system TeX on this cluster. The TeX Live at
`/ess/scratch/scratch1/t-9svena/texlive/2026` was created with the CTAN net installer
(`scheme-small`, no docs/src) plus `tlmgr install tikzposter a0poster
collection-fontsrecommended`. The conda-forge perl at `/ess/scratch/scratch1/t-9svena/envs/perl`
has to come **first** on `PATH`: `latexmk` and `tlmgr` are Perl scripts and the system perl is
missing `List::Util` and `Scalar::Util`. Any TeX Live with `tikzposter`, `a0poster` and the URW
base-35 fonts will build this file — including Overleaf's, unmodified.

### Live editing

Closest thing to Overleaf on this cluster: VS Code Remote-SSH (already in use here) plus the
**LaTeX Workshop** extension, installed into the remote. It gives build-on-save, a PDF preview
tab, and SyncTeX jumps between source and PDF. It needs to be told where the compiler is —
add to the remote `settings.json`:

```jsonc
"latex-workshop.latex.tools": [{
  "name": "latexmk-scratch",
  "command": "/ess/scratch/scratch1/t-9svena/texlive/2026/bin/x86_64-linux/latexmk",
  "args": ["-pdf", "-synctex=1", "-interaction=nonstopmode", "%DOC%"],
  "env": {"PATH": "/ess/scratch/scratch1/t-9svena/envs/perl/bin:/ess/scratch/scratch1/t-9svena/texlive/2026/bin/x86_64-linux:/usr/bin:/bin"}
}],
"latex-workshop.latex.recipes": [{"name": "latexmk", "tools": ["latexmk-scratch"]}],
"latex-workshop.latex.autoBuild.run": "onSave",
"latex-workshop.view.pdf.viewer": "tab"
```

Without the extension: run `./build.sh watch` in a terminal and keep `preview.png` open in a
VS Code tab (regenerate it with `./build.sh png`) — the tab refreshes when the file changes.

Compiles clean (one page, 36 in × 24 in) as of git commit `55f18d9` on branch
`feat/kinetic-floor`.

## Version control

`presentations/` is gitignored repo-wide (`.gitignore:234`), and nothing under it is tracked —
including the existing `gnn_methods_2026-07` deck. So this directory is local-only by
convention, and none of it is on a branch. Back it up somewhere outside the repo before the
symposium.

The two figure scripts it depends on *are* in tracked directories and are currently untracked
files:

- `analysis/poster_pipeline_figure.py`
- `analysis/poster_result_figures.py`
- `gnn/slurm/submit_poster_pipeline_figure.slurm`

They were written without moving `HEAD` (other agents are working in this checkout). They are a
distinct feature from `feat/kinetic-floor` and depend on nothing unmerged, so they belong on a
fresh branch off `origin/main` — e.g. `analysis/poster-figures`.

## Figures

| File | Made by | Source data |
|---|---|---|
| `pipeline.png` | `analysis/poster_pipeline_figure.py` via `gnn/slurm/submit_poster_pipeline_figure.slurm` (Slurm 13381288, `GRAPH_WINDOW=28`) | Case `her2_naclike_1_3_1119430…`, `preprocessing_out_v5/{centerlines,dce}`. Panels 1–4 are real data; panel 5 is schematic. |
| `q1_signal.png` | `analysis/poster_result_figures.py` | `experiments/uchicago_all_models/all_models_auc_ci.csv` |
| `q2_graph_delta.png` | `analysis/poster_result_figures.py` | `experiments/uchicago_tabular_bar_floored/gate_paired_delta.csv`, `experiments/uchicago_graph_mode_floored/graph_mode_paired_delta.csv` |
| `graph_representations.png` | copy of `presentations/gnn_methods_2026-07/figures/three_representations.png` (`make_figures.py`) | Abstract schematic — illustrative coordinates, not real anatomy |
| `contrast_pretrain_arch.png` | copy of `docs/design/contrast_pretrain_arch.png` (`analysis/plot_contrast_pretrain_flow.py`) | Schematic |
| `uw_madison_logo.pdf` | `rsvg-convert` of the UW–Madison crest-and-wordmark lockup | Home-institution logo, header only |

Regenerate the data figures:

```bash
sbatch gnn/slurm/submit_poster_pipeline_figure.slurm            # pipeline.png (needs raw 4D DCE)
python -m analysis.poster_result_figures \
    --out-dir presentations/symposium_poster_2026-08            # q1_signal.png, q2_graph_delta.png
```

## Every number on the poster

Each is also tagged with a `%` comment beside it in `poster.tex`.

| Number | Value on poster | Source |
|---|---|---|
| Acquisition of the pipeline case | 24 frames, 119.5 s, 5 precontrast | `experiments/dce_curve_example/README.md` |
| Vessel / centerline voxel counts in the pipeline figure | 53,624 / 5,669 | printed by `analysis/poster_pipeline_figure.py` for that case |
| Node feature list (7) | peak time, peak enhancement, time to enhancement, wash-in slope, wash-out slope, positive AUC, radius | `gnn/data_loader.py` `_FEATURE_ATTR`; `gnn/DESIGN_segment_graph.md` §4 |
| Edge feature list | segment length, tortuosity, volume, radius/curvature statistics, mean segment kinetics | `gnn/DESIGN_segment_graph.md` §4.1–4.2 |
| Curve → 6 scalars, edges carry no direction | — | `LAB_NOTEBOOK.md` 2026-07-28 (DCE curve example) |
| UChicago cohort size | N = 179 | `experiments/uchicago_all_models/README.md` |
| Pooled OOF AUC + 95% CI, all arms in `q1_signal.png` | junction 0.624, voxel 0.616, tabular 7-means 0.614, 2-feature control 0.492 | `experiments/uchicago_all_models/all_models_auc_ci.csv` |
| Bootstrap lower bounds clear 0.5 | 0.526–0.538 | `experiments/uchicago_graph_mode_floored/README.md` |
| I-SPY2 cohort size | n = 808 | `results/README.md` (cohort note) |
| I-SPY2 clinical + tumour size (LR) | 0.571 ± 0.046 | `results/issue118_baseline_arms_summary.csv` |
| I-SPY2 + all vessel features (XGB) | 0.606 ± 0.028 | `results/issue118_baseline_arms_summary.csv` |
| I-SPY2 paired LR gain | +0.024, p = 0.025 | `results/README.md` (#118 bottom line) |
| Paired ΔAUC, graph GNN − tabular | +0.002, 95% CI [−0.041, +0.046] | `experiments/uchicago_tabular_bar_floored/gate_paired_delta.csv` |
| Paired ΔAUC, junction/segment − voxel | +0.008 / −0.002 | `experiments/uchicago_graph_mode_floored/graph_mode_paired_delta.csv` |
| Conditioning fix, pooled AUC | 0.49 → 0.61 | `LAB_NOTEBOOK.md` 2026-07-27 (correction) |
| Floor-affected voxels | median 0.07% of vessel support per case | `LAB_NOTEBOOK.md` 2026-07-28 (cohort floor summary) |
| Unlabelled pretraining cohort | ≈1,000 graphs | `docs/design/contrast_pretraining.md` §2, §8a |
| Pretraining gates (i)–(iii) | — | `docs/design/contrast_pretraining.md` §7 |
| Tang et al., ICLR 2022 | — | `docs/design/contrast_pretraining.md` §1 |

Nothing on the poster is estimated, rounded toward a hoped-for value, or carried over from memory.

## Pending results (`TBD` blocks)

Two blocks are marked `TBD` on the poster. Both are sized and positioned as the finished content
will be, so dropping the number in will not move the layout.

1. **"Gate results on ultrafast data"** (contrast-pretraining column). Fills with: held-out
   forecasting MAE for the GNN forecaster against the two trivial baselines (last frame carried
   forward, per-node temporal mean) and against the graph-free per-node forecaster — gates §7.i
   and §7.ii — and then pretrained-vs-random-initialisation pCR AUC over ≥5 seeds (§7.iii). The
   pilot is implemented and validated end to end on Duke placeholder graphs; it has not been run
   on UChicago ultrafast data.
2. **"Sub-question 2 with temporal node representations"** (answer ② column). Fills with: the
   paired bootstrap ΔAUC of a graph model whose nodes read the full enhancement curve, against the
   same 7-mean tabular bar on the same folds. This is the push described in the poster prompt to
   get graph AUCs above tabular. **As of today the repo does not support that claim** — the
   measured paired ΔAUC is +0.002 [−0.041, +0.046], and no graph mode separates from the voxel
   graph. The poster says "NOT YET" and shows the intervals; it does not pre-write a win.

## Known polish items for the next pass

- `graph_representations.png` and `contrast_pretrain_arch.png` were made for a slide deck. Their
  in-figure labels are too small to read at three feet in a poster column. Both need regenerating
  at poster scale (larger fonts, no per-panel captions — the body text already carries them).
- The bottom of the answer-① column has roughly two inches of free space, deliberately left for
  whatever sub-question 1 result lands this week.
- The pipeline figure's panel 4 crop (`GRAPH_WINDOW=28`, seeded by the busiest branch point) is
  legible but sits near the skin line on this case. Worth trying a couple of other cases with
  `CASES="<case_id>" sbatch gnn/slurm/submit_poster_pipeline_figure.slurm`.
