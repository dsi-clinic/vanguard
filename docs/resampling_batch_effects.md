# Resampling & Batch Effects in ISPY2 Vessel Features

**Scope:** ISPY2 only, native (non-resampled) vs resampled vessel features.
**Last updated:** 2026-07-02

## Why we looked into this

Anna suspected a **batch effect** in ISPY2. The motivating puzzle: when the imaging
was **resampled** to a common voxel grid, the vessel **segmentations looked better**
(smoother, cleaner masks) but the **pCR models got worse**. If acquisition
differences (site / scanner / slice spacing) dominate the vessel features more than
the biological outcome (pCR), that is a batch effect that modeling has to reckon with.

## TL;DR — what we found

1. **There is a strong batch effect, and it is scanner-driven** (manufacturer + field
   strength), **not** slice-spacing-driven. You can predict the scanner vendor from the
   vessel features alone at AUC up to 0.92.
2. **The batch effect is nuisance variance, not an outcome confounder.** Site does not
   predict pCR (chi-square p = 0.81), so a model that "reads" the scanner off the
   features gains no real pCR shortcut.
3. **Resampling partially removes the scanner fingerprint but never erases it.** It
   helps most for geometry-based features (morph) and least for enhancement-dynamics
   features (kinematic), which are driven by contrast/sequence that voxel regridding
   cannot fix.
4. **The vessel features are weak pCR predictors** in this cohort regardless of
   native/resampled (best multivariate AUC ~0.60; morph near chance). Resampling's
   effect on the pCR signal at the feature-arm level is small.

## Data sources

All inputs (except scanner metadata, already on shared storage) have been copied to
`/gpfs/data/karczmar-lab/vanguard/batch_effect_inputs/` (md5-verified against the
originals; see the README there) so this is reproducible off personal/scratch storage.
The default paths baked into `scripts/submit_*_ispy2.sbatch` already point here.

| Thing | Path | ISPY2 n |
|---|---|---|
| Native feature matrix | `/gpfs/data/karczmar-lab/vanguard/batch_effect_inputs/features_full_labeled_native.csv` | 980 |
| Resampled feature matrix | `/gpfs/data/karczmar-lab/vanguard/batch_effect_inputs/features_full_labeled_resampled_808.csv` | 808 |
| Per-case z-spacing (native headers) | `/gpfs/data/karczmar-lab/vanguard/batch_effect_inputs/spacing_by_hospital.csv` (`z_spacing_mm`) | 980 |
| Scanner metadata | `/gpfs/data/karczmar-lab/MAMA-MIA-syn60868042/clinical_and_imaging_info.xlsx` (`dataset_info`) | 980 |

Feature arms: **morph (50) + graph (49) + kinematic (895)** = "vessel_all" (994).
Native features built from `centerlines_tc4d`; resampled from `resamples_centerlines_tc4d`
+ `resampled-tumor-masks`. Values genuinely differ between the two (85% of morph cells,
80% of graph, 53% of kinematic differ), confirming the resampled features are real
recomputations, not copies.

### Important caveat: 808 vs 980

The resampled feature matrix covers only **808 of the 980** ISPY2 cases. This is **not**
a resampling-coverage limit — all 980 resampled centerlines and masks exist on disk. The
808 is a deliberate **locked split** from the prior `independent_signal_resampled_ispy2`
experiment (`locked_split_resampled_ispy2.csv`). All native-vs-resampled comparisons
below intersect to the **matched 808** cases so the comparison is apples-to-apples.
Resampled features for the full 980 can be rebuilt from the existing resampled
centerlines if wanted.

## Analyses & results

### 1. PCA of native vessel features (`scripts/pca_ispy2_batch_effect.py`)

980 cases, 994 features, median-imputed + z-scored.

- PC1 explains 20.1% of variance, PC2 7.4% (features are highly redundant).
- **Site dominates PC1, outcome does not:**

  | Component | Factor | R² (variance explained) | p-value |
  |---|---|---|---|
  | PC1 | site | 0.109 | 2.7e-14 |
  | PC1 | pCR | 0.003 | 0.10 |
  | PC2 | site | 0.016 | 0.80 |
  | PC2 | pCR | 0.002 | 0.18 |

- **z-spacing does not align with the PCA structure**: r(PC1, z-spacing) = +0.01
  (p = 0.77), r(PC2) = +0.03 — even though z-spacing is 98% determined by site. So the
  site effect is real but is *not* the slice-thickness gradient.
- PC1's top loadings are kinematic **validity / has-signal fractions** — i.e. PC1 largely
  separates cases by *which kinematic measurements were even computable*, a
  measurability/acquisition signature.

**Conclusion:** a real site-level batch effect, far larger than any pCR signal, and not
explained by z-spacing. [confidence: high]

### 2. Is site a confounder? (pCR ~ site)

- Overall ISPY2 pCR rate 0.32. Per-site rates range 0.21–0.42 (n ≥ 20), consistent with
  sampling noise. Chi-square pCR ~ site: **p = 0.81** → site does **not** predict pCR.
- Site is nested within scanner manufacturer (21/22 sites use a single vendor:
  GE 601 / SIEMENS 252 / Philips 117), at 1.5T and 3.0T. So "site" here ≈ "scanner + field
  strength."

**Conclusion:** the batch effect is nuisance variance, not an outcome confounder.
[confidence: high]

### 3. Batch detectability (`scripts/batch_detectability_ispy2.py`)

Can a classifier predict the scanner from the features? Matched 808 cases, same folds,
native vs resampled. Metric: macro one-vs-rest ROC AUC (chance = 0.5). Representative
values (random forest):

| Arm | Predict manufacturer (native → resampled) | Predict field strength (native → resampled) |
|---|---|---|
| morph | 0.84 → 0.70 | 0.85 → 0.71 |
| graph | 0.83 → 0.73 | 0.83 → 0.77 |
| kinematic | **0.90 → 0.83** | 0.86 → 0.77 |
| vessel_all | **0.92 → 0.84** | 0.89 → 0.79 |

**Conclusion:** strong scanner fingerprint in every arm; resampling reduces it
(by ~0.07–0.16) but never removes it. Morph (geometry) is fixed most by resampling;
kinematic (contrast dynamics) is fixed least and stays highly scanner-identifiable
(AUC 0.83 even after resampling). [confidence: high]

### 4. Signal preservation (`scripts/signal_preservation_ispy2.py`)

Does resampling weaken the features' association with pCR? Matched 808 cases.

| Arm | Univariate mean \|AUC−0.5\| (nat → res) | Multivariate pCR AUC, RF (nat → res) |
|---|---|---|
| morph | 0.012 → 0.012 (no change) | 0.48 → 0.45 |
| graph | 0.020 → 0.032 (slightly up) | 0.54 → 0.56 |
| kinematic | 0.025 → 0.030 (slightly up) | 0.60 → 0.59 |
| vessel_all | 0.024 → 0.029 (slightly up) | 0.60 → 0.59 |

- The vessel arms are **weak pCR predictors** in both native and resampled
  (best ~0.60; morph at chance).
- Per-feature associations, if anything, edged *up* after resampling (consistent with
  mild denoising), but the magnitudes are near the noise floor (|AUC−0.5| ~0.03 = feature
  AUC ~0.53).

**Conclusion:** at the single-arm level, no strong evidence that resampling destroys
pCR signal — but the arms have little pCR signal to begin with, so this test has limited
power. [confidence: medium]

## Overall interpretation

- The batch effect is **real, strong, and scanner-driven**, and it is the dominant axis
  of variation in the vessel features — but it is **orthogonal to pCR**, so it is a
  nuisance rather than a leakage shortcut.
- Resampling behaves as expected: it strips a substantial chunk of scanner variance
  (good), most effectively for geometric features and least for enhancement kinetics.
- A genuine scientific tension for the project: the arm most useful for pCR
  (**kinematic**, the strongest but still modest predictor) is also the **most
  scanner-contaminated** and the one resampling helps least. Separating vessel biology
  from the scanner in the kinematic arm is the hard part, and spatial resampling is not
  the tool that does it.

## Recommended next steps

1. **ComBat harmonization by manufacturer** (fit on training folds only), then re-check
   pCR modeling. Because the batch is uncorrelated with pCR, ComBat is low-risk (little
   danger of scrubbing real signal) and directly targets the scanner fingerprint.
2. **Rebuild resampled features for the full 980** ISPY2 cases from the existing
   resampled centerlines, if full-cohort coverage is needed.
3. **Feature-selected native-vs-resampled comparison** (the pipeline's top-k / block
   selection) to quantify any resampling penalty on realistic vessel models.

## Scripts & artifacts

Scripts (in `scripts/`, all run via Slurm, `express` partition):
- `pca_ispy2_batch_effect.py` — PCA, scree, PC scatters, loadings, ANOVA.
- `batch_detectability_ispy2.py` — predict scanner from features (`--feature-prefixes`).
- `signal_preservation_ispy2.py` — per-feature + multivariate pCR signal (`--feature-prefixes`).

Output folders under `/ess/scratch/scratch1/t-9sbose/`:
- `ispy2_pca_batch_effect/`
- `ispy2_batch_detectability_{morph,kinematic,vessel_all}/` (and `ispy2_batch_detectability/` for graph)
- `ispy2_signal_preservation_{morph,kinematic,vessel_all}/` (and `ispy2_signal_preservation/` for graph)

See also the dated entry in `lab_notebook.md` (2026-07-02).
