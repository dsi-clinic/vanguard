### Exploratory Data Analysis

- Cohort composition: patients per site/dataset (`patients_per_site.png`)
- Age distributions + stratified by pCR and subtype (`age_hist.png`, `age_vs_pcr.png`, `age_vs_subtype.png`)
- Menopausal status distribution and pCR breakdown (`menopausal_status.png`, `pcr_by_menopausal.png`)
- Laterality audit and association with pCR (`laterality_counts.png`, `pcr_by_laterality_counts.png`)
- Breast density (known values only) and missingness summary (`breast_density_known.png`, `missing_percent_bar.png`)
- Scanner/acquisition variation:
  - Field strength counts (`field_strength_counts.png`)
  - Top manufacturers (`manufacturer_top.png`)
  - Echo/repetition time histograms and per-site variability (`echo_time_hist.png`, `repetition_time_hist.png`, `echo_time_by_site_box.png`, `repetition_time_by_site_box.png`)

**Tables**
- `missing_summary.csv` — % missing by column
- `field_strength_counts.csv`, `manufacturer_counts.csv`, `laterality_counts.csv`

**Where to find them**
- Figures → `eda_out/figs/`
- Tables  → `eda_out/tables/`

**Implementation notes**
- Laterality derived from `imaging_data.bilateral`.
- Breast density largely missing (60/1506 non-missing); plotted A–D only.