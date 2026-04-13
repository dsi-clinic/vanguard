# Features Notes

## Research purpose

The project aims to predict pathologic complete response (pCR) from vessel structure and vessel dynamics around the tumor, beyond what is already explained by clinical variables and tumor size. In practice, feature engineering should prioritize stable, interpretable vessel features that improve signal over the `clinical + tumor_size` baseline.

## Canonical blocks

- `clinical`: non-imaging case-level and tumor metadata
- `tumor_size`: tumor size and peritumoral shell-size summaries from the tumor mask
- `morph`: whole-network morphometry aggregates from the centerline graph
- `graph`: tumor-centered structural vessel features
- `kinematic`: tumor-centered dynamic vessel features over time

## Block inventory

### `tumor_size`

- already first-order:
  - tumor voxel count
  - per-radius region voxel counts
  - shell voxel counts
  - tumor mask existence/loading flags
- already second-order:
  - very little explicit second-order engineering beyond shell decomposition
- looks redundant:
  - nested region counts and shell counts are mechanically correlated
- clearly missing:
  - tumor-scale normalization
  - shell-to-tumor ratios
  - compact descriptors of local shell expansion

### `morph`

- already first-order:
  - segment count, bifurcation count
  - length, tortuosity, volume, curvature, radius, angle summary stats
- already second-order:
  - duplicate fraction and invalid-count bookkeeping
- looks redundant:
  - repeated `sum` / `mean` / `std` / `max` summaries for highly correlated raw quantities
- clearly missing:
  - heterogeneity summaries like coefficient of variation and quantile-spread features
  - normalized branching density

### `graph`

- already first-order:
  - node and edge totals
  - total length and volume burden
  - shell-local counts and burden summaries
  - boundary-crossing burden
- already second-order:
  - shell densities, normalized ratios, directional summaries, log transforms, derived graph ratios
- looks redundant:
  - raw counts plus ratios plus `log1p` transforms for the same primitives
  - multiple overlapping shell-burden views
- clearly missing:
  - compact core-vs-periphery concentration features
  - simple near-boundary branching enrichment summaries
  - stronger crossing-vs-near-tumor normalization

### `kinematic`

- already first-order:
  - time to enhancement, time to peak, peak enhancement, washin, washout, auc
  - per-shell hurdle summaries
- already second-order:
  - shell contrasts
  - multi-scale gradients
  - reference-normalized kinetics
  - propagation summaries
  - temporal heterogeneity
  - boundary-crossing dynamic burden
  - topology-kinetic coupling
- looks redundant:
  - similar information appears repeatedly as shell summaries, normalized summaries, and contrasts
- clearly missing:
  - compact curve-shape summaries
  - direct core-vs-periphery timing and enhancement descriptors
  - simpler near-tumor coherence biomarkers

## Proposed Project 1 features

The goal for Project 1 should be mostly second-order feature engineering, not more raw first-order counts.

| feature | block | feature type | order | reason |
|---|---|---|---|---|
| `tumor_size_equiv_radius_vox` | `tumor_size` | scale transform | second-order | more stable tumor scale than raw voxel count alone |
| `tumor_size_shell_0_2_over_tumor` | `tumor_size` | normalization | second-order | captures immediate peritumoral region size relative to tumor size |
| `tumor_size_outer_to_inner_shell_ratio` | `tumor_size` | shell contrast | second-order | summarizes how quickly local region size expands away from the tumor |
| `morph_seg_length_cv` | `morph` | heterogeneity | second-order | more informative than raw length std alone |
| `morph_radius_cv` | `morph` | heterogeneity | second-order | captures caliber variability across the vessel network |
| `morph_bifurcation_density_per_length` | `morph` | topology density | second-order | branching burden normalized by available vessel extent |
| `graph_core_to_periphery_length_ratio` | `graph` | shell contrast | second-order | tests whether vessel length is concentrated near the tumor |
| `graph_core_to_periphery_volume_ratio` | `graph` | shell contrast | second-order | burden-weighted version of the same concentration idea |
| `graph_near_branching_bias` | `graph` | topology contrast | second-order | asks whether branch complexity is enriched near the tumor boundary |
| `graph_crossing_fraction_of_near_burden` | `graph` | boundary interaction | second-order | isolates crossing burden relative to local available burden |
| `kinematic_core_to_periphery_tte_delta` | `kinematic` | timing contrast | second-order | measures whether near-tumor vessels enhance earlier or later |
| `kinematic_core_to_periphery_peak_ratio` | `kinematic` | intensity contrast | second-order | compares near-tumor enhancement strength with outer shells |
| `kinematic_near_washin_to_washout_ratio` | `kinematic` | curve shape | second-order | compact description of rise-versus-decay behavior |
| `kinematic_crossing_early_fraction` | `kinematic` | boundary-dynamic interaction | second-order | tests whether crossing vessels are preferentially early-enhancing |
| `kinematic_arrival_delay_dispersion_near` | `kinematic` | heterogeneity | second-order | summarizes coherence versus disorder in near-tumor timing |

## Recommended first pass

If Project 1 stays small, start with:

- `morph_seg_length_cv`
- `morph_radius_cv`
- `graph_core_to_periphery_length_ratio`
- `graph_crossing_fraction_of_near_burden`
- `kinematic_core_to_periphery_tte_delta`
- `kinematic_near_washin_to_washout_ratio`

These are compact, interpretable, and aligned with the README goal of finding vessel signal beyond `clinical + tumor_size`.
