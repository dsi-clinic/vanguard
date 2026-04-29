# Deep Sets point sets and point features (issue #119)

Design note for **better tumor-local point sets and point features** on the existing Deep Sets path. Scope here is documentation, path conventions, and **spatial alignment QA** figures—not changes to model architecture or training code.

Related code:

- [`build_deepsets_dataset.py`](../../build_deepsets_dataset.py) — builds one point set per case and the manifest
- [`deepsets_data.py`](../../deepsets_data.py) — loads serialized point tensors for training
- [`train_deepsets.py`](../../train_deepsets.py) — Deep Sets training entrypoint
- [`graph_extraction/README.md`](../../graph_extraction/README.md) — upstream centerline / graph outputs
- [`slurm/submit_deepsets_pipeline.sh`](../../slurm/submit_deepsets_pipeline.sh) and [`slurm/README.md`](../../slurm/README.md) — Slurm-backed dataset build and train

---

## 1. Audit: current builder and training contract

### Per-point payload (serialized `.pt`)

Each case file is a `dict` with at least:

| Field | Meaning |
|--------|---------|
| `x` | `float32` tensor of shape `[num_points, num_features]`. Columns depend on `model_params.deepsets_point_feature_set`. |
| `y` | Scalar label tensor. |
| `case_id` | String study identifier. |
| `feature_names` | Ordered list aligned with columns of `x`. |
| `local_radius_mm`, `tumor_equiv_radius_mm` | Tumor-local inclusion radius metadata (case-level). |
| `num_points` | Number of rows in `x`. |
| `used_fallback_nearest_points` | Whether the nearest-64 fallback ran when no points passed the distance filter. |
| `point_feature_set` | Active feature regime (`baseline`, `geometry_topology`, `geometry_topology_dynamic`). |
| `kinetic_timepoint_count` | Number of loaded timepoints for dynamic regime (0 otherwise). |

Feature regimes currently implemented in [`build_deepsets_dataset.py`](../../build_deepsets_dataset.py):

- `baseline` -> `["curvature_rad"]`
- `geometry_topology` -> signed-distance shells, topology flags, centroid offsets, support radius
- `geometry_topology_dynamic` -> `geometry_topology` plus voxelwise dynamic kinetics and reference-relative scalars

**Not stored per point (but computed during the build):**

- Signed distance to the tumor boundary (mm) is used only to **filter** points (`signed_distance_mm <= local_radius_mm`) and for fallback ordering; it is **not** written into `x`.
- Voxel `(x, y, z)` coordinates are **not** saved; order follows `numpy.argwhere` on the full skeleton mask, then the inclusion rule.

### Manifest (`deepsets_manifest.csv`)

Written by the builder with columns:

`case_id`, `set_path`, `label`, `dataset`, `num_points`, `local_radius_mm`, `tumor_equiv_radius_mm`, `used_fallback_nearest_points`, `point_feature_set`, `kinetic_timepoint_count`.

Training requires `case_id`, `set_path`, and the label column from config (`data_paths.deepsets_label_column`, default `label`). Feature layout consistency is validated at collate time via `feature_names` + tensor width checks in [`deepsets_data.py`](../../deepsets_data.py).

### Upstream artifacts and how the builder uses them

The builder always reads the exam skeleton mask (`{case_id}_skeleton_4d_exam_mask.npy` under `centerline_root / dataset / case_id`) and tumor mask. Additional artifacts are consumed by feature regime:

- `geometry_topology` / `geometry_topology_dynamic`: optional `{case_id}_skeleton_4d_exam_support_mask.npy` for local vessel support radius (`support_radius_mm`, `support_radius_available`).
- `geometry_topology_dynamic`: vessel 4D time-series under `data_paths.vessel_segmentation_root`, shape-aligned to skeleton via [`deepsets_volume_align.py`](../../deepsets_volume_align.py). If alignment fails, dynamic features are zeroed for that case (`kinetic_signal_ok=0`).

Not consumed by this path today:

- `{case_id}_skeleton_4d_exam_support_mask.npy` — vessel support used for local radius in graph extraction
- `{case_id}_morphometry.json` — segment-level geometry and graph primitives
- `{case_id}_tumor_graph_features.json` — tumor-centered summaries (shells, topology, kinetics)
- The multi-timepoint inputs used for kinematic features in the graph pipeline (clinical DCE NIfTIs vs vessel-segmentation NPZs, depending on cohort layout)

The builder **does** load the tumor mask and forces **shape agreement** with the skeleton using the same axis-permutation strategy as [`load_tumor_mask_zyx`](../../features/tumor_size.py).

### Slurm

Full cohort dataset builds should run **via Slurm**, not on the headnode. Use [`slurm/submit_deepsets_pipeline.sh`](../../slurm/submit_deepsets_pipeline.sh) as described in [`slurm/README.md`](../../slurm/README.md).

---

## 2. Canonical `/net` paths (cluster layout)

These paths were checked on the shared filesystem; they match [`configs/deepsets_ispy2.yaml`](../../configs/deepsets_ispy2.yaml) defaults for ISPY2.

| Path | Role |
|------|------|
| `/net/projects2/vanguard/MAMA-MIA-syn60868042/images/<case_id>/` | **Clinical DCE** phases: `{case_id}_0000.nii.gz`, `_0001`, … (`_0000` = pre-contrast in this layout). |
| `/net/projects2/vanguard/MAMA-MIA-syn60868042/segmentations/expert` | Tumor masks `{case_id}.nii.gz` (not DCE). |
| `/net/projects2/vanguard/vessel_segmentations/ISPY2/<case_id>/images/` | **`{case_id}_????_vessel_segmentation.npz`** — time series fed to tc4d; **same lineage** as the saved exam skeleton. |
| `/net/projects2/vanguard/centerlines_tc4d/studies/ISPY2/<case_id>/` | Skeleton, support, morphometry, tumor graph JSONs. |
| `/net/projects2/vanguard/centerlines` | Legacy `.vtp` centerlines (not used for this Deep Sets path). |
| `/net/projects2/vanguard/centerlines_4d` | Alternate centerline layout (not DCE). |
| `/net/projects2/vanguard/Duke-Breast-Cancer-MRI-Supplement-v3` | DUKE supplement data (different cohort). |

**Alignment script defaults** (see [`scripts/deepsets_alignment_check.py`](../../scripts/deepsets_alignment_check.py)):

- `--dce-root` → `/net/projects2/vanguard/MAMA-MIA-syn60868042/images`
- Clinical phase file: `{dce_root}/{case_id}/{case_id}_{time:04d}.nii.gz`
- `--use-vessel-segmentation-phases` with `--vessel-segmentation-root` → `/net/projects2/vanguard/vessel_segmentations/ISPY2` for NPZ phases under `<case_id>/images/`.

---

## 3. Implemented point-feature contract

Feature names are source-of-truth from `deepsets_point_feature_names()` in [`build_deepsets_dataset.py`](../../build_deepsets_dataset.py), and serialized into each case payload as `feature_names`.

Current regimes:

- `baseline`: `curvature_rad`.
- `geometry_topology`: signed distance, absolute distance, in-tumor flag, shell one-hot bins, node degree flags, xyz centroid offsets, support EDT radius, support-availability flag.
- `geometry_topology_dynamic`: all geometry/topology features plus arrival/peak timing, enhancement amplitudes, wash-in/out slopes, positive AUC, reference-relative peak/AUC, and validity flags.

**Dynamic features:** clinical DCE and vessel NPZs answer **different** questions—clinical overlays show anatomy; NPZ phases are **grid-identical** to tc4d inputs. Any new feature should state which grid it uses.

---

## 4. Spatial alignment QA

Serialized Deep Sets `.pt` files do not store voxel coordinates. QA overlays **reload** the skeleton `.npy` and tumor mask with the same shape-matching rules as the builder, then sample the chosen volume at the same indices.

### How to regenerate figures

From the repo root, with the **vanguard** environment (`micromamba activate vanguard` on the cluster):

```bash
export MPLCONFIGDIR=/tmp/$USER-mpl  # optional; avoids home cache permission issues
PYTHONPATH=. python scripts/deepsets_alignment_check.py \
  --case-ids ISPY2_100899,ISPY2_102011,ISPY2_102212 \
  --time-index 1
```

Vessel-segmentation (tc4d input grid) example:

```bash
PYTHONPATH=. python scripts/deepsets_alignment_check.py \
  --case-ids ISPY2_100899 \
  --time-index 1 \
  --use-vessel-segmentation-phases
```

Outputs default to `docs/deepsets_issue119/figures/`.

### Clinical DCE overlays (phase `0001`, signed distance coloring)

Axial slice at **z = index of maximum in-slice tumor area**; skeleton points within ±1 slice; tumor contour in red.

![ISPY2_100899 clinical alignment](figures/alignment_clinical_ISPY2_100899_z147_t0001.png)

![ISPY2_102011 clinical alignment](figures/alignment_clinical_ISPY2_102011_z132_t0001.png)

![ISPY2_102212 clinical alignment](figures/alignment_clinical_ISPY2_102212_z154_t0001.png)

### Vessel segmentation overlay (same case, same z; NPZ phase 0001)

![ISPY2_100899 vessel segmentation alignment](figures/alignment_vessel_seg_ISPY2_100899_z147_t0001.png)

### Verdict

For **ISPY2_100899**, **ISPY2_102011**, and **ISPY2_102212** at post-contrast index `0001`:

- NIfTI DCE volumes **match the skeleton grid** after the same three axis permutations used for the tumor mask in training data prep.
- On the chosen axial slices, **skeleton points fall on enhancing vessel structure** near the tumor contour, without obvious global translation between mask contour and perfused tissue.

We therefore treat **clinical DCE alignment with saved skeleton and expert tumor mask as trustworthy for ISPY2** under the current preprocessing assumptions. Any new DCE-sampled feature should still re-run this check when adding cohorts or changing mask or centerline pipelines.

---

## 5. Reproducibility checklist (issue #119 deliverable)

1. Build a manifest with the intended feature regime config (for example `configs/deepsets_ispy2_pointfeat_geom_topo.yaml` or `configs/deepsets_ispy2_pointfeat_geom_topo_dynamic.yaml`).
2. Verify one payload contract on disk:
   - `x.shape[1] == len(feature_names)`
   - `point_feature_set` matches config
   - dynamic runs expose nonzero `kinetic_timepoint_count` where vessel 4D is available
3. Run alignment QA overlays with [`scripts/deepsets_alignment_check.py`](../../scripts/deepsets_alignment_check.py) for representative cases and archive figures under `docs/deepsets_issue119/figures/`.
4. Train with [`train_deepsets.py`](../../train_deepsets.py) using the same manifest and keep `write_config_snapshot` output for provenance.
5. Record config path, git SHA, and generated figure names in experiment notes for auditability.

---

## 6. Out of scope (this note)

- Changing [`deepsets_model.py`](../../deepsets_model.py) architecture
