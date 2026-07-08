#!/usr/bin/env python3
"""Duke vessel-annotation QC -- Phase A: alignment check + 3D viz + distance histograms.

Goal (before we pick a distance tolerance for the coverage metric):
  For a handful of cases, put OUR extracted centerline (skeleton) and the
  RADIOLOGIST vessel annotation into the same physical space, look at them in 3D,
  and measure how far each annotated vessel voxel is from our nearest centerline
  voxel. Those distances tell us what tolerance means "same vessel, slightly
  offset" vs "actually missed".

Why plotly instead of pyvista: this cluster has no X server / OSMesa, so pyvista
segfaults when it tries to render. plotly builds a self-contained interactive HTML
that renders in YOUR browser (WebGL) -- same rotate/zoom experience, no server.

Key data facts (see memory: duke-vessel-qc-locations):
  - Annotation "Dense_and_Vessels" NRRD is a Slicer labelmap: label 1 = Vessels,
    label 2 = Dense/FGT. Vessel ground truth = voxels == 1.
  - Our segmentation .npz (key "vessel") is FLOAT PROBABILITIES in [0,1], 4 DCE
    timepoints. We reduce to one mask by taking the max across timepoints ("vessel
    present at any phase") then thresholding -- most generous for a recall metric.
  - Our skeleton .npy is already a binary centerline mask on the same voxel grid.
  - None of the .npy/.npz carry an affine; the physical grid comes from the
    MAMA-MIA source NIfTI (…_0000.nii.gz).
  - run_summary says the numpy masks use layout "yxz", but we do NOT trust that
    string -- we resolve the array->grid mapping empirically (below).
"""

import json
import os
import re
import sys

os.environ["MPLBACKEND"] = "Agg"  # matplotlib without any display
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import SimpleITK as sitk
from scipy.spatial import cKDTree

# ---- configurable locations ------------------------------------------------
# Every path below can be overridden with an environment variable, so anyone who
# pulls this repo can point the scripts at their own data/output locations
# WITHOUT editing code. The defaults target the karczmar-lab cluster layout.
# See this directory's README.md for how to download the Duke annotation data
# and which variables to set.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env_path(var, default):
    """Return $var if set (with ~ and $VARS expanded), else the given default."""
    return os.path.expandvars(os.path.expanduser(os.environ.get(var, default)))


# Radiologist vessel annotations (Duke supplement NRRDs, downloaded from TCIA -- see README).
ANN_ROOT = _env_path(
    "DUKE_ANN_ROOT",
    os.path.join(
        REPO_ROOT,
        "data",
        "duke_vessel_annotations",
        "PKG - Duke-Breast-Cancer-MRI-Supplement-v3",
        "Duke-Breast-Cancer-MRI-Supplement-v3",
        "Segmentation_Masks_NRRD",
    ),
)
# Our vessel-probability segmentations (pipeline output).
SEG_ROOT = _env_path(
    "DUKE_SEG_ROOT",
    "/gpfs/data/karczmar-lab/workspaces/saritbose/vessel_segmentations/DUKE",
)
# Our extracted centerlines / skeletons (pipeline output).
CL_ROOT = _env_path(
    "DUKE_CL_ROOT",
    "/gpfs/data/karczmar-lab/workspaces/saritbose/centerlines_tc4d/studies/DUKE",
)
# MAMA-MIA source NIfTI images (shared team data; supplies the physical grid).
NII_ROOT = _env_path(
    "MAMA_MIA_IMAGES", "/gpfs/data/karczmar-lab/MAMA-MIA-syn60868042/images"
)
# Phase-A QC output (distance arrays, histograms, per-case HTML).
OUT = _env_path(
    "DUKE_QC_OUT", os.path.join(REPO_ROOT, "interactive_annotation_htmls", "phaseA_qc")
)

PROB_THRESH = 0.5  # binarize our probability segmentation
# tolerances (mm) we precompute coverage for, so tolerance choice is instant later
TOLS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]


def duke_to_ann_dir(case_id):
    """DUKE_002 -> the annotation folder Breast_MRI_002."""
    num = case_id.split("_")[1]
    return os.path.join(ANN_ROOT, f"Breast_MRI_{num}")


def load_reference(case_id):
    """Load the pre-contrast source NIfTI as the reference physical grid."""
    p = os.path.join(NII_ROOT, case_id, f"{case_id}_0000.nii.gz")
    return sitk.ReadImage(p)


def vessel_label_value(ann_img):
    """Find which integer label = 'Vessels' for THIS file. The Duke supplement does
    NOT use a fixed mapping: most cases are Vessels=1/Dense=2, but some (e.g. 686)
    are swapped. So we read the Slicer SegmentN_Name / SegmentN_LabelValue metadata
    and return the label value whose name contains 'vessel'. Falls back to 1.
    """
    name, lval = {}, {}
    for k in ann_img.GetMetaDataKeys():
        m = re.match(r"Segment(\d+)_Name$", k)
        if m:
            name[m.group(1)] = ann_img.GetMetaData(k)
        m = re.match(r"Segment(\d+)_LabelValue$", k)
        if m:
            lval[m.group(1)] = ann_img.GetMetaData(k)
    for i, nm in name.items():
        if "vessel" in nm.lower():
            return int(lval.get(i, 1))
    return 1


def load_vessel_gt(case_id, ref):
    """Read the annotation, pick the correct 'Vessels' label for this file, resample
    onto the reference grid (nearest-neighbor -- NEVER interpolate a label mask),
    return vessel ground truth as (z,y,x) bool plus the label value used.
    """
    d = duke_to_ann_dir(case_id)
    num = case_id.split("_")[1]
    f = os.path.join(d, f"Segmentation_Breast_MRI_{num}_Dense_and_Vessels.seg.nrrd")
    ann = sitk.ReadImage(f)
    vlabel = vessel_label_value(ann)  # read metadata BEFORE any Cast (Cast drops it)
    if ann.GetNumberOfComponentsPerPixel() > 1:
        ann = sitk.VectorIndexSelectionCast(ann, 0)
    ann = sitk.Cast(ann, sitk.sitkUInt8)
    ann_on_ref = sitk.Resample(
        ann, ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8
    )
    arr = sitk.GetArrayFromImage(ann_on_ref)  # (z,y,x)
    return (arr == vlabel), vlabel


def load_seg_max(case_id):
    """Max vessel probability across the 4 DCE timepoints. Returns raw (512,512,142)."""
    imdir = os.path.join(SEG_ROOT, case_id, "images")
    tps = sorted(f for f in os.listdir(imdir) if f.endswith("_vessel_segmentation.npz"))
    acc = None
    for f in tps:
        v = np.load(os.path.join(imdir, f))["vessel"].astype(np.float32)
        acc = v if acc is None else np.maximum(acc, v)
    return acc


def load_skeleton(case_id):
    p = os.path.join(CL_ROOT, case_id, f"{case_id}_skeleton_4d_exam_mask.npy")
    return np.load(p) > 0  # raw (512,512,142) bool


def candidate_transforms(target_shape):
    """Yield (name, func) that reorient a raw (512,512,142) array to the reference
    (z,y,x) shape. We try the two axis permutations that place the 142-axis first,
    each combined with flips over the 3 axes -- and pick whichever best matches the
    ground truth. This empirically resolves the 'yxz' layout instead of trusting it.
    """
    perms = [("p201", (2, 0, 1)), ("p210", (2, 1, 0))]
    for pname, perm in perms:
        for fx in (False, True):
            for fy in (False, True):
                for fz in (False, True):

                    def make(perm=perm, fx=fx, fy=fy, fz=fz):
                        def fn(a):
                            b = np.transpose(a, perm)
                            if fz:
                                b = b[::-1, :, :]
                            if fy:
                                b = b[:, ::-1, :]
                            if fx:
                                b = b[:, :, ::-1]
                            return b

                        return fn

                    name = f"{pname}{'X' if fx else ''}{'Y' if fy else ''}{'Z' if fz else ''}"
                    yield name, make()


def resolve_layout(vessel_gt, seg_max):
    """Find the transform whose thresholded segmentation best RECALLS the annotation.
    Correct orientation -> high recall; wrong -> ~0. Returns (best_name, best_fn, table).
    """
    seg_bin_raw = seg_max > PROB_THRESH
    gt_n = int(vessel_gt.sum())
    table = []
    best = (None, None, -1.0)
    for name, fn in candidate_transforms(vessel_gt.shape):
        try:
            s = fn(seg_bin_raw)
            if s.shape != vessel_gt.shape:
                continue
            recall = float((vessel_gt & s).sum()) / max(gt_n, 1)
        except Exception:
            continue
        table.append((name, recall))
        if recall > best[2]:
            best = (name, fn, recall)
    table.sort(key=lambda t: -t[1])
    return best[0], best[1], table


def affine_from_ref(ref):
    """Return (3x3 direction*spacing matrix, origin) mapping sitk index (x,y,z)->world."""
    D = np.array(ref.GetDirection(), float).reshape(3, 3)  # columns = axis dirs
    S = np.array(ref.GetSpacing(), float)
    O = np.array(ref.GetOrigin(), float)
    return D @ np.diag(S), O


def voxels_to_world(mask_zyx, M, O):
    """(z,y,x) bool mask -> (N,3) world coords. sitk index order is (x,y,z)=(a2,a1,a0)."""
    zyx = np.argwhere(mask_zyx)  # columns: z,y,x
    if zyx.size == 0:
        return np.empty((0, 3))
    ijk = zyx[:, [2, 1, 0]].astype(float)  # -> x,y,z
    return (M @ ijk.T).T + O


def process_case(case_id):
    for sub in ("csv", "hist", "html"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    ref = load_reference(case_id)
    vessel_gt, vlabel = load_vessel_gt(case_id, ref)
    seg_max = load_seg_max(case_id)
    skel_raw = load_skeleton(case_id)

    best_name, best_fn, table = resolve_layout(vessel_gt, seg_max)
    seg_bin = best_fn(seg_max > PROB_THRESH)
    skel = best_fn(skel_raw)

    M, O = affine_from_ref(ref)
    gt_w = voxels_to_world(vessel_gt, M, O)
    skel_w = voxels_to_world(skel, M, O)
    seg_recall = float((vessel_gt & seg_bin).sum()) / max(int(vessel_gt.sum()), 1)

    # nearest centerline voxel for every annotated vessel voxel (physical mm)
    if len(skel_w) and len(gt_w):
        tree = cKDTree(skel_w)
        nn_dist, _ = tree.query(gt_w, k=1)
    else:
        nn_dist = np.array([])

    pct = {
        p: (float(np.percentile(nn_dist, p)) if nn_dist.size else None)
        for p in (50, 75, 90, 95)
    }
    cov = {t: (float((nn_dist <= t).mean()) if nn_dist.size else None) for t in TOLS}

    # save raw distances so tolerance choice later needs no recompute
    np.save(os.path.join(OUT, "csv", f"{case_id}_nn_dist_mm.npy"), nn_dist)

    _make_histogram(case_id, nn_dist)
    _make_html(case_id, gt_w, skel_w, best_name)

    return {
        "case_id": case_id,
        "vessel_label_used": vlabel,
        "n_vessel_gt_voxels": int(vessel_gt.sum()),
        "n_skeleton_voxels": int(skel.sum()),
        "layout": best_name,
        "seg_recall_at_%.2f" % PROB_THRESH: round(seg_recall, 4),
        "nn_p50_mm": pct[50],
        "nn_p75_mm": pct[75],
        "nn_p90_mm": pct[90],
        "nn_p95_mm": pct[95],
        **{
            f"cl_cov_within_{t}mm": (round(cov[t], 4) if cov[t] is not None else None)
            for t in TOLS
        },
        "layout_recall_table": table[:4],
    }


def _make_histogram(case_id, nn_dist):
    fig, ax = plt.subplots(figsize=(7, 4))
    if nn_dist.size:
        ax.hist(
            nn_dist,
            bins=60,
            range=(0, min(20, nn_dist.max() + 1)),
            color="#4C78A8",
            edgecolor="white",
        )
        for t in (1, 2, 3):
            ax.axvline(t, color="crimson", ls="--", lw=1)
            ax.text(t, ax.get_ylim()[1] * 0.9, f"{t}mm", color="crimson", fontsize=8)
        med = np.median(nn_dist)
        ax.set_title(
            f"{case_id}: annotated-vessel -> nearest centerline distance\n"
            f"median={med:.2f}mm  n={nn_dist.size}"
        )
    else:
        ax.set_title(f"{case_id}: no distances (empty mask)")
    ax.set_xlabel("distance to nearest centerline voxel (mm)")
    ax.set_ylabel("# annotated vessel voxels")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "hist", f"{case_id}_nn_hist.png"), dpi=110)
    plt.close(fig)


def _make_html(case_id, gt_w, skel_w, layout):
    # subsample the (potentially large) annotation cloud to keep the html light
    def sub(a, n=25000):
        if len(a) > n:
            idx = np.random.default_rng(0).choice(len(a), n, replace=False)
            return a[idx]
        return a

    g = sub(gt_w)
    s = skel_w
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=g[:, 0],
            y=g[:, 1],
            z=g[:, 2],
            mode="markers",
            marker=dict(size=1.6, color="red", opacity=0.55),
            name=f"annotated vessels (n={len(gt_w)})",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=s[:, 0],
            y=s[:, 1],
            z=s[:, 2],
            mode="markers",
            marker=dict(size=2.2, color="blue", opacity=0.9),
            name=f"our centerline (n={len(skel_w)})",
        )
    )
    fig.update_layout(
        title=f"{case_id}  (layout {layout}) — red=radiologist vessels, blue=our centerline",
        scene=dict(
            aspectmode="data",
            xaxis_title="x mm",
            yaxis_title="y mm",
            zaxis_title="z mm",
        ),
        legend=dict(itemsizing="constant"),
    )
    out = os.path.join(OUT, "html", f"{case_id}_vessel_vs_centerline.html")
    fig.write_html(out, include_plotlyjs=True, full_html=True)  # self-contained


def main():
    cases = sys.argv[1:]
    if not cases:
        print("usage: duke_qc_viz.py DUKE_002 DUKE_141 ...")
        sys.exit(1)
    rows = []
    for c in cases:
        print(f"[+] {c} ...", flush=True)
        try:
            r = process_case(c)
            rows.append(r)
            print(
                f"    layout={r['layout']}  seg_recall={r['seg_recall_at_0.50']}  "
                f"nn median={r['nn_p50_mm']}mm  cov@2mm={r['cl_cov_within_2.0mm']}  "
                f"top-layouts={r['layout_recall_table']}",
                flush=True,
            )
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"    FAILED {c}: {e}", flush=True)
    with open(os.path.join(OUT, "csv", "phaseA_summary.json"), "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"[done] wrote {len(rows)} cases -> {OUT}/csv/phaseA_summary.json")


if __name__ == "__main__":
    main()
