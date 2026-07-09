#!/usr/bin/env python3
"""Duke vessel-annotation QC -- interactive 3D HTML for ONE case.

Purpose (mentor request): a browser-viewable 3D scene where you can TOGGLE our
segmentation, our skeleton/centerline, and the radiologist annotation on/off, so you
can answer: "are the red misses (annotated vessel >5 mm from our centerline) in places
where we have NO segmentation (an upstream model miss -> would need an image-based
branch) or where segmentation exists but the skeleton failed (a skeletonization gap)?"

Layers (each toggles via a legend click; quick-view buttons combine them):
  * annotation <=2 mm    (green)   -- well recovered
  * annotation 2-5 mm    (orange)  -- marginal (the extra a 5 mm tolerance would credit)
  * annotation >5 mm     (red)     -- the misses we care about
  * our skeleton         (gray)    -- our centerline
  * our segmentation     (blue)    -- filled vessel mask (subsampled for the browser)

Also prints/labels the diagnostic split of the red misses:
  of the >5 mm misses, what fraction have OUR segmentation within 2 mm (skeleton gap)
  vs none (segmentation/model miss).

Reuses the validated Phase A helpers (duke_qc_viz.py) so alignment == the phaseB numbers.
Self-contained HTML (plotly.js embedded) -- renders in your browser via WebGL, no server.
"""

import os
import sys

os.environ["MPLBACKEND"] = "Agg"
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from duke_qc_viz import (
    PROB_THRESH,
    affine_from_ref,
    load_reference,
    load_seg_max,
    load_skeleton,
    load_vessel_gt,
    resolve_layout,
    voxels_to_world,
)

# Where the interactive HTMLs are written. Defaults to this directory so a fresh
# clone regenerates the committed HTMLs in place; override with DUKE_QC_INTERACTIVE_OUT.
OUT_DIR = os.path.expandvars(
    os.path.expanduser(
        os.environ.get(
            "DUKE_QC_INTERACTIVE_OUT", os.path.dirname(os.path.abspath(__file__))
        )
    )
)

NEAR = 2.0  # <= NEAR mm  -> green
FAR = 5.0  # NEAR..FAR   -> orange ; > FAR -> red (miss)
SEG_NEAR_MISS = 2.0  # a miss counts as "has segmentation" if seg is within this (mm)

# keep the browser scene light: cap points per layer (stats use the FULL data)
CAP_ANNOT = 40000
CAP_SKEL = 40000
CAP_SEG = 60000

C_GREEN = "#26b33f"
C_ORANGE = "#f39c12"
C_RED = "#e6191f"
C_SKEL = "#666666"
C_SEG = "#5aa0e6"


def _sub(a, cap, seed=0):
    """Randomly subsample rows of an (N,3) array down to `cap` (for display only)."""
    if len(a) <= cap:
        return a
    idx = np.random.default_rng(seed).choice(len(a), cap, replace=False)
    return a[idx]


def build(case_id):
    ref = load_reference(case_id)
    vessel_gt, _ = load_vessel_gt(case_id, ref)
    seg_max = load_seg_max(case_id)
    skel_raw = load_skeleton(case_id)

    _, best_fn, _ = resolve_layout(vessel_gt, seg_max)
    seg_bin = best_fn(seg_max > PROB_THRESH)
    skel = best_fn(skel_raw)

    M, O = affine_from_ref(ref)
    gt_w = voxels_to_world(vessel_gt, M, O)  # order matches nn_dist below
    skel_w = voxels_to_world(skel, M, O)
    seg_w = voxels_to_world(seg_bin, M, O)

    # distance from each annotated voxel to nearest centerline (mm)
    nn_dist, _ = cKDTree(skel_w).query(gt_w, k=1)
    near = nn_dist <= NEAR
    mid = (nn_dist > NEAR) & (nn_dist <= FAR)
    far = nn_dist > FAR

    # diagnostic: of the >5 mm misses, how many have OUR seg within 2 mm?
    miss_w = gt_w[far]
    if len(miss_w) and len(seg_w):
        d_seg, _ = cKDTree(seg_w).query(miss_w, k=1)
        has_seg = d_seg <= SEG_NEAR_MISS
        frac_skel_gap = float(has_seg.mean())  # seg present -> skeleton failed
    else:
        has_seg = np.zeros(len(miss_w), bool)
        frac_skel_gap = 0.0
    frac_model_miss = 1.0 - frac_skel_gap

    cov2 = float(near.mean())
    cov5 = float((near | mid).mean())
    missed5 = float(far.mean())

    # ---- traces ----
    def scatter(pts, color, name, size, visible=True):
        pts = _sub(pts, CAP_ANNOT if size >= 2 else CAP_SEG)
        return go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="markers",
            marker=dict(size=size, color=color, opacity=0.85),
            name=name,
            visible=visible,
        )

    # split the red misses into "on segmentation" vs "no segmentation" so the
    # diagnostic is visible directly, not just in the summary number
    miss_on_seg = miss_w[has_seg] if len(miss_w) else miss_w
    miss_no_seg = miss_w[~has_seg] if len(miss_w) else miss_w

    t_green = scatter(gt_w[near], C_GREEN, f"annotation ≤2 mm  ({cov2:.0%})", 2)
    t_orange = scatter(gt_w[mid], C_ORANGE, f"annotation 2–5 mm  ({cov5-cov2:.0%})", 2)
    t_red_on = go.Scatter3d(
        x=miss_on_seg[:, 0] if len(miss_on_seg) else [],
        y=miss_on_seg[:, 1] if len(miss_on_seg) else [],
        z=miss_on_seg[:, 2] if len(miss_on_seg) else [],
        mode="markers",
        marker=dict(size=3, color=C_RED, opacity=0.95, symbol="circle"),
        name=f"MISS >5 mm — on our seg (skeleton gap)  ({frac_skel_gap:.0%} of misses)",
    )
    t_red_no = go.Scatter3d(
        x=miss_no_seg[:, 0] if len(miss_no_seg) else [],
        y=miss_no_seg[:, 1] if len(miss_no_seg) else [],
        z=miss_no_seg[:, 2] if len(miss_no_seg) else [],
        mode="markers",
        marker=dict(size=3.4, color="#7d0009", opacity=0.98, symbol="diamond"),
        name=f"MISS >5 mm — NO seg (model miss)  ({frac_model_miss:.0%} of misses)",
    )
    t_skel = go.Scatter3d(
        x=_sub(skel_w, CAP_SKEL)[:, 0],
        y=_sub(skel_w, CAP_SKEL)[:, 1],
        z=_sub(skel_w, CAP_SKEL)[:, 2],
        mode="markers",
        marker=dict(size=2, color=C_SKEL, opacity=0.55),
        name=f"our centerline (n={len(skel_w)})",
    )
    t_seg = go.Scatter3d(
        x=_sub(seg_w, CAP_SEG)[:, 0],
        y=_sub(seg_w, CAP_SEG)[:, 1],
        z=_sub(seg_w, CAP_SEG)[:, 2],
        mode="markers",
        marker=dict(size=1.5, color=C_SEG, opacity=0.25),
        name=f"our segmentation (n={len(seg_w)}, subsampled)",
        visible="legendonly",
    )

    # trace order fixes the indices used by the quick-view buttons below
    traces = [t_green, t_orange, t_red_on, t_red_no, t_skel, t_seg]
    #          0        1         2         3         4       5
    fig = go.Figure(data=traces)

    def vis(show):  # helper: build a visibility list for the 6 traces
        return [(True if i in show else "legendonly") for i in range(6)]

    buttons = [
        dict(
            label="All layers",
            method="update",
            args=[{"visible": [True, True, True, True, True, "legendonly"]}],
        ),
        dict(
            label="Misses + segmentation (the diagnostic)",
            method="update",
            args=[{"visible": vis({2, 3, 5})}],
        ),
        dict(
            label="With red (annotation only)",
            method="update",
            args=[{"visible": vis({0, 1, 2, 3})}],
        ),
        dict(
            label="Without red (annotation only)",
            method="update",
            args=[{"visible": vis({0, 1})}],
        ),
        dict(
            label="Our pipeline only (seg + centerline)",
            method="update",
            args=[{"visible": vis({4, 5})}],
        ),
    ]

    subtitle = (
        f"coverage@2mm={cov2:.2f} · @5mm={cov5:.2f} · misses>5mm={missed5:.0%}"
        f"   |   of the misses: {frac_skel_gap:.0%} have segmentation nearby "
        f"(skeleton gap), {frac_model_miss:.0%} do NOT (model miss)"
    )
    fig.update_layout(
        title=dict(
            text=f"<b>{case_id}</b> — annotation vs pipeline<br>"
            f"<span style='font-size:13px'>{subtitle}</span>",
            x=0.02,
        ),
        scene=dict(
            aspectmode="data",
            xaxis_title="x (mm)",
            yaxis_title="y (mm)",
            zaxis_title="z (mm)",
        ),
        legend=dict(itemsizing="constant", x=0.0, y=-0.02, orientation="h"),
        updatemenus=[
            dict(
                type="buttons",
                direction="down",
                x=1.02,
                y=1.0,
                xanchor="left",
                showactive=True,
                buttons=buttons,
            )
        ],
        margin=dict(l=0, r=0, t=70, b=0),
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{case_id}_interactive.html")
    fig.write_html(out, include_plotlyjs=True, full_html=True)
    print(f"[done] {case_id}: cov@2={cov2:.3f} cov@5={cov5:.3f} misses={missed5:.3f}")
    print(
        f"       of {len(miss_w)} miss voxels: {frac_skel_gap:.1%} on-seg (skeleton gap), "
        f"{frac_model_miss:.1%} no-seg (model miss)"
    )
    print(f"       wrote {out}")


ALL_CASES = [
    "DUKE_" + c
    for c in (
        "002 021 041 105 141 148 287 290 338 363 383 435 464 489 501 525 530 "
        "562 595 612 616 618 636 640 652 670 681 686 688 693 694 723 762 797 "
        "805 832"
    ).split()
]


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "DUKE_693"
    cases = ALL_CASES if arg.upper() == "ALL" else [arg]
    for c in cases:
        print(f"[+] {c}", flush=True)
        try:
            build(c)
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"    FAILED {c}: {e}", flush=True)
    print(f"[batch done] {len(cases)} case(s) -> {OUT_DIR}", flush=True)
