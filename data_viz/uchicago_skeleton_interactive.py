#!/usr/bin/env python3
"""UChicago vessel QC -- interactive 3D HTML, one file per case.

Counterpart to ``interactive_annotation_htmls/duke_qc_interactive.py``, but
UChicago has NO radiologist annotations, so the annotation layers (green/orange/
red distance bands) are replaced by the 3D SKELETON (centerline) of the vessel
segmentation.

Layers (toggle via legend click; quick-view buttons combine them):
  * centerline / skeleton, REAL breast mask run   (cyan)   -- the main object
  * vessel segmentation,   REAL breast mask run   (blue)   -- filled mask, subsampled
  * centerline / skeleton, MODEL breast mask run  (orange) -- baseline, for comparison
  * whole-breast mask (real, BreastDivider)       (gray)   -- faint anatomical context

The baseline-skeleton layer is included deliberately: after the external-mask
run, the open question was "why do the vessel MIPs look sparse?", and seeing the
two centerline trees in the same 3D scene shows directly where the real breast
mask recovers vessels the model mask missed.

Skeletonization reuses the pipeline's own routine (``graph_extraction.skeleton3d.
skeletonize3d`` on an EDT priority, exactly as ``graph_extraction/tc4d.py``'s
``_skeletonize_support_mask`` does), so these centerlines are computed the same
way the real graph stage would compute them -- NOT a different ad-hoc thinning.
(The graph stage itself can't run on UChicago yet: its discovery seam,
``graph_extraction/core4d.py:111``, rejects the pipeline's own UChicago output
filenames -- a separate pre-existing bug.)

Vessel probability is combined across ALL phases (voxelwise max over time) before
thresholding, so the scene shows the full vessel tree rather than one timepoint.
UChicago is 1 mm isotropic, so voxel indices == mm.

Self-contained HTML (plotly.js embedded); renders in the browser, no server.

Usage (from the repo root, `micromamba activate vanguard`):

    # every case that has external-mask segmentation output
    python data_viz/uchicago_skeleton_interactive.py

    # only one sub-source, or only specific exams
    python data_viz/uchicago_skeleton_interactive.py --sub-source simbiosys uch_nac
    python data_viz/uchicago_skeleton_interactive.py --case-ids simbiosys_1_3_118...

Input dirs come from the segmentation runs and are overridable by flag, so this
is re-runnable against any pair of (external, baseline) output roots.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from scipy import ndimage  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from graph_extraction.skeleton3d import skeletonize3d  # noqa: E402

DEFAULT_EXTERNAL_DIR = Path(
    "/ess/scratch/scratch1/t-9sbose/uchicago_vessel_segmentations_smoke_external_breast"
)
DEFAULT_BASELINE_DIR = Path(
    "/ess/scratch/scratch1/t-9sbose/uchicago_vessel_segmentations_smoke_v2"
)
DEFAULT_BREAST_DIR = Path(
    "/ess/scratch/scratch1/t-9sbose/uchicago_breast_masks_external"
)
DEFAULT_OUT_DIR = Path(
    "/gpfs/data/karczmar-lab/workspaces/saritbose/outputs/interactive_annotation_htmls"
)
DEFAULT_MANIFEST = Path(
    "/gpfs/data/karczmar-lab/vanguard/dce2d_internal_ultrafast_manifest/"
    "dce2d_internal_ultrafast_manifest_with_whole_breast_masks.csv"
)

VESSEL_PROB_THRESH = 0.5
BREAST_PROB_THRESH = 0.5

# Keep the browser scene light: cap points per layer (stats use the FULL data).
CAP_SKEL = 60000
CAP_SEG = 60000
CAP_BREAST = 25000

C_SKEL_EXT = "#00d0d0"  # cyan  : centerline, real-breast-mask run
C_SEG_EXT = "#5aa0e6"  # blue  : segmentation, real-breast-mask run
C_SKEL_BASE = "#f39c12"  # orange: centerline, model-breast-mask baseline
C_BREAST = "#9aa4ad"  # gray  : whole-breast context


def _sub(a: np.ndarray, cap: int, seed: int = 0) -> np.ndarray:
    """Randomly subsample rows of an (N,3) array down to ``cap`` (display only)."""
    if len(a) <= cap:
        return a
    idx = np.random.default_rng(seed).choice(len(a), cap, replace=False)
    return a[idx]


def combined_vessel_prob(
    seg_dir: Path, case_id: str, sub_source: str
) -> tuple[np.ndarray | None, int]:
    """Voxelwise max vessel probability across every phase (the full vessel tree).

    Returns ``(combined, n_phases)``; ``(None, 0)`` when the case has no output.
    The phase count is returned rather than assumed: UChicago exams are 13 OR 24
    phases depending on the exam, and an earlier version of this script hardcoded
    "24 phases" into the subtitle, which mislabels every 13-phase exam.
    """
    mask_dir = seg_dir / sub_source / case_id / "images"
    paths = sorted(mask_dir.glob(f"{case_id}_phase_*_vessel_segmentation.npz"))
    if not paths:
        return None, 0
    combined: np.ndarray | None = None
    for p in paths:
        v = np.load(p)["vessel"].astype(np.float32)
        combined = v if combined is None else np.maximum(combined, v)
    return combined, len(paths)


def skeletonize_mask(binary: np.ndarray) -> np.ndarray:
    """Pipeline-identical centerline: EDT priority -> topology-preserving thinning."""
    if not np.any(binary):
        return np.zeros_like(binary, dtype=bool)
    priority = ndimage.distance_transform_edt(binary).astype(np.float32, copy=False)
    return skeletonize3d(priority, threshold=0.0) > 0


def load_breast_mask(breast_dir: Path, case_id: str) -> np.ndarray | None:
    """Load the persisted external whole-breast mask for a case (any phase; all equal).

    The manifest ships ONE mask per exam (from phase 0), fanned out across phases
    by the segmentation run, so the first phase file found is representative.
    """
    for path in sorted(breast_dir.glob(f"{case_id}_phase_*.npy")):
        return np.load(path) > BREAST_PROB_THRESH
    return None


def build(
    case_id: str,
    sub_source: str,
    external_dir: Path,
    baseline_dir: Path,
    breast_dir: Path,
    out_dir: Path,
) -> None:
    """Render one case's interactive 3D scene to ``<out_dir>/<case>_..._interactive.html``."""
    ext_prob, n_phases = combined_vessel_prob(external_dir, case_id, sub_source)
    if ext_prob is None:
        print(f"  [skip] {case_id}: no external vessel output")
        return
    ext_seg = ext_prob > VESSEL_PROB_THRESH
    ext_skel = skeletonize_mask(ext_seg)

    base_prob, _ = combined_vessel_prob(baseline_dir, case_id, sub_source)
    base_skel = None
    if base_prob is not None and base_prob.shape == ext_prob.shape:
        base_skel = skeletonize_mask(base_prob > VESSEL_PROB_THRESH)

    breast = load_breast_mask(breast_dir, case_id)

    ext_seg_pts = np.argwhere(ext_seg).astype(np.float32)
    ext_skel_pts = np.argwhere(ext_skel).astype(np.float32)
    base_skel_pts = (
        np.argwhere(base_skel).astype(np.float32)
        if base_skel is not None
        else np.empty((0, 3))
    )
    breast_pts = (
        np.argwhere(breast).astype(np.float32)
        if breast is not None
        else np.empty((0, 3))
    )

    def scatter(pts, color, name, size, opacity, cap, visible=True):  # noqa: ANN001, ANN202
        p = _sub(pts, cap)
        return go.Scatter3d(
            x=p[:, 0] if len(p) else [],
            y=p[:, 1] if len(p) else [],
            z=p[:, 2] if len(p) else [],
            mode="markers",
            marker={"size": size, "color": color, "opacity": opacity},
            name=name,
            visible=visible,
        )

    t_skel_ext = scatter(
        ext_skel_pts,
        C_SKEL_EXT,
        f"centerline — REAL breast mask (n={len(ext_skel_pts)})",
        2,
        0.9,
        CAP_SKEL,
    )
    t_seg_ext = scatter(
        ext_seg_pts,
        C_SEG_EXT,
        f"segmentation — REAL breast mask (n={len(ext_seg_pts)}, subsampled)",
        1.5,
        0.25,
        CAP_SEG,
        visible="legendonly",
    )
    t_skel_base = scatter(
        base_skel_pts,
        C_SKEL_BASE,
        f"centerline — MODEL breast mask baseline (n={len(base_skel_pts)})",
        2,
        0.8,
        CAP_SKEL,
        visible="legendonly",
    )
    t_breast = scatter(
        breast_pts,
        C_BREAST,
        f"whole-breast mask (real, context; n={len(breast_pts)}, subsampled)",
        1.5,
        0.06,
        CAP_BREAST,
        visible="legendonly",
    )

    traces = [t_skel_ext, t_seg_ext, t_skel_base, t_breast]
    #          0           1          2            3
    fig = go.Figure(data=traces)

    def vis(show: set[int]) -> list[bool | str]:
        return [(True if i in show else "legendonly") for i in range(4)]

    buttons = [
        {
            "label": "Centerline only (real mask)",
            "method": "update",
            "args": [{"visible": vis({0})}],
        },
        {
            "label": "Centerline + segmentation",
            "method": "update",
            "args": [{"visible": vis({0, 1})}],
        },
        {
            "label": "REAL vs MODEL centerline",
            "method": "update",
            "args": [{"visible": vis({0, 2})}],
        },
        {
            "label": "Centerline + breast context",
            "method": "update",
            "args": [{"visible": vis({0, 3})}],
        },
        {
            "label": "All layers",
            "method": "update",
            "args": [{"visible": [True, True, True, True]}],
        },
    ]

    n_ext, n_base = len(ext_skel_pts), len(base_skel_pts)
    gain = (
        f"{n_ext / n_base:.1f}x more centerline voxels than the model-mask baseline"
        if n_base
        else "no baseline centerline"
    )
    subtitle = (
        f"vessel coverage={ext_seg.mean() * 100:.4f}% of volume · "
        f"centerline voxels: real-mask={n_ext} vs model-mask={n_base} ({gain}) · "
        f"all {n_phases} phases combined (max over time), "
        f"thr={VESSEL_PROB_THRESH} · 1 mm isotropic"
    )

    fig.update_layout(
        title={
            "text": (
                f"<b>{case_id}</b> ({sub_source}) — 3D vessel centerline, "
                f"real vs model breast mask"
                f"<br><span style='font-size:12px'>{subtitle}</span>"
            ),
            "x": 0.02,
        },
        scene={
            "aspectmode": "data",
            "xaxis_title": "x (mm)",
            "yaxis_title": "y (mm)",
            "zaxis_title": "z (mm)",
            "bgcolor": "#0e1117",
        },
        paper_bgcolor="#0e1117",
        font={"color": "#e6edf3"},
        legend={"itemsizing": "constant", "x": 0.0, "y": -0.02, "orientation": "h"},
        updatemenus=[
            {
                "type": "buttons",
                "direction": "down",
                "x": 1.02,
                "y": 1.0,
                "xanchor": "left",
                "showactive": True,
                "buttons": buttons,
            }
        ],
        margin={"l": 0, "r": 0, "t": 80, "b": 0},
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{case_id}_uchicago_skeleton_interactive.html"
    fig.write_html(out, include_plotlyjs=True, full_html=True)
    print(f"  [ok] {case_id}: skeleton={n_ext} (baseline={n_base}) -> {out}")


def select_cases(
    manifest: Path,
    external_dir: Path,
    sub_sources: list[str] | None,
    case_ids: list[str] | None,
) -> list[tuple[str, str]]:
    """Pick (case_id, sub_source) pairs to render, in manifest order.

    A case is renderable only if the external-mask run actually produced output
    for it, so this never silently emits an empty scene for an exam that was
    never segmented. Explicit ``case_ids`` are fail-closed: one that has no
    output raises, rather than quietly rendering fewer HTMLs than asked for.
    """
    with manifest.open(newline="") as f:
        rows = list(csv.DictReader(f))

    has_output = {
        row["exam_id"]: row["dataset"]
        for row in rows
        if (external_dir / row["dataset"] / row["exam_id"]).is_dir()
    }

    if case_ids:
        missing = [c for c in case_ids if c not in has_output]
        if missing:
            raise SystemExit(
                f"No segmentation output under {external_dir} for {len(missing)} "
                f"requested case(s): {missing}"
            )
        wanted = set(case_ids)
    elif sub_sources:
        wanted = {c for c, sub in has_output.items() if sub in set(sub_sources)}
        if not wanted:
            raise SystemExit(
                f"No segmented cases under {external_dir} for sub-source(s) {sub_sources}"
            )
    else:
        wanted = set(has_output)

    return [
        (row["exam_id"], row["dataset"]) for row in rows if row["exam_id"] in wanted
    ]


def main() -> None:
    """Run the interactive-HTML CLI."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--case-ids",
        nargs="+",
        default=None,
        help="Render exactly these exam ids (default: every case with output).",
    )
    p.add_argument(
        "--sub-source",
        nargs="+",
        default=None,
        dest="sub_sources",
        help="Render only these manifest sub-sources: simbiosys|uch_nac|her2_naclike.",
    )
    p.add_argument("--external-dir", type=Path, default=DEFAULT_EXTERNAL_DIR)
    p.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    p.add_argument("--breast-dir", type=Path, default=DEFAULT_BREAST_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = p.parse_args()

    cases = select_cases(
        args.manifest, args.external_dir, args.sub_sources, args.case_ids
    )
    print(
        f"Building interactive 3D skeleton HTML for {len(cases)} case(s) -> {args.out_dir}"
    )

    failed = []
    for cid, sub in cases:
        print(f"[+] {cid} ({sub})", flush=True)
        try:
            build(
                cid,
                sub,
                args.external_dir,
                args.baseline_dir,
                args.breast_dir,
                args.out_dir,
            )
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            print(f"  [FAILED] {cid}: {e}", flush=True)
            failed.append(cid)

    print(
        f"\n[done] {len(cases) - len(failed)}/{len(cases)} rendered -> {args.out_dir}"
    )
    if failed:
        raise SystemExit(f"{len(failed)} case(s) failed: {failed}")


if __name__ == "__main__":
    main()
