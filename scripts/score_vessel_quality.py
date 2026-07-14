#!/usr/bin/env python3
"""Score UChicago vessel segmentation WITHOUT vessel ground truth, using held-out physics.

UChicago ships no vessel annotations, so there is no Dice-against-truth to optimize.
The two numbers that look like quality and are not:

* **Run-to-run agreement** (our model-mask run vs our external-mask run). Maximized
  by making the two runs identical. Measures agreement, not correctness.
* **Vessel voxel count.** Maximized by lowering the probability threshold. Every
  extra voxel can be noise.

This scores against something the model *cannot* have fitted to. **The vessel U-Net
sees one 3D phase at a time -- it never sees the time axis.** So contrast kinetics is
held-out information, and the physics is unambiguous: voxels carrying contrast agent
enhance sharply after injection; fat, fibroglandular tissue, noise and motion
artifact do not. A segmentation that is finding real vessels will select voxels whose
enhancement curves are steep; one that is finding noise will not.

Metrics, per exam (all computed inside the REAL BreastDivider whole-breast mask, so
nothing is rewarded for firing outside the breast):

  enh_auroc      AUROC of peak relative enhancement at separating predicted-vessel
                 voxels from other in-breast voxels. THE headline number: it rises
                 only if the mask picks out genuinely enhancing structures.
  enh_median_*   median peak enhancement inside vs outside the predicted vessels.
  vessel_voxels  size of the mask. Necessary but NOT sufficient -- reported so a
                 "win" driven purely by a lower threshold is visible as such.
  largest_cc_frac / n_components
                 topology. Vessels are connected tubular trees; a mask of thousands
                 of isolated specks is worse than a few large branching components at
                 the same voxel count. This is what stops "just lower the threshold"
                 from scoring well: noise adds components, not connectivity.

Enhancement is computed from the RAW-SIGNAL motion-corrected frames with the
documented precontrast baseline: E = (max_t S(t) - S0) / S0, S0 = mean of the first
`baseline_frame_count` frames. Requires the raw-signal root built by
`scripts/build_uchicago_spgr_root.py` (the legacy images are z-scored, so a ratio to
baseline is meaningless on them).

Run:
    python scripts/score_vessel_quality.py --run <seg_dir> --raw-root <spgr_mc_root> \
        --label v4-mc-external --out-csv results/vessel_quality.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import SimpleITK as sitk

MASK_MANIFEST = (
    "/gpfs/data/karczmar-lab/vanguard/dce2d_internal_ultrafast_manifest/"
    "dce2d_internal_ultrafast_manifest_with_whole_breast_masks.csv"
)
PHASE_RE = re.compile(r"_phase_(\d+)_vessel_segmentation\.npz$")
EPS = 1e-6


def find_cases(run: Path) -> dict[str, Path]:
    """case_id -> images dir, for every case with vessel output."""
    return {
        d.parent.name: d
        for d in run.glob("*/*/images")
        if any(d.glob("*_vessel_segmentation.npz"))
    }


def vessel_time_max(images_dir: Path) -> np.ndarray:
    """Per-voxel max vessel probability over all phases, in model (rows,cols,slices) space."""
    acc = None
    for f in sorted(images_dir.glob("*_vessel_segmentation.npz")):
        with np.load(f) as d:
            v = d["vessel"].astype(np.float32)
        acc = v if acc is None else np.maximum(acc, v)
    if acc is None:
        raise SystemExit(f"no vessel output under {images_dir}")
    return acc


def peak_enhancement(
    raw_root: Path, exam_id: str, dataset: str, n_baseline: int
) -> np.ndarray:
    """(max_t S - S0)/S0 from raw-signal frames, in SimpleITK (z,y,x) order.

    S0 is the mean of the first ``n_baseline`` precontrast frames. Raw signal is
    nonnegative, so the ratio is meaningful (it is NOT on the z-scored legacy images).
    """
    files = sorted((raw_root / "images" / dataset / exam_id).glob("phase_*.nii.gz"))
    if len(files) <= n_baseline:
        raise SystemExit(f"{exam_id}: {len(files)} frames <= baseline {n_baseline}")
    base = np.mean(
        [
            sitk.GetArrayFromImage(sitk.ReadImage(str(f))).astype(np.float32)
            for f in files[:n_baseline]
        ],
        axis=0,
    )
    peak = None
    for f in files[n_baseline:]:
        s = sitk.GetArrayFromImage(sitk.ReadImage(str(f))).astype(np.float32)
        peak = s if peak is None else np.maximum(peak, s)
    return (peak - base) / (base + EPS)


def load_breast_mask(exam_id: str, adapter) -> np.ndarray:  # noqa: ANN001
    """Real BreastDivider whole-breast mask, carried into model space by the adapter."""
    import external_breast_masks as ebm

    manifest = ebm.load_manifest(MASK_MANIFEST)
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        return np.asarray(
            ebm.build_whole_breast_mask(exam_id, manifest, Path(td), adapter),
            dtype=bool,
        )


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC via rank statistic (no sklearn dependency)."""
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    # average ranks for ties
    s_sorted = scores[order]
    i = 0
    while i < s_sorted.size:
        j = i
        while j + 1 < s_sorted.size and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def topology(mask: np.ndarray) -> tuple[int, float]:
    """(n_connected_components, fraction of vessel volume in the largest component)."""
    from scipy import ndimage

    lab, n = ndimage.label(mask, structure=np.ones((3, 3, 3)))
    if n == 0:
        return 0, 0.0
    sizes = np.bincount(lab.ravel())[1:]
    return int(n), float(sizes.max() / sizes.sum())


def main() -> None:
    """Score one segmentation run against held-out contrast kinetics + topology."""
    import sys

    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "segmentation"))
    sys.path.insert(0, str(root))
    from cohorts.uchicago import UChicagoDataset

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, help="vessel-segmentation output dir")
    p.add_argument(
        "--raw-root",
        required=True,
        help="raw-signal root (build_uchicago_spgr_root.py)",
    )
    p.add_argument("--label", required=True)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument(
        "--sample", type=int, default=400_000, help="voxels sampled for AUROC"
    )
    p.add_argument("--out-csv", required=True)
    args = p.parse_args()

    raw_root = Path(args.raw_root)
    adapter = UChicagoDataset(root=raw_root)
    manifest = {c: adapter.case_dataset_name(c) for c in adapter.discover_cases()}
    import pandas as pd

    mdf = pd.read_csv(raw_root / "dce2d_internal_ultrafast_manifest.csv").set_index(
        "exam_id"
    )

    cases = find_cases(Path(args.run))
    rng = np.random.default_rng(0)
    rows = []
    for exam_id, images_dir in sorted(cases.items()):
        dataset = manifest[exam_id]
        n_base = int(mdf.loc[exam_id, "baseline_frame_count"])

        prob = vessel_time_max(images_dir)
        pred = prob > args.threshold
        breast = load_breast_mask(exam_id, adapter)

        enh_zyx = peak_enhancement(raw_root, exam_id, dataset, n_base)
        enh = adapter.preprocess(enh_zyx)  # into model (rows, cols, slices) space
        if enh.shape != pred.shape:
            raise SystemExit(f"{exam_id}: enhancement {enh.shape} != pred {pred.shape}")

        inb = breast & np.isfinite(enh)
        pos = pred & inb
        neg = (~pred) & inb
        n_pos = int(pos.sum())
        if n_pos == 0:
            print(f"  {exam_id[:26]:28s} NO vessel voxels in breast -- skipping")
            continue

        # subsample negatives (there are millions) for a tractable AUROC
        pos_v = enh[pos]
        neg_idx = np.flatnonzero(neg.ravel())
        take = min(args.sample, neg_idx.size)
        neg_v = enh.ravel()[rng.choice(neg_idx, size=take, replace=False)]
        scores = np.concatenate([pos_v, neg_v])
        labels = np.concatenate([np.ones(pos_v.size, bool), np.zeros(neg_v.size, bool)])

        a = auroc(scores, labels)
        ncc, lcc = topology(pred)
        row = {
            "label": args.label,
            "case_id": exam_id,
            "sub_source": dataset,
            "threshold": args.threshold,
            "vessel_voxels": int(pred.sum()),
            "vessel_voxels_in_breast": n_pos,
            "enh_auroc": round(a, 4),
            "enh_median_vessel": round(float(np.median(pos_v)), 4),
            "enh_median_background": round(float(np.median(neg_v)), 4),
            "n_components": ncc,
            "largest_cc_frac": round(lcc, 4),
        }
        rows.append(row)
        print(
            f"  {exam_id[:26]:28s} vox={row['vessel_voxels']:>8,} "
            f"enh_auroc={a:.3f} enh(ves/bg)={row['enh_median_vessel']:.2f}/"
            f"{row['enh_median_background']:.2f} ncc={ncc:>6,} lcc={lcc:.3f}"
        )

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out.exists()
    with out.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        if write_header:
            w.writeheader()
        w.writerows(rows)

    print(f"\n=== {args.label} (threshold {args.threshold}) ===")
    print(
        f"  mean enh_auroc      {np.mean([r['enh_auroc'] for r in rows]):.3f}   (higher = mask selects genuinely enhancing voxels)"
    )
    print(f"  total vessel voxels {sum(r['vessel_voxels'] for r in rows):,}")
    print(
        f"  mean largest_cc_frac {np.mean([r['largest_cc_frac'] for r in rows]):.3f}   (higher = connected tree, not speckle)"
    )
    print(
        f"  mean n_components   {np.mean([r['n_components'] for r in rows]):,.0f}   (lower = less speckle)"
    )
    print(f"appended -> {out}")


if __name__ == "__main__":
    main()
