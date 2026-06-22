#!/usr/bin/env python3
"""Validate fast-pipeline vessel outputs against the completed ground-truth run.

Compares each ``*_vessel_segmentation.npz`` produced by the fast pipeline with the
matching file from the existing (52%-complete) run, reporting metrics that show
whether accuracy is preserved:

- shape agreement
- max / mean absolute probability difference
- Dice coefficient of the thresholded masks (default thr=0.5)
- vessel voxel counts (truth vs fast) and their ratio

Writes a summary to ``validation_report.txt`` in the output dir of this script.

This is lightweight (loads two float16 volumes at a time) and safe on the login
node. Run:

    python faster-segmentation-test/validate_outputs.py \\
        --fast-dir  /ess/scratch/scratch1/t-9sbose/vessel_segmentations_fast_smoke \\
        --truth-dir /ess/scratch/scratch1/t-9sbose/vessel_segmentations
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DEFAULT_TRUTH = "/ess/scratch/scratch1/t-9sbose/vessel_segmentations"


def dice(a: np.ndarray, b: np.ndarray, thr: float) -> float:
    A, B = a > thr, b > thr
    s = int(A.sum()) + int(B.sum())
    if s == 0:
        return 1.0
    return 2.0 * float((A & B).sum()) / s


def find_truth(truth_root: Path, name: str) -> Path | None:
    hits = list(truth_root.rglob(name))
    return hits[0] if hits else None


def main() -> int:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--fast-dir", required=True, help="Root of fast-pipeline outputs")
    ap.add_argument("--truth-dir", default=DEFAULT_TRUTH, help="Root of ground-truth outputs")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--report", default=None, help="Report path (default: <fast-dir>/validation_report.txt)")
    args = ap.parse_args()

    fast_root = Path(args.fast_dir)
    truth_root = Path(args.truth_dir)
    report_path = Path(args.report) if args.report else fast_root / "validation_report.txt"

    fast_files = sorted(fast_root.rglob("*_vessel_segmentation.npz"))
    if not fast_files:
        print(f"No fast outputs found under {fast_root}")
        return 1

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit(f"Validation: fast={fast_root}")
    emit(f"            truth={truth_root}  threshold={args.threshold}")
    emit("")
    header = f"{'case':<34} {'shape_ok':<9} {'max_abs':<9} {'mean_abs':<10} {'dice':<7} {'vox_truth':<11} {'vox_fast':<10} {'ratio':<6}"
    emit(header)
    emit("-" * len(header))

    n_compared = 0
    dice_vals = []
    max_abs_vals = []
    for fp in fast_files:
        truth = find_truth(truth_root, fp.name)
        if truth is None:
            emit(f"{fp.name:<34} (no matching ground-truth file — skipped)")
            continue
        a = np.load(fp)["vessel"].astype(np.float32)
        b = np.load(truth)["vessel"].astype(np.float32)
        shape_ok = a.shape == b.shape
        if not shape_ok:
            emit(f"{fp.name:<34} SHAPE MISMATCH fast={a.shape} truth={b.shape}")
            continue
        max_abs = float(np.max(np.abs(a - b)))
        mean_abs = float(np.mean(np.abs(a - b)))
        d = dice(a, b, args.threshold)
        vt = int((b > args.threshold).sum())
        vf = int((a > args.threshold).sum())
        ratio = (vf / vt) if vt else float("nan")
        emit(f"{fp.name:<34} {str(shape_ok):<9} {max_abs:<9.4f} {mean_abs:<10.6f} {d:<7.4f} {vt:<11} {vf:<10} {ratio:<6.3f}")
        n_compared += 1
        dice_vals.append(d)
        max_abs_vals.append(max_abs)

    emit("")
    if n_compared:
        emit(f"Compared {n_compared} case(s).")
        emit(f"  Dice:     min={min(dice_vals):.4f}  mean={np.mean(dice_vals):.4f}")
        emit(f"  Max|diff|: max={max(max_abs_vals):.4f}  mean={np.mean(max_abs_vals):.4f}")
        emit("")
        emit("Interpretation: Dice ~1.0 and small max|diff| (float16 rounding scale)")
        emit("means the fast pipeline reproduces the ground truth within precision.")
    else:
        emit("No overlapping cases compared. Pick a fast-pipeline index range whose")
        emit("cases already exist in the ground-truth run.")

    report_path.write_text("\n".join(lines) + "\n")
    print(f"\nReport written to {report_path}")
    return 0 if n_compared else 2


if __name__ == "__main__":
    raise SystemExit(main())
