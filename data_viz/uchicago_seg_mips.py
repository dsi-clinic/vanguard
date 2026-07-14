#!/usr/bin/env python3
"""MIP panels for the UChicago vessel segmentation, and a baseline-vs-external compare.

Renders, per case, three-plane maximum-intensity projections of the vessel
segmentation, for one run or for two runs side by side (model-predicted STEP-2
breast masks vs. the external BreastDivider whole-breast masks).

Two choices here are deliberate, and both are corrections of earlier mistakes:

* **Timepoint selection is by vessel content, never a fixed phase index.** DCE
  phases differ enormously in vessel conspicuity (pre-contrast phases have almost
  none), so a hardcoded phase is not a QC view -- it is a lottery. By default this
  collapses the whole time series with a per-voxel max over phases
  (``--phase-mode time-max``), which uses every timepoint and cannot be unlucky.
  ``--phase-mode richest`` instead picks the single phase with the most vessel
  voxels *summed across both runs*, so the two runs are always compared on the
  SAME phase.

* **Matched phases when comparing runs.** An earlier comparison picked each run's
  richest phase independently, so 4/5 patients were compared across *different*
  phases -- one pair scored Dice 0.0 purely because of that, not because of any
  real difference. Any per-phase quantity here is computed on the same phase in
  both runs.

Vessel outputs are ``<run>/<sub_source>/<case>/images/<case>_phase_XXXX_vessel_segmentation.npz``
with a single ``vessel`` array (float16 probabilities) in the model's
(rows, cols, slices) layout.

Example:
    python data_viz/uchicago_seg_mips.py \
        --run-a /path/uchicago_vessel_seg_v3_model_breast    --label-a "model breast mask" \
        --run-b /path/uchicago_vessel_seg_v3_external_breast --label-b "external breast mask" \
        --out-dir data_viz/uchicago_seg_v3
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

VESSEL_THRESHOLD = 0.5
PHASE_RE = re.compile(r"_phase_(\d+)_vessel_segmentation\.npz$")
PLANE_NAMES = (
    "axial (max over slices)",
    "coronal (max over rows)",
    "sagittal (max over cols)",
)


def find_cases(run: Path) -> dict[str, Path]:
    """Map case_id -> its images dir, for every case with vessel output in a run."""
    cases: dict[str, Path] = {}
    for images_dir in run.glob("*/*/images"):
        if any(images_dir.glob("*_vessel_segmentation.npz")):
            cases[images_dir.parent.name] = images_dir
    return cases


def load_phases(images_dir: Path) -> dict[int, Path]:
    """Map phase index -> npz path."""
    out: dict[int, Path] = {}
    for f in images_dir.glob("*_vessel_segmentation.npz"):
        m = PHASE_RE.search(f.name)
        if m:
            out[int(m.group(1))] = f
    return out


def load_vessel(path: Path) -> np.ndarray:
    """Load one phase's vessel probability volume as float32."""
    with np.load(path) as d:
        return d["vessel"].astype(np.float32)


def time_max(images_dir: Path) -> np.ndarray:
    """Per-voxel max over every phase -- uses the whole time series, not one phase."""
    phases = load_phases(images_dir)
    if not phases:
        raise SystemExit(f"no vessel output under {images_dir}")
    acc: np.ndarray | None = None
    for _, p in sorted(phases.items()):
        v = load_vessel(p)
        acc = v if acc is None else np.maximum(acc, v)
    return acc


def mips(volume: np.ndarray) -> list[np.ndarray]:
    """Three-plane MIPs of a (rows, cols, slices) volume."""
    return [volume.max(axis=2), volume.max(axis=0), volume.max(axis=1)]


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice between two binarized vessel volumes."""
    a_b, b_b = a > VESSEL_THRESHOLD, b > VESSEL_THRESHOLD
    denom = a_b.sum() + b_b.sum()
    if denom == 0:
        return float("nan")
    return float(2.0 * np.logical_and(a_b, b_b).sum() / denom)


def render(case_id: str, panels: list[tuple[str, np.ndarray]], out_png: Path) -> None:
    """Render one row of three-plane MIPs per run."""
    nrows = len(panels)
    fig, axes = plt.subplots(nrows, 3, figsize=(13, 4.3 * nrows), squeeze=False)
    for r, (label, vol) in enumerate(panels):
        for c, (plane, img) in enumerate(zip(PLANE_NAMES, mips(vol), strict=True)):
            ax = axes[r][c]
            ax.imshow(img.T, cmap="magma", origin="lower", vmin=0, vmax=1)
            ax.set_title(f"{label}\n{plane}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[r][0].set_ylabel(label, fontsize=9)
    n_vox = " | ".join(
        f"{lab}: {(v > VESSEL_THRESHOLD).sum():,} vessel vox" for lab, v in panels
    )
    fig.suptitle(f"{case_id}\n{n_vox}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main() -> None:
    """Render MIP panels for one or two runs, and compare them on matched phases."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-a", required=True, help="vessel-segmentation output dir")
    p.add_argument("--label-a", default="run A")
    p.add_argument("--run-b", default=None, help="optional second run, compared to A")
    p.add_argument("--label-b", default="run B")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--phase-mode", choices=("time-max", "richest"), default="time-max")
    args = p.parse_args()

    run_a, out_dir = Path(args.run_a), Path(args.out_dir)
    run_b = Path(args.run_b) if args.run_b else None
    cases_a = find_cases(run_a)
    if not cases_a:
        raise SystemExit(f"no vessel output found under {run_a}")
    cases_b = find_cases(run_b) if run_b else {}

    if run_b:
        only_a, only_b = set(cases_a) - set(cases_b), set(cases_b) - set(cases_a)
        if only_a or only_b:
            print(
                f"[warn] cases not in both runs -- only A: {sorted(only_a)}; only B: {sorted(only_b)}"
            )

    rows: list[dict[str, object]] = []
    for case_id in sorted(set(cases_a) & set(cases_b)) if run_b else sorted(cases_a):
        panels: list[tuple[str, np.ndarray]] = []

        if args.phase_mode == "time-max":
            vol_a = time_max(cases_a[case_id])
            panels.append((f"{args.label_a} (time-max)", vol_a))
            vol_b = None
            if run_b:
                vol_b = time_max(cases_b[case_id])
                panels.append((f"{args.label_b} (time-max)", vol_b))
            phase_note = "time-max over all phases"
        else:
            ph_a = load_phases(cases_a[case_id])
            ph_b = load_phases(cases_b[case_id]) if run_b else {}
            shared = sorted(set(ph_a) & set(ph_b)) if run_b else sorted(ph_a)

            # richest by COMBINED content, so both runs are read on the SAME phase
            def content(
                ph: int,
                ph_a: dict[int, Path] = ph_a,
                ph_b: dict[int, Path] = ph_b,
            ) -> int:
                tot = (load_vessel(ph_a[ph]) > VESSEL_THRESHOLD).sum()
                if ph_b:
                    tot += (load_vessel(ph_b[ph]) > VESSEL_THRESHOLD).sum()
                return int(tot)

            best = max(shared, key=content)
            vol_a = load_vessel(ph_a[best])
            panels.append((f"{args.label_a} (phase {best:04d})", vol_a))
            vol_b = load_vessel(ph_b[best]) if run_b else None
            if vol_b is not None:
                panels.append((f"{args.label_b} (phase {best:04d})", vol_b))
            phase_note = f"matched phase {best:04d}"

        render(case_id, panels, out_dir / f"{case_id}.png")

        row: dict[str, object] = {
            "case_id": case_id,
            "phase_mode": phase_note,
            f"vessel_voxels_{args.label_a.replace(' ', '_')}": int(
                (vol_a > VESSEL_THRESHOLD).sum()
            ),
        }
        if vol_b is not None:
            row[f"vessel_voxels_{args.label_b.replace(' ', '_')}"] = int(
                (vol_b > VESSEL_THRESHOLD).sum()
            )
            row["dice_a_vs_b"] = round(dice(vol_a, vol_b), 4)
        rows.append(row)
        print(f"  {case_id[:34]:36s} {row}")

    out_csv = out_dir / "summary.csv"
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} panels + {out_csv}")


if __name__ == "__main__":
    main()
