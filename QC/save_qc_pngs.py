#!/usr/bin/env python3
"""Save axial MRI slices as PNGs for native-resolution QC.

For each cohort, saves N representative exams (default 5).
For each exam, saves both the pre-contrast (t0) and peak-enhancement (t2/t3)
timepoints in separate subfolders.

Output layout:
    <out-root>/<cohort>/<exam_id>/t0/slice_0000.png   (pre-contrast)
                                  t2/slice_0000.png   (peak enhancement)

Usage examples:
    # All cohorts, 5 exams each
    python scripts/save_qc_pngs.py --n 5

    # One cohort only
    python scripts/save_qc_pngs.py --cohort DUKE --n 5

    # Different data root
    python scripts/save_qc_pngs.py --data-root /net/projects2/vanguard/MAMA-MIA/images
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

_DEFAULT_DATA_ROOT = Path(
    "/ess/scratch/scratch1/annawoodard/MAMA-MIA-syn60868042/images"
)
_DEFAULT_OUT_ROOT = Path(os.environ.get("VANGUARD_QC_OUTPUT", str(Path.home() / "vanguard_qc_pngs")))

_KNOWN_COHORT_PREFIXES = ["ISPY1", "ISPY2", "NACT", "DUKE"]


def _cohort_from_case_id(case_id: str) -> str:
    for prefix in _KNOWN_COHORT_PREFIXES:
        if case_id.startswith(prefix):
            return prefix
    return case_id.split("_")[0]


def _find_nii_for_timepoint(case_dir: Path, timepoint: int) -> Path | None:
    """Return the NIfTI file for a specific timepoint index, or None if missing."""
    tag = f"_{timepoint:04d}"
    candidates = sorted(case_dir.glob(f"*{tag}.nii.gz")) + sorted(case_dir.glob(f"*{tag}.nii"))
    return candidates[0] if candidates else None


def _find_peak_enhancement_nii(case_dir: Path) -> tuple[Path | None, str]:
    """Return (path, label) for the peak-enhancement timepoint (t2, fallback t3)."""
    for tp in (2, 3):
        p = _find_nii_for_timepoint(case_dir, tp)
        if p is not None:
            return p, f"t{tp}"
    return None, ""


def _voxel_spacing(nii_path: Path) -> tuple[float, float, float]:
    import nibabel as nib  # noqa: PLC0415

    img = nib.load(str(nii_path))
    zooms = img.header.get_zooms()[:3]
    return tuple(round(float(z), 4) for z in zooms)  # type: ignore[return-value]


def _save_slices(nii_path: Path, out_dir: Path) -> int:
    """Load volume, normalise, write one PNG per axial slice. Returns slice count."""
    import nibabel as nib  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    img = nib.load(str(nii_path))
    vol = np.asarray(img.dataobj, dtype=np.float32)

    p1, p99 = np.percentile(vol, 1), np.percentile(vol, 99)
    vol = np.clip(vol, p1, p99)
    if p99 > p1:
        vol = (vol - p1) / (p99 - p1) * 255.0
    else:
        vol[:] = 0.0
    vol = vol.astype(np.uint8)

    # NIfTI shape is (X, Y, Z); PIL expects (rows=Y, cols=X), so transpose.
    n_slices = vol.shape[2]
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_slices):
        Image.fromarray(vol[:, :, i].T).save(out_dir / f"slice_{i:04d}.png")
    return n_slices


def _process_cohort(
    cohort: str,
    case_dirs: list[Path],
    out_root: Path,
    n: int,
) -> None:
    print(f"\n=== {cohort} ({len(case_dirs)} cases — saving {n}) ===")

    saved = 0
    for case_dir in case_dirs:
        if saved >= n:
            break

        t0_path = _find_nii_for_timepoint(case_dir, 0)
        peak_path, peak_label = _find_peak_enhancement_nii(case_dir)

        if t0_path is None:
            continue

        try:
            spacing = _voxel_spacing(t0_path)
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING: could not read spacing for {case_dir.name}: {exc}")
            continue

        t0_out = out_root / cohort / case_dir.name / "t0"
        already_done = t0_out.exists() and any(t0_out.glob("slice_*.png"))
        if peak_path is not None:
            peak_out = out_root / cohort / case_dir.name / peak_label
            already_done = already_done and peak_out.exists() and any(peak_out.glob("slice_*.png"))

        if already_done:
            print(f"  {case_dir.name}  already done, skipping")
            saved += 1
            continue

        print(f"  {case_dir.name}  spacing={spacing} mm")

        try:
            n_slices = _save_slices(t0_path, t0_out)
            print(f"    t0  : saved {n_slices} slices from {t0_path.name}")
        except Exception as exc:  # noqa: BLE001
            print(f"    ERROR saving t0 for {case_dir.name}: {exc}")
            continue

        if peak_path is not None:
            try:
                n_slices = _save_slices(peak_path, peak_out)
                print(f"    {peak_label} : saved {n_slices} slices from {peak_path.name}")
            except Exception as exc:  # noqa: BLE001
                print(f"    ERROR saving {peak_label} for {case_dir.name}: {exc}")
        else:
            print(f"    peak: no t2/t3 file found, skipping")

        saved += 1

    if saved == 0:
        print(f"  No processable cases found in {cohort}.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Save axial MRI slices as PNGs at native resolution for QC. "
            "Saves both pre-contrast (t0) and peak enhancement (t2/t3) "
            "for --n exams per cohort."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=_DEFAULT_DATA_ROOT,
        help="Root directory containing one subdirectory per exam (case_id).",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=_DEFAULT_OUT_ROOT,
        help="Output root. PNGs are written to <out-root>/<cohort>/<exam_id>/t0/ and /t2/.",
    )
    parser.add_argument(
        "--cohort",
        default=None,
        metavar="NAME",
        help="Process only this cohort (e.g. DUKE, ISPY1, ISPY2, NACT). "
             "Default: all cohorts.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        metavar="N",
        help="Number of representative exams to save per cohort.",
    )
    args = parser.parse_args()

    if not args.data_root.exists():
        print(f"ERROR: --data-root does not exist: {args.data_root}", file=sys.stderr)
        sys.exit(1)

    cohort_cases: dict[str, list[Path]] = {}
    for case_dir in sorted(args.data_root.iterdir()):
        if not case_dir.is_dir():
            continue
        cohort = _cohort_from_case_id(case_dir.name)
        if args.cohort and cohort != args.cohort:
            continue
        cohort_cases.setdefault(cohort, []).append(case_dir)

    if not cohort_cases:
        msg = f"No cases found under {args.data_root}"
        if args.cohort:
            msg += f" matching cohort '{args.cohort}'"
        print(f"ERROR: {msg}", file=sys.stderr)
        sys.exit(1)

    print(f"Data root : {args.data_root}")
    print(f"Output    : {args.out_root}")
    if args.cohort:
        print(f"Cohort    : {args.cohort}")

    for cohort, case_dirs in sorted(cohort_cases.items()):
        _process_cohort(cohort, case_dirs, args.out_root, args.n)

    print(f"\nDone. All PNGs saved under: {args.out_root}")


if __name__ == "__main__":
    main()
