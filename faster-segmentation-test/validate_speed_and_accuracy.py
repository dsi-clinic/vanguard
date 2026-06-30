#!/usr/bin/env python3
r"""Validate the fast pipeline against the old pipeline: speed AND accuracy.

Runs ``segmentation/batch_segmentation.py`` (old) and
``faster-segmentation-test/batch_segmentation_fast.py`` (fast) on the *same* N
sample files, one file at a time, using the index-based selection both scripts
already expose (``--file-index`` on the old script, ``--file-start``/``--file-end``
on the fast script). File discovery, sorting, and indexing are imported directly
from ``segmentation/batch_segmentation.py`` so both pipelines see an identical
file list — this script does not introduce any new sampling logic.

For each file it reports: old runtime, fast runtime, output shape match, Dice
coefficient of the thresholded vessel masks (thr=0.5), and max/mean absolute
probability difference. Aggregates (mean +/- std) are reported across files.
Results are printed as a table and written to ``validation_results.md``.

Both pipelines call into PyTorch models, so this must run where the ``vanguard``
environment + a GPU are available (i.e. via ``srun``/``sbatch`` on randi) -- it
will not work on a CPU-only login node.

Example (5 files, GPU node):

    micromamba activate vanguard
    python faster-segmentation-test/validate_speed_and_accuracy.py \\
        --images-dir /ess/scratch/scratch1/annawoodard/MAMA-MIA-syn60868042/images \\
        --file-start 0 --file-end 4 \\
        --output-root /ess/scratch/scratch1/t-9sbose/validate_speed_and_accuracy
"""

from __future__ import annotations

import argparse
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_SEG_DIR = _PROJECT_ROOT / "segmentation"
if str(_SEG_DIR) not in sys.path:
    sys.path.insert(0, str(_SEG_DIR))

# Reuse the original discovery/output-layout helpers so both pipelines are
# indexed identically -- no new sampling approach is introduced here.
from batch_segmentation import build_output_path, find_nii_files  # noqa: E402

OLD_SCRIPT = _SEG_DIR / "batch_segmentation.py"
FAST_SCRIPT = _HERE / "batch_segmentation_fast.py"
_SUBMODULE = _PROJECT_ROOT / "vanguard-blood-vessel-segmentation"
DEFAULT_BREAST_MODEL = _SUBMODULE / "trained_models" / "breast_model.pth"
DEFAULT_VESSEL_MODEL = _SUBMODULE / "trained_models" / "dv_model.pth"


def dice(a: np.ndarray, b: np.ndarray, thr: float) -> float:
    """Dice coefficient between two arrays thresholded at ``thr``."""
    A, B = a > thr, b > thr
    s = int(A.sum()) + int(B.sum())
    if s == 0:
        return 1.0
    return 2.0 * float((A & B).sum()) / s


def run_timed(cmd: list[str]) -> tuple[float, int, str]:
    """Run ``cmd``, returning (elapsed_seconds, returncode, combined_output)."""
    t0 = time.perf_counter()
    result = subprocess.run(  # noqa: S603
        cmd, capture_output=True, text=True, shell=False
    )
    elapsed = time.perf_counter() - t0
    output = result.stdout + result.stderr
    return elapsed, result.returncode, output


def build_old_cmd(
    images_dir: str,
    output_dir: Path,
    temp_dir: Path,
    breast_model: str,
    vessel_model: str,
    file_index: int,
) -> list[str]:
    """Build the single-file CLI invocation for the old pipeline."""
    return [
        sys.executable,
        str(OLD_SCRIPT),
        "--images-dir",
        images_dir,
        "--output-dir",
        str(output_dir),
        "--temp-dir",
        str(temp_dir),
        "--breast-model-path",
        breast_model,
        "--vessel-model-path",
        vessel_model,
        "--file-index",
        str(file_index),
    ]


def build_fast_cmd(
    images_dir: str,
    output_dir: Path,
    temp_dir: Path,
    breast_model: str,
    vessel_model: str,
    file_index: int,
    batch_size: int,
) -> list[str]:
    """Build the single-file CLI invocation for the fast pipeline."""
    return [
        sys.executable,
        str(FAST_SCRIPT),
        "--images-dir",
        images_dir,
        "--output-dir",
        str(output_dir),
        "--temp-dir",
        str(temp_dir),
        "--breast-model-path",
        breast_model,
        "--vessel-model-path",
        vessel_model,
        "--file-start",
        str(file_index),
        "--file-end",
        str(file_index),
        "--batch-size",
        str(batch_size),
    ]


def main() -> int:
    """Run the old and fast pipelines on the same sample files and report results."""
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--images-dir",
        default="/ess/scratch/scratch1/annawoodard/MAMA-MIA-syn60868042/images",
        help="Directory of case subdirectories with .nii.gz files (same source for both pipelines)",
    )
    ap.add_argument(
        "--file-start",
        type=int,
        default=0,
        help="Start index (inclusive) into the sorted file list",
    )
    ap.add_argument(
        "--file-end",
        type=int,
        default=6,
        help="End index (inclusive) into the sorted file list (7 files by default)",
    )
    ap.add_argument(
        "--output-root",
        required=True,
        help="Root dir for this validation run's outputs/temp dirs (old/, fast/, tmp_old/, tmp_fast/)",
    )
    ap.add_argument("--breast-model-path", default=str(DEFAULT_BREAST_MODEL))
    ap.add_argument("--vessel-model-path", default=str(DEFAULT_VESSEL_MODEL))
    ap.add_argument(
        "--batch-size", type=int, default=16, help="Fast-pipeline batch size"
    )
    ap.add_argument(
        "--threshold", type=float, default=0.5, help="Vessel-mask threshold for Dice"
    )
    ap.add_argument(
        "--report",
        default=None,
        help="Markdown report path (default: <this script's dir>/validation_results.md)",
    )
    ap.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove the temp dirs under --output-root after running",
    )
    args = ap.parse_args()

    report_path = Path(args.report) if args.report else _HERE / "validation_results.md"

    output_root = Path(args.output_root)
    old_output_dir = output_root / "old"
    fast_output_dir = output_root / "fast"
    old_temp_dir = output_root / "tmp_old"
    fast_temp_dir = output_root / "tmp_fast"
    for d in (old_output_dir, fast_output_dir, old_temp_dir, fast_temp_dir):
        d.mkdir(parents=True, exist_ok=True)

    nii_files = sorted(find_nii_files(args.images_dir), key=lambda x: (x[0], str(x[1])))
    if args.file_end >= len(nii_files):
        print(
            f"--file-end {args.file_end} exceeds available files ({len(nii_files)}); clamping"
        )
        args.file_end = len(nii_files) - 1
    indices = list(range(args.file_start, args.file_end + 1))
    if not indices:
        print("No files selected, exiting.")
        return 1

    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit("# Validation: old vs. fast segmentation pipeline")
    emit()
    emit(f"- Images dir: `{args.images_dir}`")
    emit(
        f"- Sample files: indices {args.file_start}-{args.file_end} ({len(indices)} files)"
    )
    emit(f"- Old output dir: `{old_output_dir}`")
    emit(f"- Fast output dir: `{fast_output_dir}`")
    emit(f"- Threshold for Dice: {args.threshold}")
    emit()

    rows = []
    old_times, fast_times = [], []
    dice_vals, max_abs_vals, mean_abs_vals = [], [], []

    for idx in indices:
        case_id, file_path = nii_files[idx]
        base_name = Path(file_path).name.replace(".nii.gz", "")
        print(f"\n[{idx}] {case_id} :: {base_name}")

        old_cmd = build_old_cmd(
            args.images_dir,
            old_output_dir,
            old_temp_dir,
            args.breast_model_path,
            args.vessel_model_path,
            idx,
        )
        fast_cmd = build_fast_cmd(
            args.images_dir,
            fast_output_dir,
            fast_temp_dir,
            args.breast_model_path,
            args.vessel_model_path,
            idx,
            args.batch_size,
        )

        print(f"  old:  {' '.join(old_cmd)}")
        old_elapsed, old_rc, old_out = run_timed(old_cmd)
        if old_rc != 0:
            print(f"  OLD FAILED (rc={old_rc}):\n{old_out[-2000:]}")

        print(f"  fast: {' '.join(fast_cmd)}")
        fast_elapsed, fast_rc, fast_out = run_timed(fast_cmd)
        if fast_rc != 0:
            print(f"  FAST FAILED (rc={fast_rc}):\n{fast_out[-2000:]}")

        old_npz = build_output_path(old_output_dir, case_id, base_name)
        fast_npz = build_output_path(fast_output_dir, case_id, base_name)

        row = {
            "case": f"{case_id}/{base_name}",
            "old_s": old_elapsed,
            "fast_s": fast_elapsed,
            "shape_ok": None,
            "dice": None,
            "max_abs": None,
            "mean_abs": None,
            "note": "",
        }

        if old_rc != 0 or fast_rc != 0:
            row["note"] = "pipeline failed"
        elif not old_npz.exists() or not fast_npz.exists():
            row["note"] = "output missing"
        else:
            a = np.load(old_npz)["vessel"].astype(np.float32)
            b = np.load(fast_npz)["vessel"].astype(np.float32)
            shape_ok = a.shape == b.shape
            row["shape_ok"] = shape_ok
            if shape_ok:
                row["dice"] = dice(a, b, args.threshold)
                row["max_abs"] = float(np.max(np.abs(a - b)))
                row["mean_abs"] = float(np.mean(np.abs(a - b)))
                dice_vals.append(row["dice"])
                max_abs_vals.append(row["max_abs"])
                mean_abs_vals.append(row["mean_abs"])
            else:
                row["note"] = f"shape mismatch old={a.shape} fast={b.shape}"

        old_times.append(old_elapsed)
        fast_times.append(fast_elapsed)
        rows.append(row)

    emit("## Per-file results")
    emit()
    header = f"| {'case':<34} | {'old_s':>8} | {'fast_s':>8} | {'shape_ok':>8} | {'dice':>7} | {'max_abs':>8} | {'mean_abs':>9} | note |"
    sep = "|" + "---|" * 8
    emit(header)
    emit(sep)
    for r in rows:
        dice_str = f"{r['dice']:.4f}" if r["dice"] is not None else "n/a"
        max_abs_str = f"{r['max_abs']:.4f}" if r["max_abs"] is not None else "n/a"
        mean_abs_str = f"{r['mean_abs']:.6f}" if r["mean_abs"] is not None else "n/a"
        emit(
            f"| {r['case']:<34} | {r['old_s']:>8.2f} | {r['fast_s']:>8.2f} | "
            f"{str(r['shape_ok']):>8} | {dice_str:>7} | {max_abs_str:>8} | "
            f"{mean_abs_str:>9} | {r['note']} |"
        )

    emit()
    emit("## Summary")
    emit()
    n = len(indices)

    def fmt_mean_std(vals: list[float]) -> str:
        if not vals:
            return "n/a"
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return f"{mean:.2f} +/- {std:.2f}"

    emit(f"- Files compared: {n}")
    emit(f"- Old runtime (s): {fmt_mean_std(old_times)}")
    emit(f"- Fast runtime (s): {fmt_mean_std(fast_times)}")
    if old_times and fast_times and statistics.mean(fast_times) > 0:
        speedup = statistics.mean(old_times) / statistics.mean(fast_times)
        emit(f"- Mean speedup: {speedup:.2f}x")
    emit(f"- Shape matches: {sum(1 for r in rows if r['shape_ok'])}/{n}")
    if dice_vals:
        emit(
            f"- Dice (thr={args.threshold}): min={min(dice_vals):.4f} "
            f"mean={statistics.mean(dice_vals):.4f} max={max(dice_vals):.4f}"
        )
        emit(
            f"- Max |prob diff|: max={max(max_abs_vals):.4f} mean={statistics.mean(max_abs_vals):.4f}"
        )
        emit(f"- Mean |prob diff|: mean={statistics.mean(mean_abs_vals):.6f}")
    else:
        emit("- No file pairs had comparable output (see notes above).")

    emit()
    emit("## Exact commands used")
    emit()
    emit("```")
    emit(f"python {' '.join(sys.argv)}")
    emit("```")
    emit()
    emit(f"Old output dir: `{old_output_dir}`")
    emit()
    emit(f"Fast output dir: `{fast_output_dir}`")

    report_path.write_text("\n".join(lines) + "\n")
    print(f"\nReport written to {report_path}")

    if args.cleanup:
        shutil.rmtree(old_temp_dir, ignore_errors=True)
        shutil.rmtree(fast_temp_dir, ignore_errors=True)

    return 0 if dice_vals else 2


if __name__ == "__main__":
    raise SystemExit(main())
