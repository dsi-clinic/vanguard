#!/usr/bin/env python3
"""Resample MAMA-MIA MRI volumes into an acceptable voxel-spacing range.

WHY THIS EXISTS
---------------
Our vessel-segmentation model was trained on the DUKE cohort, whose voxel
spacing lives in a fairly narrow band. The other cohorts (ISPY1, ISPY2,
NACT-Pilot) were scanned with different spacing, which is a "domain shift" that
can hurt the model. This script brings every scan into a common, acceptable
spacing range so the model sees data that looks like what it trained on.

WHAT "VOXEL SPACING" MEANS
--------------------------
A 3D MRI is a grid of little boxes (voxels). "Spacing" is the physical size of
each box in millimetres: how far apart the samples are. Smaller spacing = finer
resolution = more voxels. Two axis groups matter here:
  - xy (in-plane): the resolution within a single slice
  - z  (slice):    the distance between slices (slice thickness)

THE ACCEPTABLE RANGE (defaults, overridable on the command line)
  xy: 0.60 - 1.02 mm
  z : 1.00 - 1.25 mm
For each axis we ask: is the current spacing inside the range?
  - inside           -> leave that axis alone
  - below the min    -> the voxels are too fine; target = the min edge. Going to a
                        LARGER spacing means fewer samples = DOWNSAMPLING.
  - above the max    -> the voxels are too coarse; target = the max edge. Going to a
                        SMALLER spacing means more samples = UPSAMPLING.

INTERPOLATION RULES (the heart of the script)
  - Upsampling   (current > target, making it finer): cubic B-spline (3rd order).
                 B-spline is smooth and good at inventing in-between values.
  - Downsampling (current < target, making it coarser): Gaussian blur FIRST,
                 then linear interpolation. The blur prevents "aliasing" --
                 jagged artefacts that appear when you throw away samples without
                 smoothing first (the same reason you blur before shrinking a photo).
                 Blur strength: sigma = sigma_factor * target_spacing (mm),
                 sigma_factor defaults to 0.5 (i.e. half the new voxel size).

THE MIXED-AXIS PROBLEM AND THE TWO-PASS SOLUTION
SimpleITK applies ONE interpolator per resample call. But it's possible for xy to
need upsampling while z needs downsampling (different methods). So we resample in
up to two passes:
  Pass A: handle the axes that need DOWNSAMPLING (blur -> linear), leaving the
          other axes' spacing untouched.
  Pass B: handle the axes that need UPSAMPLING (B-spline), leaving the rest untouched.
Axes already in range are never touched. A case that needs the same method on all
changed axes simply collapses to one pass. A case in range on every axis isn't
resampled at all -- we byte-copy it, which is far faster and bit-for-bit exact.

CONSISTENCY ACROSS TIMEPOINTS
Each patient has several DCE timepoints (_0000, _0001, ...) that share identical
geometry. We compute ONE target spacing from the _0000 header and apply that SAME
target to every timepoint, so the whole series stays aligned.

This is a CPU-only program (SimpleITK has no GPU path). It is meant to run inside
a Slurm job array; see scripts/slurm/submit_resample_array.slurm.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
from multiprocessing import Pool

import SimpleITK as sitk


# --- Where the data lives on Randi -------------------------------------------
# NOTE: the old segmentation scripts point at /net/projects2/vanguard, which is
# NOT mounted on Randi. The real MAMA-MIA NIfTI files live under this scratch path
# (owned by another user -- we only ever READ from here).
_DEFAULT_DATA_ROOT = "/ess/scratch/scratch1/annawoodard/MAMA-MIA-syn60868042/images"
_DEFAULT_OUTPUT_ROOT = "/ess/scratch/scratch1/t-9sbose/resampled_segmentation"

# Directory-name prefix -> cohort label used for the output folder. ISPY1 is
# listed before ISPY2 only for readability; "ISPY1_" can't match an "ISPY2_" name.
# The pilot NACT cohort is stored under the "NACT" prefix but we label it "NACT-Pilot".
_COHORTS = [
    ("DUKE", "DUKE"),
    ("ISPY1", "ISPY1"),
    ("ISPY2", "ISPY2"),
    ("NACT", "NACT-Pilot"),
]

_MANIFEST_NAME = "SYNAPSE_METADATA_MANIFEST.tsv"


def cohort_label(case_name: str) -> str | None:
    """Map a case folder name (e.g. 'NACT_01') to its cohort label, or None."""
    for prefix, label in _COHORTS:
        if case_name.startswith(prefix):
            return label
    return None


def list_cases(data_root: str) -> list[tuple[str, str, str]]:
    """Return a deterministic, sorted list of (case_name, cohort_label, case_dir).

    Sorting is important: every array task builds this same list and then takes a
    slice of it by index, so the slices must line up across tasks.
    """
    cases = []
    for name in sorted(os.listdir(data_root)):
        case_dir = os.path.join(data_root, name)
        if not os.path.isdir(case_dir):
            continue  # skips stray files like the top-level manifest .tsv
        label = cohort_label(name)
        if label is None:
            continue  # not one of our four cohorts
        cases.append((name, label, case_dir))
    return cases


def timepoint_files(case_dir: str) -> list[str]:
    """All NIfTI timepoints for a case (3-6 of them), sorted, .mp4/.tsv excluded.

    The four-digit glob matches DUKE_001_0000.nii.gz, _0001, ... and nothing else.
    """
    return sorted(glob.glob(os.path.join(case_dir, "*_[0-9][0-9][0-9][0-9].nii.gz")))


def read_spacing(path: str) -> tuple[float, float, float]:
    """Read (x, y, z) spacing in mm WITHOUT loading the pixel array.

    ReadImageInformation pulls just the small header, so this is cheap -- it lets
    us decide the resampling plan before paying to load the full volume.
    """
    reader = sitk.ImageFileReader()
    reader.SetFileName(path)
    reader.ReadImageInformation()
    sx, sy, sz = reader.GetSpacing()
    return float(sx), float(sy), float(sz)


def plan_axis(current: float, lo: float, hi: float) -> tuple[float, str]:
    """Decide the target spacing and method for one axis.

    Returns (target_spacing, method) where method is 'up', 'down', or 'none'.
    """
    if current < lo:
        # Spacing too fine -> snap UP to the min edge -> coarser -> downsample.
        return lo, "down"
    if current > hi:
        # Spacing too coarse -> snap DOWN to the max edge -> finer -> upsample.
        return hi, "up"
    return current, "none"


def resample(img: sitk.Image, new_spacing: list[float], interpolator: int) -> sitk.Image:
    """Resample `img` to `new_spacing` (mm) using the given interpolator.

    The new voxel count along each axis is chosen so the image covers the same
    physical extent: new_size = round(old_size * old_spacing / new_spacing).
    Origin and direction are preserved so the scan stays in the same place in space.
    """
    old_spacing = img.GetSpacing()
    old_size = img.GetSize()
    new_size = [
        max(1, int(round(osz * ospc / nspc)))
        for osz, ospc, nspc in zip(old_size, old_spacing, new_spacing)
    ]

    rf = sitk.ResampleImageFilter()
    rf.SetOutputSpacing(new_spacing)
    rf.SetSize(new_size)
    rf.SetOutputOrigin(img.GetOrigin())
    rf.SetOutputDirection(img.GetDirection())
    rf.SetOutputPixelType(img.GetPixelID())  # we work in float32 here
    rf.SetDefaultPixelValue(0.0)
    rf.SetTransform(sitk.Transform())  # identity: no rotation/translation, just a grid change
    rf.SetInterpolator(interpolator)
    return rf.Execute(img)


def process_image(
    img: sitk.Image,
    targets: list[float],
    methods: list[str],
    sigma_factor: float,
) -> sitk.Image:
    """Apply the two-pass resampling plan to a single (already float32) volume."""
    down_axes = [i for i, m in enumerate(methods) if m == "down"]
    up_axes = [i for i, m in enumerate(methods) if m == "up"]

    # --- Pass A: downsampling axes -> Gaussian blur, then linear interpolation ---
    if down_axes:
        # DiscreteGaussian blurs per-axis; variance is sigma^2 in physical (mm^2)
        # units when useImageSpacing=True. We blur ONLY the axes we're about to
        # shrink (variance 0 = no blur) so we don't soften the others.
        variance = [0.0, 0.0, 0.0]
        for i in down_axes:
            sigma = sigma_factor * targets[i]
            variance[i] = sigma * sigma
        img = sitk.DiscreteGaussian(
            img, variance=variance, useImageSpacing=True, maximumKernelWidth=64
        )
        new_spacing = list(img.GetSpacing())
        for i in down_axes:
            new_spacing[i] = targets[i]
        img = resample(img, new_spacing, sitk.sitkLinear)

    # --- Pass B: upsampling axes -> cubic B-spline interpolation -----------------
    if up_axes:
        new_spacing = list(img.GetSpacing())
        for i in up_axes:
            new_spacing[i] = targets[i]
        img = resample(img, new_spacing, sitk.sitkBSpline)

    return img


def atomic_write_image(img: sitk.Image, dest: str) -> None:
    """Write a NIfTI safely: write to a temp file in the same dir, then rename.

    os.replace is atomic on a normal filesystem, so a crashed/preempted job can
    never leave a half-written .nii.gz behind -- the file is either absent or whole.
    The temp name still ends in .nii.gz so SimpleITK gzip-compresses it correctly.
    """
    tmp = os.path.join(os.path.dirname(dest), ".partial_" + os.path.basename(dest))
    sitk.WriteImage(img, tmp)
    os.replace(tmp, dest)


def atomic_copy(src: str, dest: str) -> None:
    """Byte-copy a file atomically (used for in-range scans and the manifest)."""
    tmp = os.path.join(os.path.dirname(dest), ".partial_" + os.path.basename(dest))
    shutil.copy2(src, tmp)
    os.replace(tmp, dest)


# These globals are set once per worker process by _init_worker so we don't pass
# the (small, read-only) config through every Pool call.
_CFG: dict = {}


def _init_worker(cfg: dict) -> None:
    """Runs once when each worker process starts."""
    # Pin ITK to a single thread PER PROCESS. We get our parallelism from the
    # process Pool (one case per worker); letting each also spawn many ITK threads
    # would oversubscribe the cores and actually run slower.
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    _CFG.update(cfg)


def process_case(case: tuple[str, str, str]) -> str:
    """Resample (or copy) every timepoint of one case. Returns a one-line summary."""
    name, label, case_dir = case
    out_dir = os.path.join(_CFG["output_root"], label, name)
    if not _CFG["dry_run"]:
        os.makedirs(out_dir, exist_ok=True)

    files = timepoint_files(case_dir)
    if not files:
        return f"SKIP   {name}: no NIfTI timepoints found"

    # Pick the _0000 timepoint as the reference for the spacing decision; fall back
    # to the first file if _0000 is missing for some reason.
    ref = next((f for f in files if f.endswith("_0000.nii.gz")), files[0])
    sx, sy, sz = read_spacing(ref)

    # Decide target + method PER AXIS. x and y use the xy range; z uses the z range.
    tx, mx = plan_axis(sx, _CFG["xy_min"], _CFG["xy_max"])
    ty, my = plan_axis(sy, _CFG["xy_min"], _CFG["xy_max"])
    tz, mz = plan_axis(sz, _CFG["z_min"], _CFG["z_max"])
    targets = [tx, ty, tz]
    methods = [mx, my, mz]

    needs_resample = any(m != "none" for m in methods)

    # Always carry the manifest along for provenance (if present and not already there).
    manifest_src = os.path.join(case_dir, _MANIFEST_NAME)
    manifest_dest = os.path.join(out_dir, _MANIFEST_NAME)
    if (
        not _CFG["dry_run"]
        and os.path.exists(manifest_src)
        and not (_CFG["resume"] and os.path.exists(manifest_dest))
    ):
        atomic_copy(manifest_src, manifest_dest)

    n_done = n_copied = n_resampled = 0
    for src in files:
        dest = os.path.join(out_dir, os.path.basename(src))
        if _CFG["resume"] and os.path.exists(dest):
            n_done += 1
            continue
        if _CFG["dry_run"]:
            continue

        if not needs_resample:
            # In range on every axis -> exact byte copy (fast, lossless).
            atomic_copy(src, dest)
            n_copied += 1
            continue

        img = sitk.ReadImage(src)
        orig_pixel_id = img.GetPixelID()
        # Find the original intensity range so we can clamp afterwards. B-spline can
        # "overshoot" (ring) past the original min/max; clamping keeps values sane and
        # prevents integer wrap-around when we cast back to the original dtype.
        mm = sitk.MinimumMaximumImageFilter()
        mm.Execute(img)
        omin, omax = mm.GetMinimum(), mm.GetMaximum()

        # Do all the maths in float32 to avoid integer rounding mid-pipeline.
        img = sitk.Cast(img, sitk.sitkFloat32)
        img = process_image(img, targets, methods, _CFG["sigma_factor"])
        img = sitk.Clamp(img, lowerBound=omin, upperBound=omax)
        img = sitk.Cast(img, orig_pixel_id)  # back to the original dtype (e.g. int16)

        atomic_write_image(img, dest)
        n_resampled += 1

    action = "copy" if not needs_resample else f"resample xy={mx}/z={mz}"
    plan = f"xy {sx:.3f}->{tx:.3f}  z {sz:.3f}->{tz:.3f}"
    return (
        f"OK     {name}: {action} | {plan} | "
        f"resampled={n_resampled} copied={n_copied} skipped={n_done}"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default=_DEFAULT_DATA_ROOT)
    p.add_argument("--output-root", default=_DEFAULT_OUTPUT_ROOT)
    p.add_argument("--xy-min", type=float, default=0.60)
    p.add_argument("--xy-max", type=float, default=1.02)
    p.add_argument("--z-min", type=float, default=1.00)
    p.add_argument("--z-max", type=float, default=1.25)
    p.add_argument(
        "--sigma-factor",
        type=float,
        default=0.5,
        help="Gaussian sigma (mm) = sigma_factor * target_spacing on a downsampled axis.",
    )
    p.add_argument(
        "--case-start",
        type=int,
        default=0,
        help="Index of the first case (into the sorted case list) to process.",
    )
    p.add_argument(
        "--case-end",
        type=int,
        default=-1,
        help="Index of the LAST case to process (inclusive). -1 means 'to the end'.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)),
        help="Number of parallel worker processes (defaults to the task's CPU count).",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip outputs that already exist (safe to re-run after a failure).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan for each case but write nothing.",
    )
    args = p.parse_args()

    os.makedirs(args.output_root, exist_ok=True)

    cases = list_cases(args.data_root)
    n_total = len(cases)
    start = max(0, args.case_start)
    end = n_total - 1 if args.case_end < 0 else min(args.case_end, n_total - 1)
    chunk = cases[start : end + 1]

    print(f"Data root   : {args.data_root}")
    print(f"Output root : {args.output_root}")
    print(f"Range       : xy [{args.xy_min}, {args.xy_max}]  z [{args.z_min}, {args.z_max}]")
    print(f"Sigma factor: {args.sigma_factor}  (sigma_mm = factor * target)")
    print(f"Total cases : {n_total}")
    print(f"This task   : cases [{start}..{end}]  ({len(chunk)} cases)  workers={args.workers}")
    print(f"Resume      : {args.resume}   Dry-run: {args.dry_run}")
    print("-" * 70, flush=True)

    if not chunk:
        print("Nothing to do for this index range.")
        return 0

    cfg = {
        "output_root": args.output_root,
        "xy_min": args.xy_min,
        "xy_max": args.xy_max,
        "z_min": args.z_min,
        "z_max": args.z_max,
        "sigma_factor": args.sigma_factor,
        "resume": args.resume,
        "dry_run": args.dry_run,
    }

    # A process Pool runs `args.workers` cases at the same time. imap_unordered
    # streams results back as soon as each case finishes, so we see live progress.
    failures = 0
    with Pool(processes=args.workers, initializer=_init_worker, initargs=(cfg,)) as pool:
        for i, line in enumerate(pool.imap_unordered(process_case, chunk), 1):
            print(f"[{i}/{len(chunk)}] {line}", flush=True)
            if line.startswith("SKIP"):
                failures += 1

    print("-" * 70)
    print(f"Done. {len(chunk) - failures}/{len(chunk)} cases processed cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
