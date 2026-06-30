#!/usr/bin/env python3
"""Resample expert tumor masks onto the RESAMPLED image grid.

WHY THIS EXISTS
---------------
We resampled the MRI volumes into a common voxel-spacing range (see
``scripts/resample_to_range.py``) and segmented those resampled images, so the
vessel skeletons / centerlines now live on the *resampled* voxel grid. But the
expert tumor masks are still on each scan's *native* grid. The two grids have
different voxel counts and spacing, so the tumor-graph / kinematic feature step
cannot line the tumor up with the skeleton -- it fails with a
"mask shape mismatch" and skips those feature blocks.

This script fixes that by resampling every expert tumor mask onto the matching
resampled image's grid, so a case's mask, segmentation, and skeleton all share
one coordinate system.

HOW IT "FOLLOWS HOW WE RESAMPLED THE ACTUAL DATA"
-------------------------------------------------
The resampled NIfTI images written by ``resample_to_range.py`` (under
``--reference-root``) ARE the resampled grid. Instead of recomputing the spacing
plan, we read the case's resampled image and use it directly as the geometric
reference (SimpleITK ``SetReferenceImage``): same size, spacing, origin, and
direction, by construction. That guarantees the mask ends up on exactly the grid
the segmentation/skeleton were built on -- no rounding drift.

THE ONE DELIBERATE DIFFERENCE: INTERPOLATION
--------------------------------------------
``resample_to_range.py`` uses B-spline (up) and Gaussian-blur + linear (down)
because an MRI is a continuous intensity image. A tumor mask is a LABEL image
(integer class per voxel). Linear/B-spline would invent fractional, meaningless
"in-between" labels and blur the boundary, so we use NEAREST-NEIGHBOUR
interpolation, which copies the closest label and keeps the mask integer-valued.
This is the standard, correct way to resample a segmentation mask.

OUTPUT LAYOUT
-------------
A FLAT directory of ``<case_id>.nii.gz`` files, matching the expert-mask folder
layout the graph-extraction pipeline expects for ``--tumor-mask-dir``.

This is CPU-only (SimpleITK has no GPU path). It is light per case (one NN
resample), so it runs comfortably as a single multi-worker Slurm job; see
``scripts/slurm/submit_resample_tumor_masks.slurm``.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from multiprocessing import Pool

import SimpleITK as sitk


# --- Default paths on Randi --------------------------------------------------
# Expert tumor masks (read-only; owned by another user). Flat <case_id>.nii.gz.
_DEFAULT_MASK_DIR = (
    "/ess/scratch/scratch1/annawoodard/MAMA-MIA-syn60868042/segmentations/expert"
)
# Resampled images produced by resample_to_range.py: <cohort>/<case>/<case>_0000.nii.gz
_DEFAULT_REFERENCE_ROOT = "/ess/scratch/scratch1/t-9sbose/resampled_segmentation"
# Where the resampled masks go (derived data, regenerable from this script).
_DEFAULT_OUTPUT_ROOT = (
    "/gpfs/data/karczmar-lab/workspaces/saritbose/resampled-tumor-masks"
)

# Case-name prefix -> the cohort sub-folder name used under --reference-root.
# NOTE: resample_to_range.py writes the pilot NACT cohort under "NACT-Pilot",
# even though the case ids start with "NACT_". The others match their prefix.
_COHORT_DIRS = [
    ("DUKE", "DUKE"),
    ("ISPY1", "ISPY1"),
    ("ISPY2", "ISPY2"),
    ("NACT", "NACT-Pilot"),
]


def reference_cohort_dir(case_name: str) -> str | None:
    """Map a case id (e.g. 'ISPY2_416434') to its reference cohort folder."""
    for prefix, folder in _COHORT_DIRS:
        if case_name.startswith(prefix):
            return folder
    return None


def find_reference_image(reference_root: str, case_name: str) -> str | None:
    """Locate the resampled reference image (any timepoint) for one case.

    Prefer the _0000 timepoint under the expected cohort folder; fall back to a
    glob so we still work if the cohort naming ever changes.
    """
    folder = reference_cohort_dir(case_name)
    if folder is not None:
        cand = os.path.join(
            reference_root, folder, case_name, f"{case_name}_0000.nii.gz"
        )
        if os.path.exists(cand):
            return cand
    # Fallbacks: any timepoint under the expected folder, then anywhere.
    patterns = []
    if folder is not None:
        patterns.append(
            os.path.join(reference_root, folder, case_name, f"{case_name}_*.nii.gz")
        )
    patterns.append(
        os.path.join(reference_root, "*", case_name, f"{case_name}_*.nii.gz")
    )
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return None


def list_cases(mask_dir: str, cohort: str | None) -> list[str]:
    """Sorted list of case ids that have an expert mask, optionally one cohort.

    Sorting makes the list deterministic so array-style slicing (if ever used)
    lines up across tasks.
    """
    cases = []
    for fname in sorted(os.listdir(mask_dir)):
        if not fname.endswith(".nii.gz"):
            continue
        case = fname[: -len(".nii.gz")]
        if cohort and not case.startswith(cohort):
            continue
        cases.append(case)
    return cases


def resample_mask_to_reference(mask: sitk.Image, reference: sitk.Image) -> sitk.Image:
    """Resample a label mask onto `reference`'s grid with nearest-neighbour.

    Using the reference image directly (rather than recomputing a spacing plan)
    means the output shares the reference's size/spacing/origin/direction
    exactly -- the same grid the segmentation and skeleton sit on.
    """
    rf = sitk.ResampleImageFilter()
    rf.SetReferenceImage(reference)               # copy grid from the resampled image
    rf.SetInterpolator(sitk.sitkNearestNeighbor)  # label-preserving (no blur/overshoot)
    rf.SetDefaultPixelValue(0)                     # voxels outside the mask -> background
    rf.SetTransform(sitk.Transform())             # identity: pure grid change, no warp
    rf.SetOutputPixelType(mask.GetPixelID())      # keep the original integer label dtype
    return rf.Execute(mask)


def atomic_write_image(img: sitk.Image, dest: str) -> None:
    """Write a NIfTI safely: temp file in the same dir, then atomic rename.

    A crashed/preempted job can then never leave a half-written .nii.gz behind.
    The temp name still ends in .nii.gz so SimpleITK gzip-compresses correctly.
    """
    tmp = os.path.join(os.path.dirname(dest), ".partial_" + os.path.basename(dest))
    sitk.WriteImage(img, tmp)
    os.replace(tmp, dest)


# Globals set once per worker by _init_worker so we don't pass config every call.
_CFG: dict = {}


def _init_worker(cfg: dict) -> None:
    """Runs once when each worker process starts."""
    # One ITK thread per process: parallelism comes from the process Pool, so
    # letting each process also spawn many ITK threads would oversubscribe cores.
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    _CFG.update(cfg)


def process_case(case: str) -> str:
    """Resample one case's tumor mask. Returns a one-line status string."""
    dest = os.path.join(_CFG["output_root"], f"{case}.nii.gz")
    if _CFG["resume"] and os.path.exists(dest):
        return f"SKIP   {case}: output exists"

    ref_path = find_reference_image(_CFG["reference_root"], case)
    if ref_path is None:
        return f"MISS   {case}: no resampled reference image found"

    mask_path = os.path.join(_CFG["mask_dir"], f"{case}.nii.gz")
    if not os.path.exists(mask_path):
        return f"MISS   {case}: no expert mask at {mask_path}"

    if _CFG["dry_run"]:
        return f"DRY    {case}: would resample mask -> {dest} (ref {os.path.basename(ref_path)})"

    reference = sitk.ReadImage(ref_path)
    mask = sitk.ReadImage(mask_path)
    out = resample_mask_to_reference(mask, reference)
    os.makedirs(_CFG["output_root"], exist_ok=True)
    atomic_write_image(out, dest)

    return (
        f"OK     {case}: {tuple(mask.GetSize())} -> {tuple(out.GetSize())} "
        f"(ref grid {tuple(reference.GetSize())})"
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mask-dir", default=_DEFAULT_MASK_DIR,
                   help="Flat dir of expert tumor masks (<case_id>.nii.gz).")
    p.add_argument("--reference-root", default=_DEFAULT_REFERENCE_ROOT,
                   help="Resampled images root: <cohort>/<case>/<case>_0000.nii.gz.")
    p.add_argument("--output-root", default=_DEFAULT_OUTPUT_ROOT,
                   help="Flat output dir for resampled masks (<case_id>.nii.gz).")
    p.add_argument("--cohort", default=None,
                   help="Only process this cohort prefix (e.g. ISPY2). Default: all.")
    p.add_argument("--workers", type=int,
                   default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)),
                   help="Parallel worker processes (defaults to the task's CPU count).")
    p.add_argument("--resume", action="store_true",
                   help="Skip cases whose output already exists (safe to re-run).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan per case but write nothing.")
    args = p.parse_args()

    cases = list_cases(args.mask_dir, args.cohort)

    print(f"Mask dir       : {args.mask_dir}")
    print(f"Reference root : {args.reference_root}")
    print(f"Output root    : {args.output_root}")
    print(f"Cohort filter  : {args.cohort or '(all)'}")
    print(f"Interpolation  : nearest-neighbour (label-preserving)")
    print(f"Cases to do    : {len(cases)}   workers={args.workers}")
    print(f"Resume         : {args.resume}   Dry-run: {args.dry_run}")
    print("-" * 70, flush=True)

    if not cases:
        print("No cases found; nothing to do.")
        return 0

    if not args.dry_run:
        os.makedirs(args.output_root, exist_ok=True)

    cfg = {
        "mask_dir": args.mask_dir,
        "reference_root": args.reference_root,
        "output_root": args.output_root,
        "resume": args.resume,
        "dry_run": args.dry_run,
    }

    problems = 0
    with Pool(processes=args.workers, initializer=_init_worker, initargs=(cfg,)) as pool:
        for i, line in enumerate(pool.imap_unordered(process_case, cases), 1):
            print(f"[{i}/{len(cases)}] {line}", flush=True)
            if line.startswith(("MISS", "SKIP")):
                problems += 1

    print("-" * 70)
    print(f"Done. {len(cases) - problems}/{len(cases)} cases produced a fresh mask "
          f"({problems} skipped/missing).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
