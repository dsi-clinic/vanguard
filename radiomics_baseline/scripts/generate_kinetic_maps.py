#!/usr/bin/env python3
"""Generate voxel-wise kinetic parameter maps from DCE-MRI phase images.

Standalone preprocessing step: run BEFORE radiomics extraction.
Produces NIfTI parameter maps that can be added to ``image_patterns``
in the experiment YAML config.

Usage
-----
    python generate_kinetic_maps.py \
        --images /net/projects2/vanguard/MAMA-MIA-syn60868042/images \
        --masks  /net/projects2/vanguard/MAMA-MIA-syn60868042/segmentations/expert \
        --splits /home/summe/vanguard/radiomics_baseline/splits_train_test_ready.csv \
        --mask-pattern "{pid}.nii.gz" \
        --n-jobs 8 \
        --generate-tpeak-voxel \
        --overwrite
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import SimpleITK as sitk
from joblib import Parallel, delayed
from tqdm import tqdm

# Regex for matching phase files — reused from visualize_dce_tumor_bbox.py
_PHASE_RE = re.compile(r"_(\d{4})\.nii(?:\.gz)?$")

# The 5 core kinetic maps generated for every patient.
CORE_MAP_NAMES = ("E_early", "E_peak", "slope_in", "slope_out", "AUC")


# ---------------------------------------------------------------------------
# Phase discovery
# ---------------------------------------------------------------------------

def discover_phases(images_dir: str, pid: str) -> list[tuple[int, Path]]:
    """Discover all available DCE phase files for a patient.

    Returns a sorted list of ``(phase_index, Path)`` tuples.
    Phase 0000 is always pre-contrast.
    """
    patient_dir = Path(images_dir) / pid
    if not patient_dir.is_dir():
        raise FileNotFoundError(f"Patient directory not found: {patient_dir}")

    phases: list[tuple[int, Path]] = []
    for p in sorted(patient_dir.glob(f"{pid}_????.nii.gz")):
        m = _PHASE_RE.search(p.name)
        if m:
            phases.append((int(m.group(1)), p))

    if not phases:
        raise FileNotFoundError(f"No phase files found in {patient_dir}")

    phases.sort(key=lambda x: x[0])
    return phases


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_phase_volumes(
    phase_paths: list[tuple[int, Path]],
) -> tuple[list[int], list[np.ndarray], sitk.Image]:
    """Load all phase NIfTI files.

    Returns
    -------
    indices : list[int]
        Phase indices (e.g. [0, 1, 2, 3]).
    arrays : list[np.ndarray]
        Corresponding float32 volumes in (z, y, x) order.
    reference_img : sitk.Image
        The SimpleITK Image object for phase _0000, used as the
        spatial reference when saving output maps.
    """
    indices: list[int] = []
    arrays: list[np.ndarray] = []
    reference_img: sitk.Image | None = None

    for idx, path in phase_paths:
        img = sitk.ReadImage(str(path))
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        indices.append(idx)
        arrays.append(arr)
        if idx == 0:
            reference_img = img

    if reference_img is None:
        raise ValueError("Pre-contrast phase (_0000) not found among loaded phases.")

    return indices, arrays, reference_img


def save_map_as_nifti(
    arr: np.ndarray,
    reference_img: sitk.Image,
    output_path: Path,
) -> None:
    """Save a 3D numpy array as NIfTI, copying spatial metadata from *reference_img*."""
    out_img = sitk.GetImageFromArray(arr.astype(np.float32))
    out_img.CopyInformation(reference_img)
    sitk.WriteImage(out_img, str(output_path))


def load_mask(masks_dir: str, pid: str, mask_pattern: str) -> np.ndarray:
    """Load a binary mask for a patient, returning a uint8 array."""
    if "{pid}" in mask_pattern:
        rel = mask_pattern.format(pid=pid)
    else:
        rel = mask_pattern
    mask_path = Path(masks_dir) / rel
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    img = sitk.ReadImage(str(mask_path))
    arr = sitk.GetArrayFromImage(img)
    return (arr > 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Kinetic computations
# ---------------------------------------------------------------------------

def compute_enhancement_series(
    post_volumes: list[np.ndarray],
    pre_contrast: np.ndarray,
) -> list[np.ndarray]:
    """Compute E_t = I_t - I_0 for each post-contrast volume."""
    return [vol - pre_contrast for vol in post_volumes]


def find_tumor_peak_phase(
    enhancement_series: list[np.ndarray],
    phase_indices: list[int],
    mask: np.ndarray,
) -> int:
    """Determine tumor-level peak phase index.

    Returns the *list index* (into enhancement_series / phase_indices)
    of the phase with the highest median enhancement within the mask.
    """
    best_idx = 0
    best_median = -np.inf
    for i, E_t in enumerate(enhancement_series):
        masked_vals = E_t[mask > 0]
        med = float(np.median(masked_vals)) if masked_vals.size > 0 else 0.0
        if med > best_median:
            best_median = med
            best_idx = i
    return best_idx


def compute_kinetic_maps(
    pre_contrast: np.ndarray,
    post_volumes: list[np.ndarray],
    post_indices: list[int],
    mask: np.ndarray,
    generate_tpeak_voxel: bool = False,
) -> dict[str, np.ndarray]:
    """Compute all kinetic parameter maps.

    Parameters
    ----------
    pre_contrast : 3D array for phase _0000
    post_volumes : list of 3D arrays for post-contrast phases
    post_indices : list of integer phase indices (1, 2, 3, …)
    mask : binary 3D array of the tumor ROI
    generate_tpeak_voxel : generate voxel-wise time-to-peak (requires ≥4 phases)

    Returns
    -------
    dict mapping map name → 3D numpy array.
    """
    maps: dict[str, np.ndarray] = {}

    # Enhancement series
    enhancements = compute_enhancement_series(post_volumes, pre_contrast)

    # E_early: first post-contrast enhancement
    E_early = enhancements[0].copy()

    # Tumor-level peak phase
    peak_list_idx = find_tumor_peak_phase(enhancements, post_indices, mask)
    E_peak = enhancements[peak_list_idx].copy()

    # slope_in = E_early / Δt_early
    t_early = post_indices[0]  # typically 1
    slope_in = E_early / max(t_early, 1)

    # slope_out = (I_last - I_{t_peak}) / (t_last - t_peak)
    t_peak = post_indices[peak_list_idx]
    t_last = post_indices[-1]
    if t_last > t_peak:
        slope_out = (post_volumes[-1] - post_volumes[peak_list_idx]) / (t_last - t_peak)
    else:
        slope_out = np.zeros_like(pre_contrast, dtype=np.float32)

    # AUC: trapezoidal integration including t=0 with E=0
    all_times = [0] + list(post_indices)
    all_enhancements = [np.zeros_like(pre_contrast, dtype=np.float64)] + [
        e.astype(np.float64) for e in enhancements
    ]
    auc = np.zeros_like(pre_contrast, dtype=np.float64)
    for i in range(1, len(all_times)):
        dt = all_times[i] - all_times[i - 1]
        auc += 0.5 * dt * (all_enhancements[i - 1] + all_enhancements[i])

    maps["E_early"] = E_early
    maps["E_peak"] = E_peak
    maps["slope_in"] = slope_in
    maps["slope_out"] = slope_out
    maps["AUC"] = auc.astype(np.float32)

    # Optional: voxel-wise time-to-peak (only if ≥4 post-contrast phases)
    if generate_tpeak_voxel and len(post_volumes) >= 4:
        enhancement_stack = np.stack(enhancements, axis=0)  # (n_phases, z, y, x)
        peak_voxel_list_idx = np.argmax(enhancement_stack, axis=0)  # index into list
        # Map list index → actual phase index
        t_peak_voxel = np.zeros_like(peak_voxel_list_idx, dtype=np.float32)
        for i, t in enumerate(post_indices):
            t_peak_voxel[peak_voxel_list_idx == i] = t
        maps["t_peak_voxel"] = t_peak_voxel

    return maps


def sanitize_map(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace NaN/inf with 0, zero values outside the ROI mask."""
    out = arr.copy()
    out[~np.isfinite(out)] = 0.0
    out[mask == 0] = 0.0
    return out


# ---------------------------------------------------------------------------
# Per-patient orchestrator
# ---------------------------------------------------------------------------

def generate_maps_for_pid(
    pid: str,
    images_dir: str,
    masks_dir: str,
    mask_pattern: str,
    min_post_contrast: int = 1,
    generate_tpeak_voxel: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate and save all kinetic parameter maps for one patient.

    Saves maps as ``{images_dir}/{pid}/{pid}_kinetic_{name}.nii.gz``.

    Returns a summary dict with pid, n_phases, maps_generated, status.
    """
    result: dict[str, Any] = {"patient_id": pid, "status": "success", "error": ""}

    try:
        # Check if already generated (skip unless overwrite)
        patient_dir = Path(images_dir) / pid
        if not overwrite:
            existing = [
                name
                for name in CORE_MAP_NAMES
                if (patient_dir / f"{pid}_kinetic_{name}.nii.gz").exists()
            ]
            if len(existing) == len(CORE_MAP_NAMES):
                result["status"] = "skipped"
                result["n_phases"] = -1
                result["maps_generated"] = 0
                return result

        # Discover phases
        phases = discover_phases(images_dir, pid)
        result["n_phases"] = len(phases)

        # Need at least _0000 + min_post_contrast post-contrast phases
        pre_phases = [(i, p) for i, p in phases if i == 0]
        post_phases = [(i, p) for i, p in phases if i > 0]

        if not pre_phases:
            raise ValueError(f"No pre-contrast (_0000) phase for {pid}")
        if len(post_phases) < min_post_contrast:
            raise ValueError(
                f"{pid} has {len(post_phases)} post-contrast phases, "
                f"need at least {min_post_contrast}"
            )

        # Load volumes
        all_indices, all_arrays, ref_img = load_phase_volumes(phases)

        # Separate pre-contrast from post-contrast
        pre_idx = all_indices.index(0)
        pre_contrast = all_arrays[pre_idx]

        post_indices = [idx for idx in all_indices if idx > 0]
        post_volumes = [all_arrays[all_indices.index(idx)] for idx in post_indices]

        # Load mask
        mask = load_mask(masks_dir, pid, mask_pattern)

        # Verify shape compatibility
        if mask.shape != pre_contrast.shape:
            raise ValueError(
                f"Shape mismatch for {pid}: mask {mask.shape} vs "
                f"pre-contrast {pre_contrast.shape}"
            )

        # Compute kinetic maps
        kinetic_maps = compute_kinetic_maps(
            pre_contrast=pre_contrast,
            post_volumes=post_volumes,
            post_indices=post_indices,
            mask=mask,
            generate_tpeak_voxel=generate_tpeak_voxel,
        )

        # Sanitize and save
        maps_saved = 0
        for name, arr in kinetic_maps.items():
            clean = sanitize_map(arr, mask)
            out_path = patient_dir / f"{pid}_kinetic_{name}.nii.gz"
            save_map_as_nifti(clean, ref_img, out_path)
            maps_saved += 1

        result["maps_generated"] = maps_saved

    except Exception as exc:  # noqa: BLE001
        result["status"] = "error"
        result["error"] = str(exc)
        result["n_phases"] = result.get("n_phases", 0)
        result["maps_generated"] = 0
        warnings.warn(f"[{pid}] {exc}", stacklevel=2)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description="Generate kinetic parameter maps from DCE-MRI phases.",
    )
    ap.add_argument(
        "--images",
        required=True,
        type=str,
        help="Root directory containing patient image subdirectories.",
    )
    ap.add_argument(
        "--masks",
        required=True,
        type=str,
        help="Root directory containing patient mask files.",
    )
    ap.add_argument(
        "--splits",
        required=True,
        type=str,
        help="CSV with patient_id and split columns.",
    )
    ap.add_argument(
        "--mask-pattern",
        default="{pid}.nii.gz",
        type=str,
        help="Mask filename pattern (default: '{pid}.nii.gz').",
    )
    ap.add_argument(
        "--n-jobs",
        default=1,
        type=int,
        help="Number of parallel workers (default: 1).",
    )
    ap.add_argument(
        "--generate-tpeak-voxel",
        action="store_true",
        help="Generate voxel-wise time-to-peak map (requires >=4 post-contrast phases).",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing kinetic maps.",
    )
    ap.add_argument(
        "--summary-csv",
        type=str,
        default=None,
        help="Path to write summary CSV (default: {images}/kinetic_maps_summary.csv).",
    )
    args = ap.parse_args()

    # Load patient list from splits
    splits = pd.read_csv(args.splits)
    pids = sorted(splits["patient_id"].unique().tolist())
    print(f"[KINETIC] {len(pids)} patients from {args.splits}")

    # Generate maps in parallel
    results = Parallel(n_jobs=args.n_jobs, backend="loky")(
        delayed(generate_maps_for_pid)(
            pid=pid,
            images_dir=args.images,
            masks_dir=args.masks,
            mask_pattern=args.mask_pattern,
            generate_tpeak_voxel=args.generate_tpeak_voxel,
            overwrite=args.overwrite,
        )
        for pid in tqdm(pids, desc="Generating kinetic maps")
    )

    # Summarize
    df = pd.DataFrame(results)
    n_ok = (df["status"] == "success").sum()
    n_skip = (df["status"] == "skipped").sum()
    n_err = (df["status"] == "error").sum()
    print(f"\n[KINETIC] Done: {n_ok} success, {n_skip} skipped, {n_err} errors")

    if n_err > 0:
        print("\n[KINETIC] Errors:")
        for _, row in df[df["status"] == "error"].iterrows():
            print(f"  {row['patient_id']}: {row['error']}")

    # Save summary
    summary_path = args.summary_csv or str(
        Path(args.images) / "kinetic_maps_summary.csv"
    )
    df.to_csv(summary_path, index=False)
    print(f"[KINETIC] Summary written to {summary_path}")


if __name__ == "__main__":
    main()
