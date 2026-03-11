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
        --splits splits_train_test_ready.csv \
        --output-dir kinetic_maps \
        --mask-pattern "{pid}.nii.gz" \
        --n-jobs 8 \
        --generate-tpeak-voxel \
        --overwrite
"""

from __future__ import annotations

import argparse
import re
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

# The 5 core kinetic parameter maps generated for every patient.
KINETIC_MAP_NAMES = ("E_early", "E_peak", "slope_in", "slope_out", "AUC")

# Subtraction images generated when --generate-subtraction is requested.
# wash_in  = I_peak − I_early  (incremental late enhancement; distinct from E_peak)
# wash_out = I_last − I_peak   (signed: negative = washout, positive = persistent)
SUBTRACTION_MAP_NAMES = ("wash_in", "wash_out")
_MIN_TPEAK_PHASES = 4
_MIN_SUBTRACTION_PHASES = 2


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

    Returns:
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
    peak_list_idx: int | None = None,
) -> dict[str, np.ndarray]:
    """Compute all kinetic parameter maps.

    Parameters
    ----------
    pre_contrast : 3D array for phase _0000
    post_volumes : list of 3D arrays for post-contrast phases
    post_indices : list of integer phase indices (1, 2, 3, …)
    mask : binary 3D array of the tumor ROI
    generate_tpeak_voxel : generate voxel-wise time-to-peak (requires ≥4 phases)
    peak_list_idx : pre-computed tumor peak list index; auto-computed if None.
        Pass when sharing peak detection with ``compute_subtraction_maps``.

    Returns:
    -------
    dict mapping map name → 3D numpy array.
    """
    maps: dict[str, np.ndarray] = {}

    # Enhancement series
    enhancements = compute_enhancement_series(post_volumes, pre_contrast)

    # E_early: first post-contrast enhancement
    E_early = enhancements[0].copy()

    # Tumor-level peak phase (use pre-computed index if provided)
    if peak_list_idx is None:
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
    if generate_tpeak_voxel and len(post_volumes) >= _MIN_TPEAK_PHASES:
        enhancement_stack = np.stack(enhancements, axis=0)  # (n_phases, z, y, x)
        peak_voxel_list_idx = np.argmax(enhancement_stack, axis=0)  # index into list
        # Map list index → actual phase index
        t_peak_voxel = np.zeros_like(peak_voxel_list_idx, dtype=np.float32)
        for i, t in enumerate(post_indices):
            t_peak_voxel[peak_voxel_list_idx == i] = t
        maps["t_peak_voxel"] = t_peak_voxel

    return maps


def compute_subtraction_maps(
    post_volumes: list[np.ndarray],
    peak_list_idx: int,
) -> dict[str, np.ndarray]:
    """Compute subtraction images following Braman et al. convention.

    Parameters
    ----------
    post_volumes : list of 3D post-contrast arrays ordered by acquisition time.
        Index 0 is the first post-contrast phase (_0001).
    peak_list_idx : list index of the tumor-level peak phase, as returned by
        ``find_tumor_peak_phase``.

    Returns:
    -------
    dict with keys:

    ``wash_in``
        ``I_peak − I_early``: incremental enhancement from the first
        post-contrast phase to the peak.  Captures delayed kinetics and is
        distinct from ``E_peak`` (which uses pre-contrast as baseline).
        Zero when the peak coincides with the first post-contrast phase.
    ``wash_out``
        ``I_last − I_peak``: signed change after the peak.  Negative values
        indicate washout (BI-RADS type III kinetics); positive values indicate
        a persistent pattern (type I).
    """
    wash_in = (post_volumes[peak_list_idx] - post_volumes[0]).astype(np.float32)
    wash_out = (post_volumes[-1] - post_volumes[peak_list_idx]).astype(np.float32)
    return {"wash_in": wash_in, "wash_out": wash_out}


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
    output_dir: str | None = None,
    min_post_contrast: int = 1,
    fixed_post_phase_indices: list[int] | None = None,
    generate_tpeak_voxel: bool = False,
    generate_subtraction: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate and save all kinetic parameter maps for one patient.

    Saves maps as ``{output_dir}/{pid}/{pid}_kinetic_{name}.nii.gz``.
    If *output_dir* is ``None``, falls back to *images_dir*.

    Returns a summary dict with pid, n_phases, maps_generated, status.
    """
    result: dict[str, Any] = {"patient_id": pid, "status": "success", "error": ""}

    try:
        # Determine where to write maps
        out_root = Path(output_dir) if output_dir else Path(images_dir)
        patient_out_dir = out_root / pid
        patient_out_dir.mkdir(parents=True, exist_ok=True)

        # Check if already generated (skip unless overwrite)
        if not overwrite:
            expected_kinetic_map_names = list(KINETIC_MAP_NAMES)
            if generate_tpeak_voxel:
                expected_kinetic_map_names.append("t_peak_voxel")
            kinetic_done = all(
                (patient_out_dir / f"{pid}_kinetic_{name}.nii.gz").exists()
                for name in expected_kinetic_map_names
            )
            sub_done = (not generate_subtraction) or all(
                (patient_out_dir / f"{pid}_subtraction_{name}.nii.gz").exists()
                for name in SUBTRACTION_MAP_NAMES
            )
            if kinetic_done and sub_done:
                result["status"] = "skipped"
                result["n_phases"] = -1
                result["maps_generated"] = 0
                return result

        # Discover phases
        phases = discover_phases(images_dir, pid)
        result["n_phases"] = len(phases)

        # Select the phase set used for map computation.
        # If fixed indices are provided, kinetics are computed only from those
        # post-contrast timepoints to avoid confounding from variable phase count.
        idx_to_path = dict(phases)
        if 0 not in idx_to_path:
            raise ValueError(f"No pre-contrast (_0000) phase for {pid}")

        if fixed_post_phase_indices:
            required = [0] + fixed_post_phase_indices
            missing = [idx for idx in required if idx not in idx_to_path]
            if missing:
                raise ValueError(
                    f"{pid} missing required fixed phases: {missing} "
                    f"(requested post phases={fixed_post_phase_indices})"
                )
            phases_to_use = [(idx, idx_to_path[idx]) for idx in required]
        else:
            post_phases = [(i, p) for i, p in phases if i > 0]
            if len(post_phases) < min_post_contrast:
                raise ValueError(
                    f"{pid} has {len(post_phases)} post-contrast phases, "
                    f"need at least {min_post_contrast}"
                )
            phases_to_use = phases

        result["n_phases_used"] = len(phases_to_use)

        # Load volumes
        all_indices, all_arrays, ref_img = load_phase_volumes(phases_to_use)

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

        # Pre-compute enhancement series and tumor peak once; shared across maps.
        enhancements = compute_enhancement_series(post_volumes, pre_contrast)
        peak_list_idx = find_tumor_peak_phase(enhancements, post_indices, mask)

        # Compute and save kinetic parameter maps.
        kinetic_maps = compute_kinetic_maps(
            pre_contrast=pre_contrast,
            post_volumes=post_volumes,
            post_indices=post_indices,
            mask=mask,
            generate_tpeak_voxel=generate_tpeak_voxel,
            peak_list_idx=peak_list_idx,
        )

        maps_saved = 0
        for name, arr in kinetic_maps.items():
            clean = sanitize_map(arr, mask)
            out_path = patient_out_dir / f"{pid}_kinetic_{name}.nii.gz"
            save_map_as_nifti(clean, ref_img, out_path)
            maps_saved += 1

        # Compute and save subtraction images (Braman et al. convention).
        if generate_subtraction:
            if len(post_volumes) < _MIN_SUBTRACTION_PHASES:
                warnings.warn(
                    f"[{pid}] Subtraction maps require ≥2 post-contrast phases "
                    f"({len(post_volumes)} found); skipping subtraction.",
                    stacklevel=2,
                )
            else:
                sub_maps = compute_subtraction_maps(post_volumes, peak_list_idx)
                for name, arr in sub_maps.items():
                    clean = sanitize_map(arr, mask)
                    out_path = patient_out_dir / f"{pid}_subtraction_{name}.nii.gz"
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
        "--fixed-post-phase-indices",
        type=str,
        default=None,
        help=(
            "Optional comma-separated post-contrast phase indices (e.g. '1,2,3'). "
            "If set, kinetic and subtraction maps are computed only from these "
            "timepoints plus pre-contrast phase 0000."
        ),
    )
    ap.add_argument(
        "--generate-tpeak-voxel",
        action="store_true",
        help=(
            "Generate voxel-wise time-to-peak map"
            " (requires >=4 post-contrast phases)."
        ),
    )
    ap.add_argument(
        "--generate-subtraction",
        action="store_true",
        help=(
            "Generate wash_in and wash_out subtraction images following Braman et al. "
            "wash_in = I_peak - I_early; wash_out = I_last - I_peak. "
            "Requires >=2 post-contrast phases."
        ),
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write kinetic maps (default: same as --images).",
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
        help=(
            "Path to write summary CSV. Default: "
            "{output_dir}/kinetic_maps_summary.csv when --output-dir is set, "
            "otherwise {images}/kinetic_maps_summary.csv."
        ),
    )
    args = ap.parse_args()

    fixed_post_phase_indices: list[int] | None = None
    if args.fixed_post_phase_indices:
        parsed: list[int] = []
        for tok in args.fixed_post_phase_indices.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                idx = int(tok)
            except ValueError as exc:
                raise ValueError(
                    "--fixed-post-phase-indices must be comma-separated integers"
                ) from exc
            if idx <= 0:
                raise ValueError(
                    "--fixed-post-phase-indices must contain positive phase indices"
                )
            parsed.append(idx)
        if not parsed:
            raise ValueError("--fixed-post-phase-indices parsed to an empty list")
        if len(set(parsed)) != len(parsed):
            raise ValueError("--fixed-post-phase-indices contains duplicates")
        fixed_post_phase_indices = parsed
        print(
            "[KINETIC] fixed post phases enabled: "
            f"{fixed_post_phase_indices} (plus pre-contrast 0000)"
        )

    # Load patient list from splits
    splits = pd.read_csv(args.splits)
    pids = sorted({str(pid) for pid in splits["patient_id"].dropna().tolist()})
    print(f"[KINETIC] {len(pids)} patients from {args.splits}")

    # Generate maps in parallel
    results = Parallel(n_jobs=args.n_jobs, backend="loky")(
        delayed(generate_maps_for_pid)(
            pid=pid,
            images_dir=args.images,
            masks_dir=args.masks,
            mask_pattern=args.mask_pattern,
            output_dir=args.output_dir,
            fixed_post_phase_indices=fixed_post_phase_indices,
            generate_tpeak_voxel=args.generate_tpeak_voxel,
            generate_subtraction=args.generate_subtraction,
            overwrite=args.overwrite,
        )
        for pid in tqdm(pids, desc="Generating kinetic maps")
    )

    # Summarize
    results_df = pd.DataFrame(results)
    n_ok = (results_df["status"] == "success").sum()
    n_skip = (results_df["status"] == "skipped").sum()
    n_err = (results_df["status"] == "error").sum()
    print(f"\n[KINETIC] Done: {n_ok} success, {n_skip} skipped, {n_err} errors")

    if n_err > 0:
        print("\n[KINETIC] Errors:")
        for _, row in results_df[results_df["status"] == "error"].iterrows():
            print(f"  {row['patient_id']}: {row['error']}")

    # Save summary
    summary_root = Path(args.output_dir) if args.output_dir else Path(args.images)
    summary_path = args.summary_csv or str(summary_root / "kinetic_maps_summary.csv")
    results_df.to_csv(summary_path, index=False)
    print(f"[KINETIC] Summary written to {summary_path}")


if __name__ == "__main__":
    main()
