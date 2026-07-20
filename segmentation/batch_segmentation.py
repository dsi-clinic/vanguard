#!/usr/bin/env python3
"""Batch segmentation driver: in-process, batched, AMP inference.

For the vessel-segmentation stage (breast mask STEP-2 -> vessel STEP-3):

1. STEP-1 preprocessing runs in parallel (``ThreadPoolExecutor``).
2. STEP-2 (breast) and STEP-3 (vessel) run **in-process** -- no subprocess,
   each model is loaded exactly once and kept on the GPU (no Python restart,
   no reload).
3. Inference is **batched** (``predict_fast``) and uses **AMP** on the GPU.

This replaced an earlier subprocess-per-stage implementation that shelled out
to the submodule's ``predict.py`` twice per file. That implementation was
validated against this one (16.15x mean speedup, Dice 0.9998-1.0 on a 7-file
sample) before removal -- see ``validation_results.md``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import SimpleITK as sitk
import torch

from preprocessing.model import frozen_model_intensity_preprocess

if TYPE_CHECKING:
    from cohorts.base import DatasetAdapter

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_SUBMODULE = _PROJECT_ROOT / "vanguard-blood-vessel-segmentation"
for _p in (str(_HERE), str(_SUBMODULE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import predict_fast  # noqa: E402


def find_nii_files(images_dir: str) -> list[tuple[str, str]]:
    """Find all .nii.gz files in the images directory.

    Args:
        images_dir: Path to the images directory

    Returns:
        List of tuples (case_id, file_path)
    """
    nii_files = []

    # Get all case directories
    patient_dirs = [d.name for d in Path(images_dir).iterdir() if d.is_dir()]

    for case_id in patient_dirs:
        patient_path = Path(images_dir) / case_id

        # Find all .nii.gz files in this case directory
        files = list(patient_path.glob("*.nii.gz"))

        for file_path in files:
            nii_files.append((case_id, file_path))

    return nii_files


def find_case_files(
    adapter: DatasetAdapter,
    case_limit: int | None = None,
) -> list[tuple[str, str]]:
    """Enumerate (case_id, file_path) pairs to preprocess.

    Discovery routes through ``adapter.discover_cases()`` /
    ``adapter.load_timepoints(case_id)`` instead of assuming the MAMA-MIA
    ``<images_dir>/<case_id>/*.nii.gz`` layout -- e.g. UChicago's
    manifest-driven, sub-source-partitioned images would otherwise be silently
    missed or misread by ``find_nii_files``. The adapter owns its own root, so
    there is no separate ``images_dir`` argument here.

    ``case_limit`` (applied here, at case granularity, before file expansion)
    exists because the CLI's ``--patient-limit`` slices the flat file list --
    imprecise for a case with more than one phase file.
    """
    case_ids = adapter.discover_cases()
    if case_limit is not None:
        case_ids = case_ids[:case_limit]
    pairs: list[tuple[str, str]] = []
    for case_id in case_ids:
        for phase_path in adapter.load_timepoints(case_id):
            pairs.append((case_id, str(phase_path)))
    return pairs


def _base_name_for(case_id: str, file_path: str) -> str:
    """Derive the unique intermediate-file base name for one (case, phase) file.

    UChicago's phase files are named identically across *every* case
    (``phase_0000.nii.gz`` for every exam); without a case-id prefix, different
    patients' same-numbered phase would collide on the same path in the flat
    step1/step2/step3 intermediate directories, silently overwriting each
    other's preprocessed output. Prefixing with ``case_id`` avoids that
    collision -- :func:`build_output_path` already expects and strips exactly
    this prefix when recovering the timepoint for the final output filename.

    MAMA-MIA filenames already embed the case id (``DUKE_001_0000.nii.gz``), so
    prefixing again would double it up (``DUKE_001_DUKE_001_0000``). Only
    prefix when the stem doesn't already start with ``case_id``, so both
    shapes come out correct with the adapter always present (Step 5).
    """
    raw = Path(file_path).name.replace(".nii.gz", "")
    if raw == case_id or raw.startswith(f"{case_id}_"):
        return raw
    return f"{case_id}_{raw}"


def preprocess_image(
    input_path: str, output_path: str, adapter: DatasetAdapter
) -> bool:
    """Preprocess a single .nii.gz file (STEP-1).

    Args:
        input_path: Path to input .nii.gz file
        output_path: Path to save preprocessed .npy file
        adapter: Dataset adapter (see cohorts/README.md). The geometry
            reorientation is taken from ``adapter.preprocess`` -- e.g.
            ``MamaMiaDataset`` encodes the historical MAMA-MIA axis transform,
            UChicago ships already-oriented data and overrides this to a
            pass-through. Intensity normalization is *not* the adapter's job:
            this stage always applies the shared
            ``preprocessing.model.frozen_model_intensity_preprocess`` contract,
            matching cohorts/base.py.

    Returns:
        True if successful, False otherwise
    """
    try:
        # Load the image
        original_array = sitk.GetArrayFromImage(sitk.ReadImage(str(input_path)))

        # Two separable jobs, kept separate: the adapter owns the *spatial*
        # reorientation, and the *intensity* contract is always the shared
        # frozen-model one from preprocessing.model -- never a local
        # re-implementation, so segmentation can't drift from what the pinned
        # model was trained against.
        reoriented = adapter.preprocess(original_array)
        preprocessed_array = frozen_model_intensity_preprocess(reoriented)

        # Save as .npy
        np.save(output_path, preprocessed_array)
        return True

    except Exception as e:
        print(f"Error preprocessing {input_path}: {e}")
        return False


def build_output_path(
    output_dir: Path,
    case_id: str,
    base_name: str,
    adapter: DatasetAdapter,
) -> Path:
    """Build output path in a source/case/images layout.

    The top-level ``source`` directory is the case's dataset identity, from
    ``adapter.case_dataset_name`` (one authoritative answer -- e.g. UChicago's
    manifest sub-source, or MAMA-MIA's case-id prefix).
    """
    source = adapter.case_dataset_name(case_id)
    timepoint = (
        base_name[len(case_id) + 1 :]
        if base_name.startswith(f"{case_id}_")
        else base_name
    )
    filename = (
        f"{case_id}_{timepoint}_vessel_segmentation.npz"
        if timepoint
        else f"{case_id}_vessel_segmentation.npz"
    )
    output_subdir = output_dir / source / case_id / "images"
    output_subdir.mkdir(parents=True, exist_ok=True)
    return output_subdir / filename


def collect_all_step3_files(output_dir: str) -> list[str]:
    """Collect all STEP-3 vessel segmentation .npz files from the output directory.

    Args:
        output_dir: Directory containing the processed files

    Returns:
        List of paths to all vessel segmentation .npz files
    """
    npy_files = []
    for root, _dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith(".npz") and "vessel_segmentation" in file:
                npy_files.append(Path(root) / file)
    return npy_files


def preprocess_parallel(
    file_list: list[tuple[str, str]],
    step1_dir: Path,
    workers: int,
    adapter: DatasetAdapter,
) -> tuple[dict[str, str], list[str]]:
    """Run STEP-1 preprocessing across a thread pool. Returns base_name->case_id."""
    base_name_to_case = {}
    failed = []

    def _work(item: tuple[str, str]) -> tuple[str, str, bool]:
        case_id, file_path = item
        base_name = _base_name_for(case_id, file_path)
        step1_file = step1_dir / f"{base_name}.npy"
        ok = preprocess_image(file_path, step1_file, adapter=adapter)
        return case_id, base_name, ok

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for case_id, base_name, ok in ex.map(_work, file_list):
            if ok:
                base_name_to_case[base_name] = case_id
            else:
                failed.append(case_id)
                print(f"  ✗ preprocessing failed: {case_id}")
    return base_name_to_case, failed


def run_inference_in_process(
    step1_dir: Path,
    step2_dir: Path,
    step3_dir: Path,
    breast_model_path: str,
    vessel_model_path: str,
    batch_size: int,
    num_workers: int,
    use_amp: bool,
) -> None:
    """Load each model once and run breast then vessel inference in-process."""
    import torchio as tio
    from dataset_3d import Dataset3DDivided, Dataset3DSimple

    predict_fast.apply_shape_patch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── STEP-2: breast ────────────────────────────────────────────────────
    breast_unet, _, _ = predict_fast.build_unet("breast")
    breast_unet = predict_fast.load_model(breast_unet, breast_model_path, device)
    breast_ds = Dataset3DSimple(
        image_dir=str(step1_dir),
        mask_dir=None,
        transforms=tio.Compose([tio.Resize((144, 144, 96))]),
        image_only=True,
    )
    predict_fast.predict_breast_batched(
        breast_unet,
        breast_ds,
        str(step2_dir),
        device=device,
        batch_size=max(2, batch_size // 2),
        num_workers=num_workers,
        use_amp=use_amp,
    )
    del breast_unet
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── STEP-3: vessel ────────────────────────────────────────────────────
    vessel_unet, _, n_classes = predict_fast.build_unet("dv")
    vessel_unet = predict_fast.load_model(vessel_unet, vessel_model_path, device)
    vessel_ds = Dataset3DDivided(
        image_dir=str(step1_dir),
        mask_dir=None,
        additional_input_dir=str(step2_dir),
        input_dim=96,
        x_y_divisions=8,
        z_division=3,
        transforms=tio.Compose([]),
        one_hot_mask=True,
        image_only=True,
    )
    predict_fast.predict_vessel_batched(
        vessel_unet,
        vessel_ds,
        n_classes=n_classes,
        save_masks_dir=str(step3_dir),
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        use_amp=use_amp,
    )


def main() -> None:
    """Run the batch-segmentation CLI."""
    p = argparse.ArgumentParser(
        description="In-process batched vessel segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir", default=str(_PROJECT_ROOT / "vessel_segmentations"))
    p.add_argument(
        "--temp-dir", required=True, help="Scratch dir for STEP-1/2/3 intermediates"
    )
    p.add_argument(
        "--breast-model-path",
        default=str(_SUBMODULE / "trained_models" / "breast_model.pth"),
    )
    p.add_argument(
        "--vessel-model-path",
        default=str(_SUBMODULE / "trained_models" / "dv_model.pth"),
    )
    p.add_argument("--patient-limit", type=int, default=None)
    p.add_argument("--file-start", type=int, default=None)
    p.add_argument("--file-end", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=3)
    p.add_argument("--preprocess-workers", type=int, default=4)
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--cleanup", action="store_true")
    p.add_argument(
        "--dataset-name",
        required=True,
        help=(
            "Build a DatasetAdapter via cohorts.factory and route "
            "discovery/preprocessing/output-path through it (currently "
            "'mamamia' only -- 'uchicago' is rejected here: its imaging route "
            "is the paired raw-DICOM pipeline, see preprocessing/README.md)."
        ),
    )
    p.add_argument(
        "--dataset-root",
        required=True,
        help="Root path for --dataset-name (e.g. the MAMA-MIA data root).",
    )
    p.add_argument(
        "--dataset-cohort",
        default=None,
        help="mamamia only: duke|ispy1|ispy2|nact, or omit for all four combined.",
    )
    args = p.parse_args()

    from cohorts.factory import require_imaging_adapter_from_config
    from config import ConfigNode

    dataset_config = ConfigNode._wrap(
        {
            "dataset": {
                "name": args.dataset_name,
                "cohort": args.dataset_cohort,
                "root": args.dataset_root,
                "split_policy": "auto",
            }
        }
    )
    adapter = require_imaging_adapter_from_config(dataset_config)
    print(f"Using dataset adapter: {type(adapter).__name__}")

    out_dir = Path(args.output_dir)
    temp_dir = Path(args.temp_dir)
    step1_dir, step2_dir, step3_dir = (
        temp_dir / "step1",
        temp_dir / "step2",
        temp_dir / "step3",
    )
    for d in (out_dir, step1_dir, step2_dir, step3_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Discovering cases via {type(adapter).__name__}...")
    # Case-level limit applied before file expansion (see find_case_files):
    # a file-list-level limit would be imprecise once a case has more than
    # one phase file.
    nii_files = sorted(
        find_case_files(adapter, case_limit=args.patient_limit),
        key=lambda x: (x[0], str(x[1])),
    )
    print(f"Found {len(nii_files)} file(s) across the selected cases")

    if args.file_start is not None:
        end = args.file_end if args.file_end is not None else args.file_start
        end = min(end, len(nii_files) - 1)
        nii_files = nii_files[args.file_start : end + 1]
        print(f"Range {args.file_start}-{end}: {len(nii_files)} files")

    if args.resume:
        before = len(nii_files)
        nii_files = [
            (cid, fp)
            for cid, fp in nii_files
            if not build_output_path(
                out_dir, cid, _base_name_for(cid, fp), adapter=adapter
            ).exists()
        ]
        print(f"Resume: {len(nii_files)} remaining (skipped {before - len(nii_files)})")
    if not nii_files:
        print("Nothing to do, exiting.")
        return

    t0 = time.time()
    use_amp = not args.no_amp
    print(
        f"Preprocessing {len(nii_files)} file(s) with {args.preprocess_workers} workers..."
    )
    base_name_to_case, failed = preprocess_parallel(
        nii_files, step1_dir, args.preprocess_workers, adapter=adapter
    )
    t_pre = time.time()
    print(
        f"Preprocessing done in {t_pre - t0:.1f}s ({len(base_name_to_case)} ok, {len(failed)} failed)"
    )
    if not base_name_to_case:
        print("All files failed preprocessing — skipping inference.")
        return

    print(
        f"Inference (batch_size={args.batch_size}, num_workers={args.num_workers}, amp={use_amp})..."
    )
    run_inference_in_process(
        step1_dir,
        step2_dir,
        step3_dir,
        args.breast_model_path,
        args.vessel_model_path,
        args.batch_size,
        args.num_workers,
        use_amp,
    )
    t_inf = time.time()
    print(f"Inference done in {t_inf - t_pre:.1f}s")

    # ── Move STEP-3 outputs to final layout ───────────────────────────────
    successful, failed_cases = [], list(failed)
    for base_name, case_id in base_name_to_case.items():
        step3_file = step3_dir / f"{base_name}.npz"
        if step3_file.exists():
            dst = build_output_path(out_dir, case_id, base_name, adapter=adapter)
            shutil.move(str(step3_file), str(dst))
            successful.append(str(dst))
            print(f"  ✓ {case_id}: {dst}")
        else:
            failed_cases.append(case_id)
            print(f"  ✗ vessel output missing: {case_id} ({base_name}.npz)")

    total = time.time() - t0
    print(f"\n{'=' * 60}\nBATCH SEGMENTATION COMPLETE")
    print(
        f"Files: {len(nii_files)}  Successful: {len(successful)}  Failed: {len(failed_cases)}"
    )
    print(f"Total: {total:.1f}s  ({total / len(nii_files):.1f}s/file)")
    print(f"  preprocess: {t_pre - t0:.1f}s  inference: {t_inf - t_pre:.1f}s")
    print(f"Outputs collected: {len(collect_all_step3_files(str(out_dir)))}")

    if args.cleanup:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
