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

import numpy as np
import SimpleITK as sitk
import torch

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_SUBMODULE = _PROJECT_ROOT / "vanguard-blood-vessel-segmentation"
for _p in (str(_HERE), str(_SUBMODULE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# FIXME: this is cobbling together an installation process in the script itself
# packages must be installed in an environment
try:
    from preprocessing import normalize_image, zscore_image  # noqa: E402

except ImportError:

    def normalize_image(*args, **_kwargs):  # noqa: ANN201, D103
        raise ImportError("Required preprocessing function not found")  # noqa: F821

    def zscore_image(*args, **_kwargs):  # noqa: ANN201, D103
        raise ImportError("Required preprocessing function not found")  # noqa: F821


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


def preprocess_image(input_path: str, output_path: str) -> bool:
    """Preprocess a single .nii.gz file (STEP-1).

    Args:
        input_path: Path to input .nii.gz file
        output_path: Path to save preprocessed .npy file

    Returns:
        True if successful, False otherwise
    """
    try:
        # Load the image
        original_array = sitk.GetArrayFromImage(sitk.ReadImage(str(input_path)))

        # Preprocess: rotate axes and normalize
        preprocessed_array = zscore_image(
            normalize_image(np.swapaxes(np.swapaxes(original_array, 0, 2), 0, 1)[::-1])
        )

        # Save as .npy
        np.save(output_path, preprocessed_array)
        return True

    except Exception as e:
        print(f"Error preprocessing {input_path}: {e}")
        return False


def build_output_path(output_dir: Path, case_id: str, base_name: str) -> Path:
    """Build output path in a source/case/images layout."""
    source = case_id.split("_")[0]
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
    file_list: list[tuple[str, str]], step1_dir: Path, workers: int
) -> tuple[dict[str, str], list[str]]:
    """Run STEP-1 preprocessing across a thread pool. Returns base_name->case_id."""
    base_name_to_case = {}
    failed = []

    def _work(item: tuple[str, str]) -> tuple[str, str, bool]:
        case_id, file_path = item
        base_name = Path(file_path).name.replace(".nii.gz", "")
        step1_file = step1_dir / f"{base_name}.npy"
        ok = preprocess_image(file_path, step1_file)
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
    p.add_argument(
        "--images-dir",
        default=os.environ.get(
            "IMAGES_DIR", "/gpfs/data/karczmar-lab/MAMA-MIA-syn60868042/images"
        ),
        help="Directory containing case subdirectories with .nii.gz files",
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
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    temp_dir = Path(args.temp_dir)
    step1_dir, step2_dir, step3_dir = (
        temp_dir / "step1",
        temp_dir / "step2",
        temp_dir / "step3",
    )
    for d in (out_dir, step1_dir, step2_dir, step3_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Finding .nii.gz files in {args.images_dir}...")
    nii_files = sorted(find_nii_files(args.images_dir), key=lambda x: (x[0], str(x[1])))
    print(f"Found {len(nii_files)} .nii.gz files")

    if args.file_start is not None:
        end = args.file_end if args.file_end is not None else args.file_start
        end = min(end, len(nii_files) - 1)
        nii_files = nii_files[args.file_start : end + 1]
        print(f"Range {args.file_start}-{end}: {len(nii_files)} files")
    if args.patient_limit:
        nii_files = nii_files[: args.patient_limit]

    if args.resume:
        before = len(nii_files)
        nii_files = [
            (cid, fp)
            for cid, fp in nii_files
            if not build_output_path(
                out_dir, cid, Path(fp).name.replace(".nii.gz", "")
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
        nii_files, step1_dir, args.preprocess_workers
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
            dst = build_output_path(out_dir, case_id, base_name)
            shutil.move(str(step3_file), str(dst))
            successful.append(str(dst))
            print(f"  ✓ {case_id}: {dst}")
        else:
            failed_cases.append(case_id)
            print(f"  ✗ vessel output missing: {case_id} ({base_name}.npz)")

    total = time.time() - t0
    print(f"\n{'='*60}\nBATCH SEGMENTATION COMPLETE")
    print(
        f"Files: {len(nii_files)}  Successful: {len(successful)}  Failed: {len(failed_cases)}"
    )
    print(f"Total: {total:.1f}s  ({total/len(nii_files):.1f}s/file)")
    print(f"  preprocess: {t_pre - t0:.1f}s  inference: {t_inf - t_pre:.1f}s")
    print(f"Outputs collected: {len(collect_all_step3_files(str(out_dir)))}")

    if args.cleanup:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
