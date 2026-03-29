#!/usr/bin/env python3
"""Batch segmentation script for processing all .nii.gz files in /images directory

and extracting STEP-2 breast mask segmentations (.npy files).

This script:
1. Finds all .nii.gz files in the images directory
2. Processes each file through the breast segmentation pipeline (STEP-1 → STEP-2)
3. Collects all STEP-2 .npy files (breast masks) for further processing
4. Provides options for parallel processing and progress tracking
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import SimpleITK as sitk

# FIXME: this is cobbling together an installation process in the script itself
# packages must be installed in an environment
try:
    SCRIPT_DIR = Path(__file__).parent
    sys.path.insert(0, str(SCRIPT_DIR.parent / "vanguard-blood-vessel-segmentation"))

    from preprocessing import normalize_image, zscore_image  # noqa: E402

except ImportError:

    def normalize_image(*args, **_kwargs):  # noqa: ANN201, D103
        raise ImportError("Required preprocessing function not found")  # noqa: F821

    def zscore_image(*args, **_kwargs):  # noqa: ANN201, D103
        raise ImportError("Required preprocessing function not found")  # noqa: F821


def find_nii_files(images_dir: str) -> list[tuple[str, str]]:
    """Find all .nii.gz files in the images directory.

    Args:
        images_dir: Path to the images directory

    Returns:
        List of tuples (case_id, file_path)
    """
    nii_files = []

    # Get all case directories
    patient_dirs = [
        d for d in os.listdir(images_dir) if (Path(images_dir) / d).is_dir()
    ]

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


def run_vessel_segmentation(
    step1_dir: str,
    step2_dir: str,
    step3_dir: str,
    breast_model_path: str,
    vessel_model_path: str,
) -> bool:
    """Run complete segmentation pipeline: breast mask (STEP-2) then vessel segmentation (STEP-3).

    Args:
        step1_dir: Directory containing STEP-1 preprocessed files
        step2_dir: Directory to save STEP-2 breast masks
        step3_dir: Directory to save STEP-3 vessel segmentations
        breast_model_path: Path to the breast segmentation model
        vessel_model_path: Path to the vessel segmentation model

    Returns:
        True if successful, False otherwise
    """
    try:
        # Change to the segmentation project directory
        original_cwd = Path.cwd()
        script_dir = Path(__file__).parent
        os.chdir(script_dir.parent / "vanguard-blood-vessel-segmentation")

        # STEP-2: Run breast segmentation
        print("  Running breast segmentation (STEP-2)...")
        cmd_breast = [
            "python",
            "predict.py",
            "--target-tissue",
            "breast",
            "--image",
            step1_dir,
            "--save-masks-dir",
            step2_dir,
            "--model-save-path",
            breast_model_path,
        ]

        result_breast = subprocess.run(  # noqa: S603
            cmd_breast, capture_output=True, text=True, shell=False
        )

        if result_breast.returncode != 0:
            print(f"Breast segmentation failed: {result_breast.stderr}")
            os.chdir(original_cwd)
            return False

        # STEP-3: Run vessel segmentation
        print("  Running vessel segmentation (STEP-3)...")
        cmd_vessel = [
            "python",
            "predict.py",
            "--target-tissue",
            "dv",
            "--image",
            step1_dir,
            "--input-mask",
            step2_dir,
            "--save-masks-dir",
            step3_dir,
            "--model-save-path",
            vessel_model_path,
        ]

        result_vessel = subprocess.run(  # noqa: S603
            cmd_vessel, capture_output=True, text=True, shell=False
        )

        # Restore original working directory
        os.chdir(original_cwd)

        if result_vessel.returncode == 0:
            return True
        else:
            print(f"Vessel segmentation failed: {result_vessel.stderr}")
            return False

    except Exception as e:
        print(f"Error running segmentation: {e}")
        return False


def process_single_file(
    args: tuple[str, str, str, str, str, str],
) -> tuple[str, bool, str]:
    """Process a single .nii.gz file through the complete pipeline.

    Args:
        args: Tuple containing (case_id, file_path, temp_dir, output_dir, breast_model_path, vessel_model_path)

    Returns:
        Tuple of (case_id, success, output_path)
    """
    (
        case_id,
        file_path,
        temp_dir,
        output_dir,
        breast_model_path,
        vessel_model_path,
    ) = args

    try:
        # Create temporary directories for this file
        step1_dir = Path(temp_dir) / f"{case_id}_step1"
        step2_dir = Path(temp_dir) / f"{case_id}_step2"
        step3_dir = Path(temp_dir) / f"{case_id}_step3"

        step1_dir.mkdir(parents=True, exist_ok=True)
        step2_dir.mkdir(parents=True, exist_ok=True)
        step3_dir.mkdir(parents=True, exist_ok=True)

        # Extract filename without extension
        filename = Path(file_path).name
        base_name = filename.replace(".nii.gz", "")

        # STEP-1: Preprocess
        step1_file = step1_dir / f"{base_name}.npy"
        if not preprocess_image(file_path, step1_file):
            return case_id, False, ""

        # STEP-2 & STEP-3: Complete segmentation pipeline
        if not run_vessel_segmentation(
            step1_dir, step2_dir, step3_dir, breast_model_path, vessel_model_path
        ):
            return case_id, False, ""

        # Move the STEP-3 result to output directory
        step3_file = step3_dir / f"{base_name}.npz"
        output_file = build_output_path(Path(output_dir), case_id, base_name)

        if step3_file.exists():
            shutil.move(step3_file, output_file)
            return case_id, True, output_file
        else:
            return case_id, False, ""

    except Exception as e:
        print(f"Error processing {case_id}: {e}")
        return case_id, False, ""


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


def main() -> None:
    """Main function to run batch segmentation processing."""
    # Get script directory for relative paths
    script_dir = Path(__file__).parent

    parser = argparse.ArgumentParser(
        description="Batch process all .nii.gz files and extract vessel segmentations (STEP-3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--images-dir",
        default="/net/projects2/vanguard/MAMA-MIA-syn60868042/images",
        help="Directory containing case subdirectories with .nii.gz files",
    )

    parser.add_argument(
        "--output-dir",
        default=str(script_dir.parent / "vessel_segmentations"),
        help="Directory to save all STEP-3 vessel segmentation .npz files",
    )

    parser.add_argument(
        "--temp-dir",
        default=tempfile.mkdtemp(prefix="batch_segmentation_"),
        help="Temporary directory for intermediate processing",
    )

    parser.add_argument(
        "--breast-model-path",
        default=str(
            Path(__file__).parent.parent
            / "vanguard-blood-vessel-segmentation"
            / "trained_models"
            / "breast_model.pth"
        ),
        help="Path to the breast segmentation model (STEP-2)",
    )

    parser.add_argument(
        "--vessel-model-path",
        default=str(
            Path(__file__).parent.parent
            / "vanguard-blood-vessel-segmentation"
            / "trained_models"
            / "dv_model.pth"
        ),
        help="Path to the vessel segmentation model (STEP-3)",
    )

    parser.add_argument(
        "--max-workers", type=int, default=4, help="Maximum number of parallel workers"
    )

    parser.add_argument(
        "--patient-limit",
        type=int,
        default=None,
        help="Limit processing to first N cases (for testing)",
    )

    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up temporary files after processing",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume processing, skip already processed files",
    )

    parser.add_argument(
        "--file-index",
        type=int,
        default=None,
        help="Process only the file at this index in the sorted list (for SLURM array jobs)",
    )

    parser.add_argument(
        "--file-start",
        type=int,
        default=None,
        help="Process files from this start index (inclusive) in the sorted list",
    )

    parser.add_argument(
        "--file-end",
        type=int,
        default=None,
        help="Process files up to this end index (inclusive) in the sorted list",
    )

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.temp_dir).mkdir(parents=True, exist_ok=True)

    print(f"Finding .nii.gz files in {args.images_dir}...")
    nii_files = find_nii_files(args.images_dir)

    # Sort files for reproducible ordering
    nii_files = sorted(nii_files, key=lambda x: (x[0], str(x[1])))

    if args.patient_limit:
        nii_files = nii_files[: args.patient_limit]
        print(f"Limited to first {args.patient_limit} files for testing")

    print(f"Found {len(nii_files)} .nii.gz files to process")

    # If file-index is specified, process only that file (for SLURM array jobs)
    # Do this BEFORE resume filtering so we can check if the specific file was already processed
    if args.file_index is not None:
        if args.file_index < 0 or args.file_index >= len(nii_files):
            print(
                f"Error: file-index {args.file_index} is out of range (0-{len(nii_files)-1})"
            )
            return [], []

        nii_files = [nii_files[args.file_index]]
        print(f"Processing single file at index {args.file_index}: {nii_files[0][0]}")
    elif args.file_start is not None or args.file_end is not None:
        if args.file_start is None:
            print("Error: file-start is required when using file-end")
            return [], []

        file_start = args.file_start
        file_end = args.file_end if args.file_end is not None else args.file_start

        if file_start < 0 or file_start >= len(nii_files):
            print(
                f"Error: file-start {file_start} is out of range (0-{len(nii_files)-1})"
            )
            return [], []

        if file_end < file_start:
            print("Error: file-end must be >= file-start")
            return [], []

        if file_end >= len(nii_files):
            file_end = len(nii_files) - 1
            print(f"file-end exceeds max index, clamping to {file_end}")

        nii_files = nii_files[file_start : file_end + 1]
        print(f"Processing file range {file_start}-{file_end} ({len(nii_files)} files)")

    # Filter out already processed files if resuming
    if args.resume:
        original_count = len(nii_files)
        nii_files = [
            (case_id, file_path)
            for case_id, file_path in nii_files
            if not build_output_path(
                Path(args.output_dir),
                case_id,
                Path(file_path).name.replace(".nii.gz", ""),
            ).exists()
        ]
        skipped_count = original_count - len(nii_files)
        if skipped_count > 0:
            print(
                f"Resuming: {len(nii_files)} files remaining (skipped {skipped_count} already processed)"
            )
        if len(nii_files) == 0:
            print("All files already processed, exiting")
            return [], []

    # Prepare arguments for processing
    process_args = [
        (
            case_id,
            file_path,
            args.temp_dir,
            args.output_dir,
            args.breast_model_path,
            args.vessel_model_path,
        )
        for case_id, file_path in nii_files
    ]

    # Process files
    successful_files = []
    failed_files = []

    start_time = time.time()

    # If processing a single file (SLURM array job), don't use ProcessPoolExecutor
    if args.file_index is not None:
        # Process single file directly
        (
            case_id,
            file_path,
            temp_dir,
            output_dir,
            breast_model_path,
            vessel_model_path,
        ) = process_args[0]
        result = process_single_file(process_args[0])
        case_id, success, output_path = result

        if success:
            successful_files.append(output_path)
            print(f"✓ {case_id}: {output_path}")
        else:
            failed_files.append(case_id)
            print(f"✗ {case_id}: Failed")
    else:
        # Process multiple files in parallel
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            # Submit all tasks
            future_to_patient = {
                executor.submit(process_single_file, arg): arg[0]
                for arg in process_args
            }

            # Process completed tasks
            for i, future in enumerate(as_completed(future_to_patient), 1):
                case_id, success, output_path = future.result()

                if success:
                    successful_files.append(output_path)
                    print(f"[{i}/{len(nii_files)}] ✓ {case_id}: {output_path}")
                else:
                    failed_files.append(case_id)
                    print(f"[{i}/{len(nii_files)}] ✗ {case_id}: Failed")

    end_time = time.time()

    # Summary
    print(f"\n{'='*60}")
    print("BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total files processed: {len(nii_files)}")
    print(f"Successful: {len(successful_files)}")
    print(f"Failed: {len(failed_files)}")
    print(f"Processing time: {end_time - start_time:.2f} seconds")
    print(
        f"Average time per file: {(end_time - start_time) / len(nii_files):.2f} seconds"
    )

    if failed_files:
        print("\nFailed files:")
        for case_id in failed_files:
            print(f"  - {case_id}")

    # Collect all STEP-3 vessel segmentation files
    all_npy_files = collect_all_step3_files(args.output_dir)
    print(f"\nAll STEP-3 vessel segmentation .npz files saved to: {args.output_dir}")
    print(f"Total vessel segmentation files: {len(all_npy_files)}")

    # Cleanup
    if args.cleanup:
        print(f"\nCleaning up temporary directory: {args.temp_dir}")
        shutil.rmtree(args.temp_dir, ignore_errors=True)

    return successful_files, failed_files


if __name__ == "__main__":
    main()
