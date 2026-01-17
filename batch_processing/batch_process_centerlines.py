"""Batch process vessel segmentations to extract centerlines and convert to JSON."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import SimpleITK as sitk

# Get the directory paths
SCRIPT_DIR = Path(__file__).parent.parent
VESSEL_SEG_DIR = Path("/net/projects2/vanguard/vessel_segmentations")
IMAGES_DIR = Path("/net/projects2/vanguard/MAMA-MIA-syn60868042/images")
OUTPUT_DIR = SCRIPT_DIR / "centerline_json_outputs"


def get_spacing_from_segmentation_file(seg_file: Path) -> tuple[float, float, float]:
    """Extract spacing from original .nii.gz file based on segmentation filename.

    Segmentation files are named: {patient_id}_{base_name}_vessel_segmentation.npy
    Original files are at: {IMAGES_DIR}/{patient_id}/{base_name}.nii.gz

    Args:
        seg_file: Path to segmentation .npy file

    Returns:
        Tuple of (x, y, z) spacing in mm, or (1.0, 1.0, 1.0) if not found
    """
    # Parse filename: {patient_id}_{base_name}_vessel_segmentation.npy
    stem = seg_file.stem  # Remove .npy extension
    if not stem.endswith("_vessel_segmentation"):
        return (1.0, 1.0, 1.0)

    # Remove "_vessel_segmentation" suffix
    prefix = stem[: -len("_vessel_segmentation")]

    # Get list of all patient directories to match against
    if not IMAGES_DIR.exists():
        print(f"  Warning: Images directory {IMAGES_DIR} does not exist")
        return (1.0, 1.0, 1.0)

    patient_dirs = [d.name for d in IMAGES_DIR.iterdir() if d.is_dir()]

    # Try different splits: check if prefix starts with any patient_id
    # and if the remaining part matches a file in that directory
    parts = prefix.split("_")

    for i in range(1, len(parts)):
        candidate_patient_id = "_".join(parts[:i])
        candidate_base_name = "_".join(parts[i:])

        # Check if this patient_id exists as a directory
        if candidate_patient_id in patient_dirs:
            original_file = (
                IMAGES_DIR / candidate_patient_id / f"{candidate_base_name}.nii.gz"
            )
            if original_file.exists():
                try:
                    img = sitk.ReadImage(str(original_file))
                    spacing_xyz = img.GetSpacing()
                    print(f"  Found spacing from {original_file.name}: {spacing_xyz}")
                    return tuple(spacing_xyz)
                except Exception as e:
                    print(
                        f"  Warning: Could not read spacing from {original_file}: {e}"
                    )
                    continue

    # If we can't find the original file, return default spacing
    print(
        f"  Warning: Could not find original .nii.gz file for {seg_file.name}, using default spacing (1.0, 1.0, 1.0)"
    )
    return (1.0, 1.0, 1.0)


def process_file(input_file: Path, output_json: Path) -> bool:
    """Process a single file: extract centerlines and convert to JSON."""
    print(f"\n{'='*80}")
    print(f"Processing: {input_file.name}")
    print(f"{'='*80}")

    # Temporary centerline file
    temp_centerline = output_json.parent / f"{output_json.stem}_temp.vtp"

    try:
        # Extract spacing from original .nii.gz file
        print("Step 0: Extracting spacing from original image...")
        spacing = get_spacing_from_segmentation_file(input_file)
        spacing_str = f"{spacing[0]} {spacing[1]} {spacing[2]}"
        print(f"  Using spacing: {spacing_str}")

        # Step 1: Extract centerlines
        print("Step 1: Extracting centerlines...")
        cmd1 = [
            sys.executable,
            str(
                SCRIPT_DIR
                / "thinning_based_centerline_extraction"
                / "extract_centerlines.py"
            ),
            str(input_file),
            str(temp_centerline),
            "--no-visualizations",
        ]
        result1 = subprocess.run(cmd1, capture_output=True, text=True)  # noqa: S603

        if result1.returncode != 0:
            print("❌ Centerline extraction failed:")
            print(result1.stderr)
            return False

        # Check if centerlines were extracted
        if not temp_centerline.exists() or temp_centerline.stat().st_size == 0:
            print("⚠️  No centerlines extracted, skipping JSON conversion")
            return False

        # Step 2: Convert to JSON
        print("Step 2: Converting to JSON...")
        cmd2 = [
            sys.executable,
            str(SCRIPT_DIR / "centerline_to_json.py"),
            str(temp_centerline),
            str(output_json),
            "--segmentation",
            str(input_file),
            "--spacing",
            str(spacing[0]),
            str(spacing[1]),
            str(spacing[2]),
        ]
        result2 = subprocess.run(cmd2, capture_output=True, text=True)  # noqa: S603

        if result2.returncode != 0:
            print("❌ JSON conversion failed:")
            print(result2.stderr)
            return False

        # Verify output
        if output_json.exists() and output_json.stat().st_size > 0:
            print(f"✅ Successfully created: {output_json}")
            return True
        else:
            print("⚠️  JSON file is empty or missing")
            return False

    except Exception as e:
        print(f"❌ Error processing {input_file.name}: {e}")
        return False
    finally:
        # Clean up temporary file
        if temp_centerline.exists():
            temp_centerline.unlink()


def main() -> None:
    """Process all files from vessel_segmentations directory."""
    # Get list of all .npy files
    all_files = sorted(VESSEL_SEG_DIR.glob("*.npy"))

    if len(all_files) == 0:
        print("No .npy files found in vessel_segmentations directory")
        return

    # Process all files
    files_to_process = all_files

    if len(files_to_process) == 0:
        print("No files to process")
        return

    print(f"Processing {len(files_to_process)} files...")
    print(f"Output directory: {OUTPUT_DIR}")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    successful = 0
    failed = 0

    for input_file in files_to_process:
        # Create output JSON filename based on input filename
        output_json = OUTPUT_DIR / f"{input_file.stem}_centerlines.json"

        # Skip if already processed
        if output_json.exists() and output_json.stat().st_size > 0:
            print(f"⏭️  Skipping {input_file.name} (already processed)")
            successful += 1
            continue

        if process_file(input_file, output_json):
            successful += 1
        else:
            failed += 1

    print(f"\n{'='*80}")
    print("Batch processing complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
