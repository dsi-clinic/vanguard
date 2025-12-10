"""Batch extract centerlines from vessel segmentations using v07 script."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Configuration
SCRIPT_DIR = Path(__file__).parent.parent
SEGMENTATION_DIR = Path("/net/projects2/vanguard/vessel_segmentations")
OUTPUT_DIR = Path("/net/projects2/vanguard/centerlines")
SCRIPT_PATH = SCRIPT_DIR / "3dslicer_v07_graph_based.py"
NUM_FILES = 1004


def main() -> None:
    """Extract centerlines from first N vessel segmentations."""
    # Get list of segmentation files
    seg_files = sorted(SEGMENTATION_DIR.glob("*.npy"))[:NUM_FILES]

    if len(seg_files) == 0:
        print("No segmentation files found!")
        return

    print(f"Processing {len(seg_files)} vessel segmentations...")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Script: {SCRIPT_PATH}")
    print("=" * 60)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    successful = 0
    failed = 0

    for i, seg_file in enumerate(seg_files, 1):
        # Create output filename
        output_file = OUTPUT_DIR / f"{seg_file.stem}_centerlines.vtp"

        # Skip if already exists
        if output_file.exists():
            print(f"[{i}/{len(seg_files)}] ⏭️  Skipped (exists): {seg_file.name}")
            successful += 1
            continue

        print(f"[{i}/{len(seg_files)}] Processing: {seg_file.name}")

        # Run centerline extraction
        cmd = [
            sys.executable,
            str(SCRIPT_PATH),
            str(seg_file),
            str(output_file),
            "--no-visualizations",
            "--no-island-connection",  # Optional: faster processing
        ]

        try:
            result = subprocess.run(  # noqa: S603
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout per file
            )

            if result.returncode == 0 and output_file.exists():
                # Check if centerlines were actually extracted
                if output_file.stat().st_size > 0:
                    print(f"  ✅ Success: {output_file.name}")
                    successful += 1
                else:
                    print(f"  ⚠️  Empty output: {output_file.name}")
                    failed += 1
            else:
                print(
                    f"  ❌ Failed: {result.stderr[:200] if result.stderr else 'Unknown error'}"
                )
                failed += 1

        except subprocess.TimeoutExpired:
            print(f"  ⏱️  Timeout: {seg_file.name}")
            failed += 1
        except Exception as e:
            print(f"  ❌ Error: {e}")
            failed += 1

        # Progress update every 50 files
        if i % 50 == 0:
            print(f"\nProgress: {i}/{len(seg_files)} ({100*i/len(seg_files):.1f}%)")
            print(f"  Successful: {successful}, Failed: {failed}\n")

    print("\n" + "=" * 60)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total files: {len(seg_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
