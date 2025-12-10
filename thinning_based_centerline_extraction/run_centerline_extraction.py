"""Run centerline extraction and convert to JSON in one step."""

from __future__ import annotations

# Import the extraction function
import importlib.util
import sys
from pathlib import Path

# Import from same directory
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Add project root to path for centerline_to_json import
sys.path.insert(0, str(PROJECT_ROOT))

spec = importlib.util.spec_from_file_location(
    "extract_module", str(SCRIPT_DIR / "extract_centerlines.py")
)
extract_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract_module)
extract_adaptive_centerlines = extract_module.extract_adaptive_centerlines

import json  # noqa: E402

from centerline_to_json import centerlines_to_json  # noqa: E402

# Minimum number of arguments required (script name, input, output)
MIN_ARGS = 3


def main() -> None:
    """Run centerline extraction and convert to JSON in one step."""
    if len(sys.argv) < MIN_ARGS:
        print(
            "Usage: python run_centerline_extraction.py <input_segmentation> <output_json> [options]"
        )
        print("\nOptions:")
        print(
            "  --binarize-threshold FLOAT    Threshold for binarization (default: 0.5)"
        )
        print(
            "  --max-connection-distance-mm FLOAT  Max distance to connect islands (default: 15.0)"
        )
        print("  --no-island-connection        Disable island connection")
        print(
            "  --spacing X Y Z               Voxel spacing in mm (default: 1.0 1.0 1.0)"
        )
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])

    # Parse options
    binarize_threshold = 0.5
    max_connection_distance_mm = 15.0
    use_island_connection = True
    spacing = (1.0, 1.0, 1.0)

    i = 3
    while i < len(sys.argv):
        if sys.argv[i] == "--binarize-threshold" and i + 1 < len(sys.argv):
            binarize_threshold = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--max-connection-distance-mm" and i + 1 < len(sys.argv):
            max_connection_distance_mm = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--no-island-connection":
            use_island_connection = False
            i += 1
        elif sys.argv[i] == "--spacing" and i + 3 < len(sys.argv):
            spacing = (
                float(sys.argv[i + 1]),
                float(sys.argv[i + 2]),
                float(sys.argv[i + 3]),
            )
            i += 4
        else:
            i += 1

    # Temporary centerline file
    temp_centerline = output_path.parent / f"{output_path.stem}_temp.vtp"

    try:
        # Step 1: Extract centerlines
        print(f"Step 1: Extracting centerlines from {input_path}...")
        extract_adaptive_centerlines(
            str(input_path),
            str(temp_centerline),
            binarize_threshold=binarize_threshold,
            max_connection_distance_mm=max_connection_distance_mm,
            use_island_connection=use_island_connection,
            enable_visualizations=False,
        )

        # Step 2: Convert to JSON
        print("\nStep 2: Converting centerlines to JSON...")
        result = centerlines_to_json(
            str(temp_centerline), str(input_path), spacing=spacing
        )

        # Step 3: Save JSON
        with output_path.open("w") as f:
            json.dump(result, f, indent=4)

        print(f"\n✅ Successfully created JSON output: {output_path}")
        print(f"   Found {len(result)} vessel(s)")

    finally:
        # Clean up temporary file
        if temp_centerline.exists():
            temp_centerline.unlink()


if __name__ == "__main__":
    main()
