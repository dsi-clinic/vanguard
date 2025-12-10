"""CLI helper to run the vessel processing pipeline on a single .npy file."""

import sys
from pathlib import Path


def main() -> None:
    """Parse CLI arguments, skip existing outputs, and call the pipeline."""
    project_root = Path(__file__).resolve().parents[2]
    sys.path.append(str(project_root / "graph_pruning_centerline_extraction"))
    sys.path.append(
        str(project_root / "graph_pruning_centerline_extraction" / "skeleton3d_utils")
    )

    from skeleton3d_utils.pipeline import process_vessel_image

    expected_args = 4
    if len(sys.argv) != expected_args:
        print(
            "Usage: python run_pipeline.py <input_npy_path> <output_folder> <threshold>"
        )
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    threshold = float(sys.argv[3])

    # Derive image name and expected JSON path
    image_name = input_path.stem
    image_output_dir = output_dir / image_name
    output_json_path = image_output_dir / f"{image_name}_morphometry.json"

    # ---- Skip if already processed ----
    if output_json_path.exists():
        print(f"[SKIP] {image_name} already processed at {output_json_path}")
        return

    # Otherwise, run full pipeline
    process_vessel_image(str(input_path), threshold, str(output_dir))


if __name__ == "__main__":
    main()
