import sys
import os

PROJECT_ROOT = "/home/jmcarias/vanguard"
sys.path.append(os.path.join(PROJECT_ROOT, "notebooks"))
sys.path.append(os.path.join(PROJECT_ROOT, "notebooks/skeleton3d_utils"))
from skeleton3d_utils.pipeline import process_vessel_image

def main():
    if len(sys.argv) != 4:
        print("Usage: python run_pipeline.py <input_npy_path> <output_folder> <threshold>")
        sys.exit(1)

    input_npy_path = sys.argv[1]
    output_folder = sys.argv[2]
    threshold = float(sys.argv[3])

    # Derive image name and expected JSON path
    image_name = os.path.basename(input_npy_path).replace(".npy", "")
    image_output_dir = os.path.join(output_folder, image_name)
    output_json_path = os.path.join(image_output_dir, f"{image_name}_morphometry.json")

    # ---- Skip if already processed ----
    if os.path.exists(output_json_path):
        print(f"[SKIP] {image_name} already processed at {output_json_path}")
        return

    # Otherwise, run full pipeline
    process_vessel_image(input_npy_path, threshold, output_folder)


if __name__ == "__main__":
    main()