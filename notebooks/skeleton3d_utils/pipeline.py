import os
import sys
import numpy as np

sys.path.append(os.path.abspath("vanguard/notebooks"))
from skeleton3d_utils.skeleton3d import skeletonize3d
from skeleton3d_utils.skeleton3d_visuals import edges_to_segments
from skeleton3d_utils.skeleton_to_graph import *


def process_vessel_image(input_npy_path: str, threshold: float, output_folder: str):
    # Extract image name
    image_name = os.path.basename(input_npy_path).replace(".npy", "")

    # JSON output will be saved directly in output_folder
    output_json_path = os.path.join(output_folder, f"{image_name}_morphometry.json")

    # Skip if already processed
    if os.path.exists(output_json_path):
        print(f"[SKIP] {image_name} already processed at {output_json_path}")
        return output_json_path

    # 1. Load
    final_image = np.load(input_npy_path)
    vessels = final_image[1]

    # 2. Skeleton
    skeleton = skeletonize3d(vessels, threshold=threshold)

    # 3. Segments
    segments = edges_to_segments(skeleton)

    # 4. Graph
    G = segments_to_graph(segments)

    # 5. Radius map
    radius_map = obtain_radius_map(vessels, G)

    # 6. Paths
    segment_paths = extract_segments(G)

    # 7. Bifurcations
    bifurcations = detect_bifurcations(G)

    # 8. Component labels
    vessel_labels = assign_component_labels(G)

    # 9. Save ONLY the JSON (no folders, no .npy intermediate files)
    build_vessel_json(
        G,
        vessel_labels,
        segment_paths,
        radius_map,
        bifurcations,
        output_path=output_json_path,
    )

    print(f"[DONE] Saved JSON: {output_json_path}")
    return output_json_path