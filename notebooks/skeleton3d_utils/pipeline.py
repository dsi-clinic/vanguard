import os
import sys
import numpy as np

sys.path.append(os.path.abspath("vanguard/notebooks"))
from skeleton3d_utils.skeleton3d import skeletonize3d
from skeleton3d_utils.skeleton3d_visuals import edges_to_segments
from skeleton3d_utils.skeleton_to_graph import *


def process_vessel_image(input_npy_path: str, threshold: float, output_folder: str):
    """
    Complete end-to-end pipeline:
    .npy → vessels → skeleton → segments → graph → metrics → JSON

    Parameters
    ----------
    input_npy_path : str
        Path to the input .npy array file.
    threshold : float
        Skeletonization threshold (0–1).
    output_folder : str
        Folder where all outputs will be saved.

    Returns
    -------
    dict
        The final JSON morphometry dictionary.
    """
    # 1. LOAD IMAGE
    image_name = os.path.basename(input_npy_path).replace(".npy", "")
    image_output_dir = os.path.join(output_folder, image_name)
    os.makedirs(image_output_dir, exist_ok=True)

    print("1. Loading .npy image...")
    final_image = np.load(input_npy_path)
    vessels = final_image[1]    # Extract layer 1

    # 2. SKELETONIZATION
    print("2. Running skeletonization...")
    skeleton = skeletonize3d(vessels, threshold=threshold)

    # 3. EXTRACT SEGMENTS (raw voxel edges)
    print("3. Extracting voxel-level segments...")
    segments = edges_to_segments(skeleton)

    # 4. BUILD GRAPH
    print("4. Building skeleton graph...")
    G = segments_to_graph(segments)

    # 5. RADIUS MAP
    print("5. Computing radius map...")
    radius_map = obtain_radius_map(vessels, G)

    # 6. PATH SEGMENTS (between bifurcations)
    print("6. Extracting anatomical segment paths...")
    segment_paths = extract_segments(G)

    # 7. BIFURCATIONS
    print("7. Detecting bifurcations...")
    bifurcations = detect_bifurcations(G)

    # 8. CONNECTED COMPONENT LABELS
    print("8. Assigning component labels...")
    vessel_labels = assign_component_labels(G)

    # 9. BUILD JSON METRICS
    print("9. Computing vessel morphometry JSON...")
    output_json_path = os.path.join(image_output_dir, f"{image_name}_morphometry.json")

    build_vessel_json(
        G,
        vessel_labels,
        segment_paths,
        radius_map,
        bifurcations,
        output_path=output_json_path
    )

    print("10. Pipeline complete.")
    print(f"📄 Output saved to: {output_json_path}")