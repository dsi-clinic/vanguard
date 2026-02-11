"""Graph-based centerline extraction from vessel segmentations.

Key features:
- Single extraction strategy: binarize → skeletonize → graph-based centerlines
- Optimized island connection to heal fragmented skeletons before graph building
- Off-screen PyVista visualizations retained for debugging
- Based on Matlab Skel2Graph3D.m, Conn_nearest_points_v2.m, Vessel_morph.m

Dependencies (conda-forge):
  - python==3.11
  - vtk>=9.2
  - nibabel, nrrd, einops
  - scikit-image (for skeletonization)
  - scipy (for k-nearest neighbor search and convolution)
"""

from __future__ import annotations

import os
import traceback
from pathlib import Path

import numpy as np

# VTK imports
import vtk
from einops import rearrange
from scipy.spatial import cKDTree
from skimage.morphology import label, skeletonize
from vtkmodules.util import numpy_support as vtknp
from vtkmodules.vtkCommonCore import vtkTypeFloat32Array
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPolyData
from vtkmodules.vtkIOImage import vtkNIFTIImageReader
from vtkmodules.vtkIOLegacy import vtkPolyDataWriter
from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter

# Constants
DIMENSIONS_3D = 3
THRESHOLD_DEFAULT = 0.5
NUMPY_4D_DIMENSION = 4
MIN_DOWNSAMPLE_DIMENSION = 2
MIN_NEIGHBORS_FOR_JUNCTION = 2
MAX_NEIGHBORS_FOR_ENDPOINT = 1
CANAL_NEIGHBORS = 2
MULTI_LABEL_THRESHOLD = 2

# Visualization optimization settings
VIZ_N_FRAMES = 60  # Reduced from 120 for faster generation
VIZ_FRAMERATE = 15
VIZ_WINDOW_SIZE = [1280, 720]  # Reduced from 1920x1080 for faster rendering
MIN_VOXELS_FOR_VOLUME_RENDER = 20000  # Skip volume viz when data is extremely sparse

# Default connection parameters (from Matlab code)
DEFAULT_MAX_CONNECTION_DISTANCE_MM = 15.0  # Maximum distance to connect islands (mm)
MAX_ENDPOINTS_PER_COMPONENT = (
    256  # Limit endpoints sampled per component when connecting islands
)
KD_TREE_K_NEIGHBORS = 32  # Nearest neighbors checked per endpoint (per iteration)

# PyVista for 3D vessel visualization
# IMPORTANT: All visualizations are configured for headless/offscreen rendering
# This is safe for SLURM compute nodes - no windows will pop up
# Based on working approach from visualize_all_labels_3d.py

# Detect if we're running without a DISPLAY (remote/headless/compute node)
NO_DISPLAY = not os.environ.get("DISPLAY")

# Force offscreen rendering for remote servers (set BEFORE importing pyvista)
if NO_DISPLAY:
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    os.environ.setdefault("PYVISTA_USE_PANEL", "false")
    os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "3.3")
    os.environ.setdefault("MESA_GLSL_VERSION_OVERRIDE", "330")
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    os.environ.setdefault("VTK_USE_X", "0")
    os.environ.setdefault("DISPLAY", "")
    print("No DISPLAY detected; enabling offscreen mode for compute nodes")
else:
    # Even with DISPLAY, use offscreen for video generation
    os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
    os.environ.setdefault("PYVISTA_USE_PANEL", "false")
    print("DISPLAY detected; using offscreen mode for video generation")

try:
    import pyvista as pv
except ImportError as exc:
    raise ImportError(
        "PyVista is required for visualization. Install it in the vmtk environment "
        "with `micromamba install -n vmtk -c conda-forge pyvista`."
    ) from exc

# Always use offscreen for video generation
pv.OFF_SCREEN = True

# Try to start Xvfb if available and no DISPLAY (like visualize_all_labels_3d.py)
if NO_DISPLAY and hasattr(pv, "start_xvfb"):
    try:
        pv.start_xvfb()
        print("Started Xvfb for offscreen rendering")
    except Exception as xvfb_err:
        print(f"Note: Could not start Xvfb automatically: {xvfb_err}")
        print(
            "  If rendering fails, ensure the upgraded VTK build supports EGL/OSMesa."
        )

# Try to reduce rendering overhead
try:
    if hasattr(pv.global_theme, "anti_aliasing"):
        pv.global_theme.anti_aliasing = False
except (AttributeError, Exception):  # noqa: S110
    pass  # Ignore if attribute doesn't exist or can't be set

# Suppress VTK render window warnings
try:
    vtk.vtkRenderWindow.SetGlobalWarningDisplay(0)
except Exception:  # noqa: S110
    pass

# Set theme (optional, won't affect headless rendering)
try:
    pv.set_plot_theme("dark")
except Exception:  # noqa: S110
    pass

print("PyVista loaded with offscreen rendering enabled")

__all__ = ["extract_adaptive_centerlines"]


class UnionFind:
    """Union-Find (Disjoint Set) data structure for tracking component merges."""

    def __init__(self: UnionFind, n: int) -> None:
        """Initialize with n elements."""
        self.parent = np.arange(n, dtype=np.int32)
        self.rank = np.zeros(n, dtype=np.int32)

    def find(self: UnionFind, x: int) -> int:
        """Find root of x with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self: UnionFind, x: int, y: int) -> bool:
        """Union x and y. Returns True if they were in different sets."""
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False

        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1

        return True

    def get_components(self: UnionFind) -> dict[int, set[int]]:
        """Get all components as a dictionary mapping root to set of elements."""
        components = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            if root not in components:
                components[root] = set()
            components[root].add(i)
        return components


def _create_visualization_dir(output_path: Path) -> Path:
    """Create visualization directory based on output path."""
    viz_dir = output_path.parent / f"{output_path.stem}_visualizations"
    viz_dir.mkdir(exist_ok=True)
    return viz_dir


def _find_skeleton_endpoints_vectorized(skeleton: np.ndarray) -> np.ndarray:
    """Find endpoints in a 3D skeleton using vectorized convolution.

    An endpoint is a skeleton point with exactly one neighbor.
    Uses 26-connectivity (all neighbors including diagonals).

    This is much faster than the loop-based version for large skeletons.

    Args:
        skeleton: Binary 3D skeleton array

    Returns:
        Array of (z, y, x) coordinates of endpoints, shape (n_endpoints, 3)
    """
    # Create 3x3x3 kernel for 26-connectivity neighbor counting
    # The kernel counts neighbors (excluding center)
    kernel = np.ones((3, 3, 3), dtype=np.float32)
    kernel[1, 1, 1] = 0  # Exclude center point

    # Use correlation to count neighbors for each point
    from scipy.ndimage import correlate

    neighbor_count = correlate(
        skeleton.astype(np.float32), kernel, mode="constant", cval=0.0
    )

    # Endpoints have exactly 1 neighbor (and are skeleton points)
    endpoint_mask = (neighbor_count == 1) & (skeleton > 0)

    # Get coordinates
    endpoints = np.column_stack(np.where(endpoint_mask))

    return endpoints  # Shape: (n_endpoints, 3) with columns [z, y, x]


def _find_skeleton_endpoints(skeleton: np.ndarray) -> list[tuple[int, int, int]]:
    """Find endpoints in a 3D skeleton (fallback to vectorized version).

    Args:
        skeleton: Binary 3D skeleton array

    Returns:
        List of (z, y, x) coordinates of endpoints
    """
    endpoints_array = _find_skeleton_endpoints_vectorized(skeleton)
    return [tuple(ep) for ep in endpoints_array]


def _print_viz_debug(stage: str, **info: object) -> None:
    """Print structured visualization debugging info."""
    info_items = ", ".join(f"{k}={v}" for k, v in info.items())
    env = {
        "DISPLAY": os.environ.get("DISPLAY", ""),
        "PYVISTA_OFF_SCREEN": os.environ.get("PYVISTA_OFF_SCREEN", ""),
        "PYVISTA_USE_PANEL": os.environ.get("PYVISTA_USE_PANEL", ""),
    }
    env_items = ", ".join(f"{k}={repr(v)}" for k, v in env.items())
    print(f"[viz-debug] stage={stage} | {info_items}")
    print(f"[viz-debug] stage={stage} | env={env_items}")


def _skeletonize_binary_volume(volume: np.ndarray, pad: int = 1) -> np.ndarray:
    """Skeletonize a binary 3D volume after cropping to its tight bounding box.

    Cropping dramatically reduces the workload for sparse masks while producing
    the same final skeleton when the cropped region is placed back into the full
    frame.
    """
    if volume.dtype != np.bool_:
        volume = volume.astype(bool, copy=False)

    coords = np.argwhere(volume)
    if coords.size == 0:
        return np.zeros_like(volume, dtype=bool)

    mins = np.maximum(coords.min(axis=0) - pad, 0)
    maxs = np.minimum(coords.max(axis=0) + pad + 1, volume.shape)

    subvolume = volume[mins[0] : maxs[0], mins[1] : maxs[1], mins[2] : maxs[2]]
    skeleton_sub = skeletonize(subvolume)

    skeleton_full = np.zeros_like(volume, dtype=bool)
    skeleton_full[mins[0] : maxs[0], mins[1] : maxs[1], mins[2] : maxs[2]] = (
        skeleton_sub
    )
    return skeleton_full


def _ensure_zyx(volume: np.ndarray, name: str = "") -> np.ndarray:
    """Ensure volume is arranged as (z, y, x)."""
    if volume.ndim != DIMENSIONS_3D:
        return volume
    original_shape = volume.shape
    if volume.shape[0] == volume.shape[1] and volume.shape[2] != volume.shape[0]:
        volume = np.moveaxis(volume, -1, 0)
        if name:
            print(f"  Reordered {name} from {original_shape} to {volume.shape} (z,y,x)")
    return volume


def _draw_line_3d(
    skel: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
) -> None:
    """Draw a 3D line between two points in the skeleton array.

    Uses vectorized line drawing with np.linspace.

    Args:
        skel: Skeleton array to modify (in-place)
        p1: Start point (x, y, z)
        p2: End point (x, y, z)
    """
    # Calculate number of points needed
    diff = np.abs(p2 - p1)
    n_points = int(np.ceil(np.max(diff) * np.sqrt(2))) + 1

    # Generate line points using linspace (vectorized)
    t_values = np.linspace(0, 1, n_points)
    points = p1[None, :] + t_values[:, None] * (p2 - p1)[None, :]

    # Round to integer coordinates
    points_int = np.round(points).astype(np.int32)

    # Filter valid points (within bounds)
    valid_mask = (
        (points_int[:, 0] >= 0)
        & (points_int[:, 0] < skel.shape[2])
        & (points_int[:, 1] >= 0)
        & (points_int[:, 1] < skel.shape[1])
        & (points_int[:, 2] >= 0)
        & (points_int[:, 2] < skel.shape[0])
    )

    # Set skeleton points (convert x,y,z to z,y,x for array indexing)
    valid_points = points_int[valid_mask]
    if len(valid_points) > 0:
        skel[valid_points[:, 2], valid_points[:, 1], valid_points[:, 0]] = 1


def _connect_nearest_islands_optimized(
    skeleton: np.ndarray,
    spacing: tuple[float, float, float],
    max_distance_mm: float = DEFAULT_MAX_CONNECTION_DISTANCE_MM,
    max_endpoints_per_component: int = MAX_ENDPOINTS_PER_COMPONENT,
) -> np.ndarray:
    """Connect sparse islands in skeleton using optimized batch connection method.

    Args:
        skeleton: Binary skeleton array (z, y, x).
        spacing: Voxel spacing (z, y, x) in mm.
        max_distance_mm: Maximum distance to connect islands (mm).
        max_endpoints_per_component: Maximum endpoints to sample per component.

    OPTIMIZATIONS (v05):
    - Batch connection: Find all valid connections in one pass, connect them all
    - Precomputed endpoints: Find all endpoints once, group by component
    - Global KD-tree: Build one per iteration instead of per component
    - Vectorized operations: Use numpy operations instead of loops
    - Component size precomputation: Use np.bincount for O(n) size calculation
    - Union-Find tracking: Track component merges without full relabeling

    Expected speedup: 10-100x for datasets with many components.

    Args:
        skeleton: Binary 3D skeleton array (Z, Y, X)
        spacing: Voxel spacing (Z, Y, X) in mm
        max_distance_mm: Maximum distance to connect islands (mm)

    Returns:
        Connected skeleton array
    """
    print(
        f"Connecting sparse islands (optimized v05, max distance: {max_distance_mm} mm)..."
    )

    # Convert to binary
    skel = skeleton.copy().astype(np.uint8)
    skel[skel > 0] = 1

    # Calculate aspect ratio for distance calculation
    ratio = (1.0, 1.0, round(spacing[2] / spacing[0], 2))

    # Convert max distance from mm to voxels (using minimum spacing)
    min_spacing = min(spacing)
    max_distance_voxels = max_distance_mm / min_spacing

    # Initial connected components labeling
    labeled = label(skel, connectivity=3)  # 3 = 26-connectivity in 3D
    unique_labels = np.unique(labeled)
    unique_labels = unique_labels[unique_labels > 0]  # Remove background
    num_components = len(unique_labels)

    print(f"  Found {num_components} disconnected components")

    if num_components <= 1:
        print("  No islands to connect")
        return skel

    # Initialize tracking structures lazily so we can refresh them only when needed
    label_to_idx: dict[int, int] = {}
    uf: UnionFind | None = None

    iteration = 0
    max_iterations = min(
        num_components * 2, 1000
    )  # Safety limit (reduced from num_components)
    relabel_interval = max(25, num_components // 50)  # Relabel less frequently
    relabel_needed = True

    # Iterate until all components connected or max distance exceeded
    while num_components > 1 and iteration < max_iterations:
        iteration += 1

        # Relabel periodically to update component structure
        if relabel_needed or iteration % relabel_interval == 0 or iteration == 1:
            labeled = label(skel, connectivity=3)
            unique_labels = np.unique(labeled)
            unique_labels = unique_labels[unique_labels > 0]
            num_components = len(unique_labels)

            # Rebuild union-find mapping
            label_to_idx = {lab: idx for idx, lab in enumerate(unique_labels)}
            uf = UnionFind(num_components)
            relabel_needed = False

        if num_components <= 1:
            break

        print(f"  Iteration {iteration}: {num_components} components remaining")

        # Precompute component sizes using np.bincount (vectorized, O(n))
        component_sizes = np.bincount(labeled.ravel())
        component_sizes = component_sizes[unique_labels]  # Only non-zero labels

        # Get component size pairs (size, label) and sort by size (smallest first)
        size_label_pairs = [
            (component_sizes[i], lab) for i, lab in enumerate(unique_labels)
        ]
        size_label_pairs.sort()

        # Precompute ALL endpoints once (vectorized)
        all_endpoints_zyx = _find_skeleton_endpoints_vectorized(skel)

        if len(all_endpoints_zyx) == 0:
            print("    No endpoints found, stopping")
            break

        # Group endpoints by component label
        endpoint_labels = labeled[
            all_endpoints_zyx[:, 0], all_endpoints_zyx[:, 1], all_endpoints_zyx[:, 2]
        ]
        endpoints_by_component = {}
        for i, (ep_z, ep_y, ep_x) in enumerate(all_endpoints_zyx):
            comp_label = endpoint_labels[i]
            if comp_label > 0:  # Skip background
                if comp_label not in endpoints_by_component:
                    endpoints_by_component[comp_label] = []
                endpoints_by_component[comp_label].append((ep_z, ep_y, ep_x))

        # Build global KD-tree of ALL skeleton points (not just endpoints)
        # This allows finding nearest points from any component
        all_skeleton_points = np.column_stack(np.where(skel > 0))
        if len(all_skeleton_points) == 0:
            break

        # Convert to (x, y, z) and scale by aspect ratio
        all_coords_xyz = all_skeleton_points[:, [2, 1, 0]]  # Convert z,y,x to x,y,z
        all_coords_scaled = all_coords_xyz.astype(np.float64)
        all_coords_scaled[:, 0] *= ratio[0]
        all_coords_scaled[:, 1] *= ratio[1]
        all_coords_scaled[:, 2] *= ratio[2]
        point_labels = labeled[
            all_skeleton_points[:, 0],
            all_skeleton_points[:, 1],
            all_skeleton_points[:, 2],
        ]

        # Build global KD-tree
        global_tree = cKDTree(all_coords_scaled)

        # BATCH CONNECTION: Find all valid connections in one pass
        connections_to_make = []

        # Try to connect smallest components first
        for size, target_label in size_label_pairs:
            if size <= 1:  # Skip single-pixel components
                continue

            if target_label not in endpoints_by_component:
                continue

            target_endpoints = endpoints_by_component[target_label]
            if len(target_endpoints) > max_endpoints_per_component:
                step = max(1, len(target_endpoints) // max_endpoints_per_component)
                target_endpoints = target_endpoints[::step][
                    :max_endpoints_per_component
                ]

            # Find closest connection for this component
            best_distance = float("inf")
            best_connection = None

            for ep_z, ep_y, ep_x in target_endpoints:
                # Convert endpoint to (x, y, z) and scale
                ep_coord = np.array([ep_x, ep_y, ep_z], dtype=np.float64)
                ep_coord_scaled = ep_coord.copy()
                ep_coord_scaled[0] *= ratio[0]
                ep_coord_scaled[1] *= ratio[1]
                ep_coord_scaled[2] *= ratio[2]

                # Query a small batch of nearest neighbors with an upper distance bound
                k = min(KD_TREE_K_NEIGHBORS, len(all_coords_scaled))
                distances, indices = global_tree.query(
                    ep_coord_scaled,
                    k=k,
                    distance_upper_bound=max_distance_voxels,
                )

                distances = np.atleast_1d(distances)
                indices = np.atleast_1d(indices)
                valid_mask = np.isfinite(distances) & (indices < len(all_coords_scaled))
                if not np.any(valid_mask):
                    continue

                candidate_indices = indices[valid_mask].astype(int, copy=False)
                candidate_labels = point_labels[candidate_indices]
                different_component_mask = candidate_labels != target_label
                if not np.any(different_component_mask):
                    continue

                filtered_indices = candidate_indices[different_component_mask]
                filtered_distances = distances[valid_mask][different_component_mask]

                min_idx = int(np.argmin(filtered_distances))
                min_dist = float(filtered_distances[min_idx])

                if min_dist < best_distance:
                    candidate_coords = all_coords_xyz[filtered_indices]
                    best_distance = min_dist
                    best_connection = (
                        (ep_z, ep_y, ep_x),
                        tuple(candidate_coords[min_idx]),
                        min_dist,
                    )

            if best_connection is not None:
                connections_to_make.append((target_label, best_connection))

        # Make all connections found in this iteration
        if len(connections_to_make) == 0:
            print(
                f"    No more connections possible (distance > {max_distance_voxels:.2f} voxels)"
            )
            break

        # Sort connections by distance (connect closest first)
        connections_to_make.sort(key=lambda x: x[1][2])  # Sort by distance

        connections_made = 0
        for target_label, (
            (ep_z, ep_y, ep_x),
            (target_x, target_y, target_z),
            _dist,
        ) in connections_to_make:
            # Check if components are still separate (using union-find)
            target_idx = label_to_idx.get(target_label)
            if target_idx is None:
                continue

            # Find which component the target point belongs to
            target_point_label = labeled[target_z, target_y, target_x]
            if target_point_label == 0 or target_point_label == target_label:
                continue

            target_point_idx = label_to_idx.get(target_point_label)
            if target_point_idx is None:
                continue

            # Check if already connected (union-find)
            if uf is not None and uf.find(target_idx) == uf.find(target_point_idx):
                continue  # Already connected

            # Draw connection line
            p1 = np.array([ep_x, ep_y, ep_z], dtype=np.float64)
            p2 = np.array([target_x, target_y, target_z], dtype=np.float64)
            _draw_line_3d(skel, p1, p2)

            # Update union-find
            uf.union(target_idx, target_point_idx)
            connections_made += 1

        print(f"    Connected {connections_made} component pair(s) in this iteration")

        # Update component count (approximate, will be exact after relabel)
        num_components = max(1, num_components - connections_made)
        relabel_needed = connections_made > 0

    # Final relabel to get accurate count
    labeled = label(skel, connectivity=3)
    final_components = len(np.unique(labeled)) - 1
    print(f"  Final: {final_components} connected component(s)")

    return skel


# Alias for backward compatibility
_connect_nearest_islands = _connect_nearest_islands_optimized


def _render_volume_visualization(
    volume: np.ndarray,
    viz_dir: Path,
    *,
    stage: str,
    description: str,
    filename: str,
    cmap: str = "Blues",
    surface_stage: str | None = None,
    surface_description: str | None = None,
    surface_color: str = "cyan",
    nonzero_threshold: float = 0.0,
    treat_probability: bool = False,
) -> None:
    """Render a volume as a rotating MP4, falling back to an isosurface if sparse."""
    data = volume.astype(np.float32, copy=False)
    print(f"Data shape: {data.shape}")
    data_min, data_max = np.min(data), np.max(data)
    print(f"Data range: [{data_min:.4f}, {data_max:.4f}]")

    if treat_probability:
        is_probability = (
            data_max <= 1.0
            and data_min >= 0.0
            and not np.allclose(data, data.astype(bool))
        )
        if is_probability:
            print(
                "  Detected probability data - using input values directly for rendering"
            )
        else:
            print("  Detected binary data - casting to float for rendering")

    non_zero_voxels = int(np.count_nonzero(data > nonzero_threshold))
    voxel_fraction = non_zero_voxels / data.size if data.size else 0.0
    print(
        f"Voxels above threshold: {non_zero_voxels} ({100*voxel_fraction:.2f}% of volume)"
    )
    _print_viz_debug(
        stage,
        voxels=non_zero_voxels,
        fraction=f"{voxel_fraction:.6f}",
        dtype=str(data.dtype),
    )

    if non_zero_voxels < MIN_VOXELS_FOR_VOLUME_RENDER:
        surface_stage = surface_stage or f"{stage}_surface"
        surface_description = surface_description or description
        success = _render_isosurface_visualization(
            data,
            viz_dir,
            surface_stage,
            surface_description,
            color=surface_color,
        )
        if not success:
            raise RuntimeError(f"Isosurface visualization failed for {surface_stage}")
        return

    grid = pv.wrap(data)
    plotter = pv.Plotter(off_screen=True, window_size=VIZ_WINDOW_SIZE)
    plotter.add_volume(grid, opacity="linear", cmap=cmap)
    plotter.background_color = "white"
    plotter.show_axes()

    output_path = viz_dir / filename
    print(f"Saving {description} visualization to: {output_path}")

    try:
        success = _record_pyvista_rotation(plotter, output_path, stage)
    finally:
        plotter.close()

    if not success:
        raise RuntimeError(f"Failed to record PyVista rotation for stage {stage}")


def _visualize_3d_vessels_from_numpy(vessel_data: np.ndarray, viz_dir: Path) -> None:
    """Create 3D vessel visualization directly from numpy array (notebook style)."""
    print("Creating 3D vessel visualization (notebook style - direct numpy)...")
    _render_volume_visualization(
        vessel_data,
        viz_dir,
        stage="vessels_numpy",
        description="Input volume (NumPy)",
        filename="vessels_3d.mp4",
        cmap="Blues_r",
        surface_stage="vessels_numpy_surface",
        surface_description="Sparse vessel surface",
        surface_color="darkblue",
        nonzero_threshold=0.0,
        treat_probability=True,
    )


def _visualize_3d_vessels(img: vtkImageData, viz_dir: Path) -> None:
    """Create a 3D rotating volume visualization of the vessel mask stored in a VTK image."""
    scalars = img.GetPointData().GetScalars()
    if not scalars:
        raise RuntimeError("VTK image does not contain scalar data for visualization")

    data = vtknp.vtk_to_numpy(scalars)
    dims = img.GetDimensions()

    data_3d = data.reshape((dims[2], dims[1], dims[0])) if data.ndim == 1 else data

    print("Creating 3D vessel visualization from VTK image...")
    _render_volume_visualization(
        data_3d,
        viz_dir,
        stage="vessels_vtk",
        description="Input volume (VTK)",
        filename="vessels_3d.mp4",
        cmap="Blues_r",
        surface_stage="vessels_vtk_surface",
        surface_description="Sparse vessel surface (VTK)",
        surface_color="darkblue",
        nonzero_threshold=0.0,
        treat_probability=True,
    )


def _visualize_intermediate_stage(
    data: np.ndarray, viz_dir: Path, stage: str, description: str = ""
) -> None:
    """Visualize intermediate processing stages as MP4."""
    if len(data.shape) != DIMENSIONS_3D:
        print(f"Expected 3D data, got shape {data.shape}")
        return

    print(f"Creating {stage} 3D visualization...")
    if "skeleton" in stage.lower():
        cmap = "hot"
    elif "dilated" in stage.lower():
        cmap = "viridis"
    else:
        cmap = "magma"

    _render_volume_visualization(
        data,
        viz_dir,
        stage=stage,
        description=description or stage,
        filename=f"{stage}_stage_visualization.mp4",
        cmap=cmap,
        surface_stage=f"{stage}_surface",
        surface_description=description or stage,
        surface_color="darkorange" if "skeleton" in stage else "darkmagenta",
        nonzero_threshold=0.01,
    )


def _record_pyvista_rotation(
    plotter: pv.Plotter, output_path: Path, stage: str
) -> bool:
    """Capture rotating PyVista frames and encode with imageio."""
    try:
        import imageio as iio
    except ImportError:
        print("imageio not available; cannot capture PyVista movies.")
        return False

    n_frames = VIZ_N_FRAMES
    azimuth_angles = 180 + np.arange(n_frames, dtype=np.float32) * 360.0 / n_frames
    plotter.camera_position = "yz"
    plotter.camera.elevation = 30

    try:
        writer = iio.get_writer(
            str(output_path), fps=VIZ_FRAMERATE, codec="libx264", bitrate="2000k"
        )
    except Exception as e:
        print(f"⚠ Warning: Could not open video writer for {stage}: {e}")
        traceback.print_exc()
        return False

    try:
        for i, angle in enumerate(azimuth_angles):
            plotter.camera.azimuth = angle
            plotter.render()
            frame = plotter.screenshot(return_img=True)
            if frame is None:
                raise RuntimeError("PyVista screenshot returned None")
            writer.append_data(np.ascontiguousarray(frame))
            if (i + 1) % max(1, n_frames // 4) == 0 or i == n_frames - 1:
                print(
                    f"    Progress: {i + 1}/{n_frames} frames ({100*(i+1)/n_frames:.1f}%)"
                )
    except Exception as e:
        print(f"⚠ Warning: PyVista frame capture failed for {stage}: {e}")
        traceback.print_exc()
        writer.close()
        return False
    else:
        writer.close()
        if output_path.exists():
            size = output_path.stat().st_size / (1024 * 1024)
            print(f"✓ PyVista movie saved: {output_path} ({size:.2f} MB)")
        else:
            print(f"⚠ Warning: Movie file missing for {stage}")
        return True


def _render_isosurface_visualization(
    volume: np.ndarray,
    viz_dir: Path,
    stage: str,
    description: str,
    color: str = "cyan",
) -> bool:
    """Render a sparse volume as an isosurface using PyVista."""
    try:
        data = volume.astype(np.float32, copy=False)
        max_cells = 6_000_000  # limit cells to reduce VTK crashes
        downsample_factor = 1
        while data.size > max_cells and min(data.shape) >= MIN_DOWNSAMPLE_DIMENSION:
            data = data[::2, ::2, ::2]
            downsample_factor *= 2
        if downsample_factor > 1:
            print(
                f"  Downsampled volume by factor {downsample_factor} for isosurface "
                f"(new shape: {data.shape})"
            )

        grid = pv.wrap(data)
        surface = grid.contour([0.5])
        if surface.n_points == 0:
            print(f"No isosurface points generated for {stage}")
            return False
        plotter = pv.Plotter(off_screen=True, window_size=VIZ_WINDOW_SIZE)
        plotter.add_mesh(surface, color=color, opacity=1.0)
        plotter.background_color = "white"
        plotter.show_axes()
        output_path = viz_dir / f"{stage}_stage_visualization.mp4"
        print(f"Saving {stage} isosurface visualization to: {output_path}")
        success = False
        try:
            success = _record_pyvista_rotation(plotter, output_path, stage)
        finally:
            plotter.close()
        return success
    except Exception as e:
        print(f"⚠ Warning: Isosurface visualization failed for {stage}: {e}")
        traceback.print_exc()
        return False


def _visualize_centerlines(centerlines: vtkPolyData, viz_dir: Path) -> None:
    """Visualize extracted centerlines as 3D rotating MP4."""
    if centerlines.GetNumberOfPoints() == 0:
        print("⚠️  No centerlines detected - cannot create visualization")
        print(
            "   This usually means VMTK network extraction failed or returned empty results."
        )
        print("   Possible causes:")
        print("     - Preprocessed data is too sparse or disconnected")
        print("     - Surface generation failed (empty surface)")
        print("     - VMTK network extraction could not find valid centerlines")
        return

    print("Creating centerlines 3D visualization...")
    print(f"Centerlines points: {centerlines.GetNumberOfPoints()}")
    print(f"Centerlines lines: {centerlines.GetNumberOfCells()}")

    plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
    plotter.background_color = "white"
    centerlines_pv = pv.wrap(centerlines)
    centerlines_tubes = centerlines_pv.tube(radius=0.3, n_sides=8)
    plotter.add_mesh(centerlines_tubes, color="darkblue")

    output_path = viz_dir / "final_centerlines_3d.mp4"
    print(f"Saving centerlines visualization to: {output_path}")

    try:
        success = _record_pyvista_rotation(plotter, output_path, "centerlines")
    finally:
        plotter.close()

    if not success:
        raise RuntimeError("Failed to record PyVista rotation for centerlines")


def _extract_skeleton_centerlines(
    img: vtkImageData,
    use_island_connection: bool = False,
    max_connection_distance_mm: float = DEFAULT_MAX_CONNECTION_DISTANCE_MM,
    viz_dir: Path | None = None,
) -> vtkPolyData:
    """Extract centerlines using 3D skeletonization.

    Args:
        img: Input VTK ImageData (should already be binary from main extraction function)
        use_island_connection: Whether to connect sparse islands before skeletonization
        max_connection_distance_mm: Maximum distance to connect islands (if enabled)
        viz_dir: Optional visualization directory for skeleton visualization
    """
    scalars = img.GetPointData().GetScalars()
    data = vtknp.vtk_to_numpy(scalars)

    # Reshape to 3D
    dims = img.GetDimensions()
    # Data is already binary from extract_adaptive_centerlines (thresholded at 0.01)
    # Just ensure it's uint8 and properly shaped
    binary_data = (
        (data > 0).reshape((dims[2], dims[1], dims[0])).astype(bool, copy=False)
    )

    voxel_count = np.count_nonzero(binary_data)
    if voxel_count == 0:
        print(
            "  ⚠️  No foreground voxels detected after binarization; returning empty result."
        )
        return vtkPolyData()
    print(
        f"  Binary data for skeletonization: {voxel_count} voxels ({voxel_count/binary_data.size*100:.4f}% of volume)"
    )

    # Extract 3D skeleton (crop to bounding box for speed)
    skeleton = _skeletonize_binary_volume(binary_data)
    skeleton_points = np.count_nonzero(skeleton)
    print(f"  Skeleton points: {skeleton_points}")

    # Visualize skeleton before connection (if requested)
    if viz_dir and use_island_connection:
        try:
            _visualize_intermediate_stage(
                skeleton.astype(np.float32),
                viz_dir,
                "skeleton_before_connection",
                "Skeleton before island connection",
            )
        except Exception as e:
            print(f"Warning: Skeleton visualization failed: {e}")

    # Optionally connect sparse islands to create more coherent centerlines
    if use_island_connection:
        spacing_xyz = img.GetSpacing()
        spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
        skeleton = _connect_nearest_islands_optimized(
            skeleton, spacing_zyx, max_connection_distance_mm
        )

        # Visualize skeleton after connection
        if viz_dir:
            try:
                _visualize_intermediate_stage(
                    skeleton.astype(np.float32),
                    viz_dir,
                    "skeleton_after_connection",
                    "Skeleton after island connection",
                )
            except Exception as e:
                print(f"Warning: Connected skeleton visualization failed: {e}")

    # Convert skeleton to VTK PolyData
    return _skeleton_to_polydata(skeleton, img.GetSpacing())


def _skel2graph3d(
    skeleton_array: np.ndarray,
    z_coords: np.ndarray,
    y_coords: np.ndarray,
    x_coords: np.ndarray,
    coord_to_idx: dict,
    graph: list[list[int]],
    min_branch_length: int = 10,
) -> tuple[list[dict], list[dict]]:
    """Convert 3D skeleton to graph structure (nodes and links) - Python version of Skel2Graph3D.

    Args:
        skeleton_array: Binary skeleton array.
        z_coords: Z coordinates of skeleton points.
        y_coords: Y coordinates of skeleton points.
        x_coords: X coordinates of skeleton points.
        coord_to_idx: Mapping from (z, y, x) coordinates to point indices.
        graph: Adjacency list representation of skeleton graph.
        min_branch_length: Minimum branch length to keep.

    Based on Matlab Skel2Graph3D.m:
    - Identifies nodes (junctions with >2 neighbors, endpoints with 1 neighbor)
    - Follows links from nodes to other nodes
    - Each link contains all points along the path (link.point equivalent)

    Args:
        skeleton_array: Binary skeleton array
        z_coords, y_coords, x_coords: Coordinate arrays for skeleton points
        coord_to_idx: Mapping from (z,y,x) to point index
        graph: Adjacency list representation
        min_branch_length: Minimum length of branches to keep

    Returns:
        (nodes, links) where:
        - nodes: List of node dicts with 'idx', 'comx', 'comy', 'comz', 'ep' (endpoint flag)
        - links: List of link dicts with 'n1', 'n2', 'point' (list of point indices)
    """
    n_points = len(z_coords)

    # Identify nodes: junctions (>2 neighbors) and endpoints (1 neighbor)
    # Also identify canal voxels (exactly 2 neighbors - part of a path)
    junction_voxels = []  # Indices with >2 neighbors
    endpoint_voxels = []  # Indices with exactly 1 neighbor
    canal_voxels = []  # Indices with exactly 2 neighbors

    for i in range(n_points):
        num_neighbors = len(graph[i])
        if num_neighbors > MIN_NEIGHBORS_FOR_JUNCTION:
            junction_voxels.append(i)
        elif num_neighbors == 1:
            endpoint_voxels.append(i)
        elif num_neighbors == CANAL_NEIGHBORS:
            canal_voxels.append(i)

    # Group adjacent junction voxels into nodes (like Matlab Skel2Graph3D)
    # Use connected components to group junction voxels
    nodes = []
    node_idx_map = {}  # Maps point index to node index

    # Create a temporary array to find connected junction voxels
    if len(junction_voxels) > 0:
        junction_mask = np.zeros(n_points, dtype=bool)
        for idx in junction_voxels:
            junction_mask[idx] = True

        # Build graph for junction voxels only
        junction_graph = [[] for _ in range(n_points)]
        for idx in junction_voxels:
            for neighbor in graph[idx]:
                if junction_mask[neighbor]:
                    junction_graph[idx].append(neighbor)

        # Find connected components of junction voxels
        visited = set()
        junction_components = []

        def dfs_junction(start_idx: int, component: list[int]) -> None:
            stack = [start_idx]
            while stack:
                node_idx = stack.pop()
                if node_idx in visited:
                    continue
                visited.add(node_idx)
                component.append(node_idx)
                for neighbor in junction_graph[node_idx]:
                    if neighbor not in visited:
                        stack.append(neighbor)

        for idx in junction_voxels:
            if idx not in visited:
                component = []
                dfs_junction(idx, component)
                junction_components.append(component)

        # Create nodes from junction components
        for i, component in enumerate(junction_components):
            for point_idx in component:
                node_idx_map[point_idx] = i
            # Calculate center of mass for the node
            comx = np.mean([x_coords[idx] for idx in component])
            comy = np.mean([y_coords[idx] for idx in component])
            comz = np.mean([z_coords[idx] for idx in component])
            nodes.append(
                {
                    "idx": component,
                    "comx": comx,
                    "comy": comy,
                    "comz": comz,
                    "ep": 0,  # Not an endpoint
                    "links": [],
                    "conn": [],
                }
            )
    else:
        junction_components = []

    # Create nodes from endpoints (each endpoint is its own node)
    num_junction_nodes = len(nodes)
    for i, point_idx in enumerate(endpoint_voxels):
        node_idx = num_junction_nodes + i
        node_idx_map[point_idx] = node_idx
        nodes.append(
            {
                "idx": [point_idx],
                "comx": x_coords[point_idx],
                "comy": y_coords[point_idx],
                "comz": z_coords[point_idx],
                "ep": 1,  # Is an endpoint
                "links": [],
                "conn": [],
            }
        )

    # Create mapping for canal voxels (points with 2 neighbors)
    canal_to_neighbors = {}
    for point_idx in canal_voxels:
        neighbors = graph[point_idx]
        canal_to_neighbors[point_idx] = neighbors

    # Follow links from each node
    links = []
    link_idx = 0
    processed_links = set()  # Track processed links to avoid duplicates

    def follow_link(
        start_node_idx: int, start_point_idx: int, direction_point_idx: int
    ) -> tuple[list[int], int, bool]:
        """Follow a link from a node until reaching another node or endpoint.

        Returns:
            (voxel_path, end_node_idx, is_endpoint)
        """
        path = [start_point_idx]
        current = direction_point_idx
        prev = start_point_idx

        while True:
            if current in node_idx_map:
                # Reached a node
                end_node_idx = node_idx_map[current]
                path.append(current)
                is_endpoint = nodes[end_node_idx]["ep"] == 1
                return path, end_node_idx, is_endpoint

            if current not in canal_to_neighbors:
                # Not a canal voxel, stop
                break

            path.append(current)

            # Get the two neighbors of this canal voxel
            neighbors = canal_to_neighbors[current]
            # Choose the neighbor that's not the previous point
            next_point = None
            for neighbor in neighbors:
                if neighbor != prev:
                    next_point = neighbor
                    break

            if next_point is None:
                break

            prev = current
            current = next_point

        # Reached end without finding a node
        return path, None, False

    # Visit all nodes and follow their links
    for node_idx, node in enumerate(nodes):
        # For junction nodes, check all voxels in the node
        # For endpoint nodes, just check the single voxel
        node_voxels = node["idx"]

        # Collect all neighbors of all voxels in this node
        all_neighbors = set()
        for voxel_idx in node_voxels:
            for neighbor in graph[voxel_idx]:
                # Only consider neighbors that are not part of this node
                if neighbor not in node_voxels:
                    all_neighbors.add(neighbor)

        for neighbor in all_neighbors:
            # Find which voxel in this node connects to the neighbor
            start_voxel = None
            for voxel_idx in node_voxels:
                if neighbor in graph[voxel_idx]:
                    start_voxel = voxel_idx
                    break

            if start_voxel is None:
                continue

            if neighbor in node_idx_map:
                # Direct connection to another node
                end_node_idx = node_idx_map[neighbor]
                if end_node_idx != node_idx:
                    # Avoid duplicate links (process each link only once)
                    link_key = tuple(sorted([node_idx, end_node_idx]))
                    if link_key not in processed_links:
                        processed_links.add(link_key)
                        link = {
                            "n1": node_idx,
                            "n2": end_node_idx,
                            "point": [start_voxel, neighbor],
                        }
                        links.append(link)
                        node["links"].append(link_idx)
                        node["conn"].append(end_node_idx)
                        nodes[end_node_idx]["links"].append(link_idx)
                        nodes[end_node_idx]["conn"].append(node_idx)
                        link_idx += 1
            elif neighbor in canal_to_neighbors:
                # Follow link through canal voxels
                path, end_node_idx, is_endpoint = follow_link(
                    node_idx, start_voxel, neighbor
                )

                if (
                    end_node_idx is not None
                    and end_node_idx != node_idx
                    and len(path) >= min_branch_length
                ):
                    # Avoid duplicate links
                    link_key = tuple(sorted([node_idx, end_node_idx]))
                    if link_key not in processed_links:
                        processed_links.add(link_key)
                        link = {"n1": node_idx, "n2": end_node_idx, "point": path}
                        links.append(link)
                        node["links"].append(link_idx)
                        node["conn"].append(end_node_idx)
                        nodes[end_node_idx]["links"].append(link_idx)
                        nodes[end_node_idx]["conn"].append(node_idx)
                        link_idx += 1

    return nodes, links


def _skeleton_to_polydata(
    skeleton_array: np.ndarray, spacing: tuple[float, float, float]
) -> vtkPolyData:
    """Convert 3D skeleton array to VTK PolyData using graph-based approach (Skel2Graph3D style).

    Uses graph structure to extract coherent centerline segments (links) between nodes.
    This produces much better centerlines than path tracing.

    Based on Matlab Skel2Graph3D.m and Vessel_morph.m

    Note: spacing is in VTK (x, y, z) order from img.GetSpacing()
    skeleton_array is in numpy (z, y, x) order
    """
    skeleton_points = np.where(skeleton_array > 0)

    if len(skeleton_points[0]) == 0:
        return vtkPolyData()

    # Get all skeleton point coordinates (z, y, x) from numpy where
    z_coords = skeleton_points[0]
    y_coords = skeleton_points[1]
    x_coords = skeleton_points[2]

    print(f"  Skeleton array shape: {skeleton_array.shape} (z, y, x)")
    print(
        f"  Skeleton point ranges: z=[{z_coords.min()}, {z_coords.max()}], y=[{y_coords.min()}, {y_coords.max()}], x=[{x_coords.min()}, {x_coords.max()}]"
    )
    print(f"  VTK spacing (x, y, z): {spacing}")

    # Create mapping from (z, y, x) to point index
    n_points = len(z_coords)
    coord_to_idx = {}
    for i in range(n_points):
        coord_to_idx[(z_coords[i], y_coords[i], x_coords[i])] = i

    # Convert to world coordinates (x, y, z) for VTK using standard axis order
    points_world = np.column_stack(
        [
            x_coords * spacing[0],  # numpy x -> world x
            y_coords * spacing[1],  # numpy y -> world y
            z_coords * spacing[2],  # numpy z -> world z
        ]
    )

    print(
        f"  World coordinate ranges (PyVista-aligned): x=[{points_world[:, 0].min():.2f}, {points_world[:, 0].max():.2f}], y=[{points_world[:, 1].min():.2f}, {points_world[:, 1].max():.2f}], z=[{points_world[:, 2].min():.2f}, {points_world[:, 2].max():.2f}]"
    )

    # Create VTK points
    vtk_points = vtk.vtkPoints()
    for point in points_world:
        vtk_points.InsertNextPoint(point)

    # Build graph: use 26-connectivity (like Matlab Skel2Graph3D)
    offsets = [
        (dz, dy, dx)
        for dz in [-1, 0, 1]
        for dy in [-1, 0, 1]
        for dx in [-1, 0, 1]
        if not (dz == 0 and dy == 0 and dx == 0)
    ]

    # Build adjacency list (graph representation)
    graph = [[] for _ in range(n_points)]
    for i in range(n_points):
        z, y, x = z_coords[i], y_coords[i], x_coords[i]

        for dz, dy, dx in offsets:
            nz, ny, nx = z + dz, y + dy, x + dx

            if (
                nz < 0
                or nz >= skeleton_array.shape[0]
                or ny < 0
                or ny >= skeleton_array.shape[1]
                or nx < 0
                or nx >= skeleton_array.shape[2]
            ):
                continue

            if skeleton_array[nz, ny, nx] > 0:
                neighbor_idx = coord_to_idx.get((nz, ny, nx))
                if neighbor_idx is not None:
                    graph[i].append(neighbor_idx)

    # Convert skeleton to graph structure (like Skel2Graph3D)
    print("  Converting skeleton to graph structure (Skel2Graph3D style)...")
    min_branch_length = 5  # Minimum branch length in voxels
    nodes, links = _skel2graph3d(
        skeleton_array,
        z_coords,
        y_coords,
        x_coords,
        coord_to_idx,
        graph,
        min_branch_length,
    )

    print(f"  Graph structure: {len(nodes)} nodes, {len(links)} links")

    # Extract centerlines from links (each link.point is a centerline segment)
    lines = vtk.vtkCellArray()
    for link in links:
        point_indices = link["point"]
        if len(point_indices) > 1:
            lines.InsertNextCell(len(point_indices))
            for point_idx in point_indices:
                lines.InsertCellPoint(point_idx)

    # Create polydata
    polydata = vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetLines(lines)

    num_lines = lines.GetNumberOfCells()
    num_nodes = len(nodes)
    num_endpoints = sum(1 for node in nodes if node["ep"] == 1)
    print(
        f"  Created {num_lines} centerline segments from {num_nodes} nodes ({num_endpoints} endpoints)"
    )

    return polydata


def _numpy_to_vtk_image(
    vol_zyx: np.ndarray, spacing: tuple[float, float, float]
) -> vtkImageData:
    """Convert numpy array to VTK ImageData."""
    nz, ny, nx = map(int, vol_zyx.shape)
    vol_xyz = np.transpose(vol_zyx, (2, 1, 0)).copy(order="F")  # (x, y, z)

    img = vtkImageData()
    img.SetDimensions(nx, ny, nz)
    img.SetSpacing(spacing[2], spacing[1], spacing[0])  # VTK uses x,y,z order
    img.SetOrigin(0.0, 0.0, 0.0)
    scalars = vtknp.numpy_to_vtk(
        vol_xyz.ravel(order="F"),
        deep=True,
        array_type=vtkTypeFloat32Array().GetDataType(),
    )
    scalars.SetName("Scalars")
    img.GetPointData().SetScalars(scalars)
    return img


def _load_and_binarize_image(
    ipath: Path,
    threshold: float,
    extract_label: int | None = None,
    npy_channel: int = 1,
) -> tuple[vtkImageData, np.ndarray | None]:
    """Load segmentation, apply thresholding, and return binary VTK image."""
    original_volume: np.ndarray | None = None

    if ipath.suffix.lower() == ".npy":
        data = np.load(str(ipath))
        print(f"Loaded .npy data shape: {data.shape}")
        spacing_zyx = (1.0, 1.0, 1.0)

        if extract_label is not None:
            if data.ndim == NUMPY_4D_DIMENSION:
                print(
                    "Detected multi-channel NumPy volume; collapsing via argmax for label extraction."
                )
                label_volume = np.argmax(data, axis=0).astype(np.int16, copy=False)
            elif data.ndim == DIMENSIONS_3D:
                label_volume = data.astype(np.int16, copy=False)
            else:
                raise ValueError(
                    f"Expected 3D or channel-first 4D numpy array, got shape {data.shape}"
                )

            unique_vals = np.unique(label_volume)
            if extract_label not in unique_vals:
                raise ValueError(
                    f"Requested label {extract_label} not present in NumPy volume. Unique values: {unique_vals[:10]}"
                )
            mask = (label_volume == int(extract_label)).astype(np.float32, copy=False)
            voxels = int(mask.sum())
            print(
                f"Extracted label {extract_label}: {voxels} voxels ({voxels / mask.size * 100:.4f}% of volume)"
            )
            original_volume = _ensure_zyx(mask, f"label_{extract_label}_mask")
            print(f"  Original volume shape (z,y,x): {original_volume.shape}")
            binary_volume = original_volume
        else:
            if data.ndim == NUMPY_4D_DIMENSION:
                if npy_channel < 0 or npy_channel >= data.shape[0]:
                    raise ValueError(
                        f"Requested channel {npy_channel} but input only has {data.shape[0]} channels."
                    )
                print(
                    f"Detected multi-channel NumPy volume; extracting channel {npy_channel}."
                )
                data = data[npy_channel]
            elif data.ndim != DIMENSIONS_3D:
                raise ValueError(
                    f"Expected 3D or channel-first 4D numpy array, got shape {data.shape}"
                )
            original_volume = _ensure_zyx(
                data.astype(np.float32, copy=False),
                "original_volume",
            )
            print(f"  Original volume shape (z,y,x): {original_volume.shape}")
            binary_volume = (original_volume >= threshold).astype(
                np.float32, copy=False
            )
            print(f"  Binary volume shape (z,y,x): {binary_volume.shape}")

        img = _numpy_to_vtk_image(binary_volume, spacing_zyx)
        return img, original_volume

    if ipath.name.lower().endswith(".nii.gz") or ipath.suffix.lower() == ".nii":
        reader = vtkNIFTIImageReader()
        reader.SetFileName(str(ipath))
        reader.Update()
        vtk_img = reader.GetOutput()
        scalars = vtk_img.GetPointData().GetScalars()
        if scalars is None:
            raise ValueError("NIfTI file does not contain scalar data")
        dims = vtk_img.GetDimensions()
        data = vtknp.vtk_to_numpy(scalars).reshape((dims[2], dims[1], dims[0]))
        original_volume = data.astype(np.float32, copy=False)
        binary_volume = (original_volume >= threshold).astype(np.float32, copy=False)
        spacing_xyz = vtk_img.GetSpacing()
        spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
        img = _numpy_to_vtk_image(binary_volume, spacing_zyx)
        return img, original_volume

    if ipath.suffix.lower() == ".nrrd":
        import nrrd

        data, header = nrrd.read(str(ipath))
        if data.ndim != DIMENSIONS_3D:
            raise ValueError(f"Expected 3D NRRD, got shape {tuple(data.shape)}")

        unique_values = np.unique(data)
        if unique_values.size > MULTI_LABEL_THRESHOLD and extract_label is None:
            raise ValueError(
                "Multi-label NRRD detected; specify --extract-label to choose the vessel label."
            )

        if extract_label is not None:
            data = (data == extract_label).astype(np.float32)
            print(f"Extracted label {extract_label}: {np.count_nonzero(data)} voxels")
        else:
            data = data.astype(np.float32, copy=False)

        vol_zyx = (
            rearrange(data, "x y z -> z y x")
            if header.get("dimension") == DIMENSIONS_3D
            else data
        )
        original_volume = vol_zyx.astype(np.float32, copy=False)
        binary_volume = (original_volume >= threshold).astype(np.float32, copy=False)
        sdirs = header.get("space directions")
        if sdirs is None:
            raise ValueError("Missing 'space directions' in NRRD header")
        spacing_xyz = tuple(float(np.linalg.norm(np.asarray(v))) for v in sdirs)
        spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
        img = _numpy_to_vtk_image(binary_volume, spacing_zyx)
        return img, original_volume

    raise ValueError(f"Unsupported input format: {ipath}")


def _write_polydata(poly: vtkPolyData, out_path: Path) -> None:
    """Write PolyData to file."""
    sx = out_path.suffix.lower()
    if sx == ".vtp":
        w = vtkXMLPolyDataWriter()
        w.SetFileName(str(out_path))
        w.SetInputData(poly)
        w.SetDataModeToBinary()
        w.Write()
    elif sx == ".vtk":
        w = vtkPolyDataWriter()
        w.SetFileName(str(out_path))
        w.SetInputData(poly)
        w.SetFileTypeToBinary()
        w.Write()
    else:
        raise ValueError(f"unsupported output extension: {out_path.suffix}")


def extract_adaptive_centerlines(
    input_segmentation_path: str,
    output_centerline_path: str,
    *,
    binarize_threshold: float = THRESHOLD_DEFAULT,
    max_connection_distance_mm: float = DEFAULT_MAX_CONNECTION_DISTANCE_MM,
    use_island_connection: bool = True,
    enable_visualizations: bool = True,
    extract_label: int | None = None,
    npy_channel: int = 1,
) -> None:
    """Extract graph-based centerlines via the streamlined skeleton pipeline."""
    ipath = Path(input_segmentation_path)
    opath = Path(output_centerline_path)

    viz_dir = _create_visualization_dir(opath) if enable_visualizations else None

    print(f"Loading input from {ipath} (threshold={binarize_threshold})...")
    img, original_volume = _load_and_binarize_image(
        ipath,
        binarize_threshold,
        extract_label,
        npy_channel=npy_channel,
    )
    if enable_visualizations and viz_dir:
        print("Creating 3D vessel visualization...")
        if original_volume is not None:
            _visualize_3d_vessels_from_numpy(original_volume, viz_dir)
        else:
            _visualize_3d_vessels(img, viz_dir)

        if extract_label is not None and original_volume is not None:
            _visualize_intermediate_stage(
                original_volume,
                viz_dir,
                f"selected_label_{extract_label}",
                f"Label {extract_label} selection mask",
            )

    print("Using skeletonization + graph extraction pipeline...")
    centerlines = _extract_skeleton_centerlines(
        img,
        use_island_connection=use_island_connection,
        max_connection_distance_mm=max_connection_distance_mm,
        viz_dir=viz_dir if enable_visualizations else None,
    )

    if enable_visualizations and viz_dir:
        print("Creating final centerlines visualization...")
        _visualize_centerlines(centerlines, viz_dir)

    _write_polydata(centerlines, opath)

    num_points = centerlines.GetNumberOfPoints()
    if num_points > 0:
        print(f"✅ Successfully extracted {num_points} centerline points")
        if enable_visualizations and viz_dir:
            print(f"\n📊 Visualizations saved to: {viz_dir}")
    else:
        print("\n" + "=" * 70)
        print("⚠️  CENTERLINE EXTRACTION FAILED")
        print("=" * 70)
        print(
            "No centerlines extracted; check binarization threshold and connectivity settings."
        )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Graph-based skeleton centerline extraction with optional island connection"
    )
    p.add_argument("input", help="Input .nii.gz/.nrrd/.npy file")
    p.add_argument("output", help="Output .vtp or .vtk file")
    p.add_argument(
        "--binarize-threshold",
        type=float,
        default=THRESHOLD_DEFAULT,
        help=f"Threshold applied before skeletonization (default: {THRESHOLD_DEFAULT})",
    )
    p.add_argument(
        "--max-connection-distance-mm",
        type=float,
        default=DEFAULT_MAX_CONNECTION_DISTANCE_MM,
        help=f"Maximum distance to connect skeleton islands in mm (default: {DEFAULT_MAX_CONNECTION_DISTANCE_MM})",
    )
    p.add_argument(
        "--no-island-connection",
        action="store_true",
        help="Disable the island-connection step (uses raw skeleton)",
    )
    p.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Disable MP4 visualization generation",
    )
    p.add_argument(
        "--extract-label",
        type=int,
        default=1,
        help="Label id to extract from multi-label inputs (default: 1 for vessels)",
    )
    p.add_argument(
        "--npy-channel",
        type=int,
        default=1,
        help="Channel index to use when loading 4D NumPy arrays (default: 1 for vessels)",
    )
    args = p.parse_args()

    extract_adaptive_centerlines(
        args.input,
        args.output,
        binarize_threshold=args.binarize_threshold,
        max_connection_distance_mm=args.max_connection_distance_mm,
        use_island_connection=not args.no_island_connection,
        enable_visualizations=not args.no_visualizations,
        extract_label=args.extract_label,
        npy_channel=args.npy_channel,
    )
