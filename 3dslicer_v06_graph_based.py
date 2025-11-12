"""Adaptive centerline extraction with graph-based approach (v06)

This version incorporates ideas from Matlab Skel2Graph3D:
- Graph-based centerline extraction (nodes and links structure)
- Each link represents a coherent centerline segment between nodes
- Extracts centerlines from link.point arrays (like Matlab)
- Handles junctions naturally by splitting into multiple links
- Produces coherent centerlines instead of fragmented paths

Also includes v05 optimizations:
- Batch connection strategy for island connection
- Vectorized operations
- Union-Find data structure

Based on Matlab code: Skel2Graph3D.m, Conn_nearest_points_v2.m, Vessel_morph.m

Dependencies (conda-forge):
  - python==3.11
  - vmtk==1.5.0
  - vtk>=9.2
  - nibabel, nrrd, einops
  - scikit-image (for skeletonization)
  - scipy (for k-nearest neighbor search and convolution)
"""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import binary_dilation

# VTK imports
import vtk
from einops import rearrange
from skimage.measure import label
from skimage.morphology import binary_dilation as sk_binary_dilation, binary_erosion, skeletonize
from vmtk import vmtkscripts
from vtkmodules.util import numpy_support as vtknp
from vtkmodules.vtkCommonCore import vtkTypeFloat32Array
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPolyData
from vtkmodules.vtkFiltersCore import (
    vtkCleanPolyData,
    vtkDecimatePro,
    vtkFlyingEdges3D,
    vtkPolyDataConnectivityFilter,
    vtkTriangleFilter,
)
from vtkmodules.vtkIOImage import vtkNIFTIImageReader
from vtkmodules.vtkIOLegacy import vtkPolyDataWriter
from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter

# Constants
DIMENSIONS_3D = 3
THRESHOLD_DEFAULT = 0.5
SPARSITY_THRESHOLD_HIGH = 0.95
SPARSITY_THRESHOLD_LOW = 0.1
COMPONENTS_THRESHOLD = 100

# Visualization optimization settings
VIZ_N_FRAMES = 60  # Reduced from 120 for faster generation
VIZ_FRAMERATE = 15
VIZ_WINDOW_SIZE = [1280, 720]  # Reduced from 1920x1080 for faster rendering

# Default connection parameters (from Matlab code)
DEFAULT_MAX_CONNECTION_DISTANCE_MM = 15.0  # Maximum distance to connect islands (mm)
DEFAULT_MIN_BRANCH_LENGTH_MM = 10.0  # Minimum branch length to keep (mm)

# PyVista for 3D vessel visualization
# IMPORTANT: All visualizations are configured for headless/offscreen rendering
# This is safe for SLURM compute nodes - no windows will pop up
# Based on working approach from visualize_all_labels_3d.py

# Detect if we're running without a DISPLAY (remote/headless/compute node)
NO_DISPLAY = not os.environ.get("DISPLAY")

# Force offscreen rendering for remote servers (set BEFORE importing pyvista)
if NO_DISPLAY:
    os.environ.setdefault('PYVISTA_OFF_SCREEN', 'true')
    os.environ.setdefault('PYVISTA_USE_PANEL', 'false')
    os.environ.setdefault('MESA_GL_VERSION_OVERRIDE', '3.3')
    os.environ.setdefault('MESA_GLSL_VERSION_OVERRIDE', '330')
    os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE', '1')
    os.environ.setdefault('VTK_USE_X', '0')
    os.environ.setdefault('DISPLAY', '')
    print("No DISPLAY detected; enabling offscreen mode for compute nodes")
else:
    # Even with DISPLAY, use offscreen for video generation
    os.environ.setdefault('PYVISTA_OFF_SCREEN', 'true')
    os.environ.setdefault('PYVISTA_USE_PANEL', 'false')
    print("DISPLAY detected; using offscreen mode for video generation")

try:
    import pyvista as pv
    
    # Always use offscreen for video generation
    pv.OFF_SCREEN = True
    
    # Try to start Xvfb if available and no DISPLAY (like visualize_all_labels_3d.py)
    if NO_DISPLAY and hasattr(pv, "start_xvfb"):
        try:
            pv.start_xvfb()
            print("Started Xvfb for offscreen rendering")
        except Exception as xvfb_err:
            print(f"Note: Could not start Xvfb automatically: {xvfb_err}")
            print("  If rendering fails, install vtk-osmesa or run inside xvfb-run")
    
    # Try to reduce rendering overhead
    try:
        if hasattr(pv.global_theme, 'anti_aliasing'):
            pv.global_theme.anti_aliasing = False
    except (AttributeError, Exception):
        pass  # Ignore if attribute doesn't exist or can't be set
    
    # Suppress VTK render window warnings
    try:
        vtk.vtkRenderWindow.SetGlobalWarningDisplay(0)
    except Exception:
        pass
    
    # Set theme (optional, won't affect headless rendering)
    try:
        pv.set_plot_theme("dark")
    except Exception:
        pass
    
    PYVISTA_AVAILABLE = True
    print("PyVista loaded with offscreen rendering enabled")
except ImportError:
    PYVISTA_AVAILABLE = False
    print("Warning: PyVista not available. 3D vessel visualization will be skipped.")

__all__ = ["extract_adaptive_centerlines"]


class UnionFind:
    """Union-Find (Disjoint Set) data structure for tracking component merges."""
    
    def __init__(self, n: int):
        """Initialize with n elements."""
        self.parent = np.arange(n, dtype=np.int32)
        self.rank = np.zeros(n, dtype=np.int32)
    
    def find(self, x: int) -> int:
        """Find root of x with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
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
    
    def get_components(self) -> dict[int, set[int]]:
        """Get all components as a dictionary mapping root to set of elements."""
        components = {}
        for i in range(len(self.parent)):
            root = self.find(i)
            if root not in components:
                components[root] = set()
            components[root].add(i)
        return components


def _check_display_available() -> bool:
    """Check if display is available for visualizations."""
    return PYVISTA_AVAILABLE


def _check_mp4_progress(viz_dir: Path, expected_files: list[tuple[str, str]]) -> dict:
    """Check progress of MP4 file creation.
    
    Args:
        viz_dir: Visualization directory
        expected_files: List of (step_name, filename) tuples
        
    Returns:
        Dictionary with status information
    """
    results = {
        "total": len(expected_files),
        "created": 0,
        "missing": [],
        "files": {}
    }
    
    for step_name, filename in expected_files:
        filepath = viz_dir / filename
        if filepath.exists():
            file_size = filepath.stat().st_size / (1024 * 1024)  # MB
            # Check if file is valid (has reasonable size > 0)
            if file_size > 0:
                results["created"] += 1
                results["files"][step_name] = {
                    "path": str(filepath),
                    "size_mb": file_size,
                    "status": "✓ Created"
                }
            else:
                results["missing"].append(step_name)
                results["files"][step_name] = {
                    "path": str(filepath),
                    "size_mb": 0,
                    "status": "⚠ Empty file"
                }
        else:
            results["missing"].append(step_name)
            results["files"][step_name] = {
                "path": str(filepath),
                "size_mb": 0,
                "status": "✗ Missing"
            }
    
    return results


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
    neighbor_count = correlate(skeleton.astype(np.float32), kernel, mode='constant', cval=0.0)
    
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
        (points_int[:, 0] >= 0) & (points_int[:, 0] < skel.shape[2]) &
        (points_int[:, 1] >= 0) & (points_int[:, 1] < skel.shape[1]) &
        (points_int[:, 2] >= 0) & (points_int[:, 2] < skel.shape[0])
    )
    
    # Set skeleton points (convert x,y,z to z,y,x for array indexing)
    valid_points = points_int[valid_mask]
    if len(valid_points) > 0:
        skel[valid_points[:, 2], valid_points[:, 1], valid_points[:, 0]] = 1


def _connect_nearest_islands_optimized(
    skeleton: np.ndarray,
    spacing: tuple[float, float, float],
    max_distance_mm: float = DEFAULT_MAX_CONNECTION_DISTANCE_MM,
) -> np.ndarray:
    """Connect sparse islands in skeleton using optimized batch connection method.
    
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
    print(f"Connecting sparse islands (optimized v05, max distance: {max_distance_mm} mm)...")
    
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
    
    # Initialize Union-Find for tracking component merges
    # Map component labels to union-find indices
    label_to_idx = {lab: idx for idx, lab in enumerate(unique_labels)}
    uf = UnionFind(num_components)
    
    iteration = 0
    max_iterations = min(num_components * 2, 1000)  # Safety limit (reduced from num_components)
    relabel_interval = max(10, num_components // 100)  # Relabel every N connections
    
    # Iterate until all components connected or max distance exceeded
    while num_components > 1 and iteration < max_iterations:
        iteration += 1
        
        # Relabel periodically to update component structure
        if iteration % relabel_interval == 0 or iteration == 1:
            labeled = label(skel, connectivity=3)
            unique_labels = np.unique(labeled)
            unique_labels = unique_labels[unique_labels > 0]
            num_components = len(unique_labels)
            
            # Rebuild union-find mapping
            label_to_idx = {lab: idx for idx, lab in enumerate(unique_labels)}
            uf = UnionFind(num_components)
        
        if num_components <= 1:
            break
        
        print(f"  Iteration {iteration}: {num_components} components remaining")
        
        # Precompute component sizes using np.bincount (vectorized, O(n))
        component_sizes = np.bincount(labeled.ravel())
        component_sizes = component_sizes[unique_labels]  # Only non-zero labels
        
        # Get component size pairs (size, label) and sort by size (smallest first)
        size_label_pairs = [(component_sizes[i], lab) for i, lab in enumerate(unique_labels)]
        size_label_pairs.sort()
        
        # Precompute ALL endpoints once (vectorized)
        all_endpoints_zyx = _find_skeleton_endpoints_vectorized(skel)
        
        if len(all_endpoints_zyx) == 0:
            print("    No endpoints found, stopping")
            break
        
        # Group endpoints by component label
        endpoint_labels = labeled[all_endpoints_zyx[:, 0], all_endpoints_zyx[:, 1], all_endpoints_zyx[:, 2]]
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
            
            # Find closest connection for this component
            best_distance = float('inf')
            best_connection = None
            
            for ep_z, ep_y, ep_x in target_endpoints:
                # Convert endpoint to (x, y, z) and scale
                ep_coord = np.array([ep_x, ep_y, ep_z], dtype=np.float64)
                ep_coord_scaled = ep_coord.copy()
                ep_coord_scaled[0] *= ratio[0]
                ep_coord_scaled[1] *= ratio[1]
                ep_coord_scaled[2] *= ratio[2]
                
                # Query nearest neighbor (excluding points from same component)
                # Use query_ball_point to find all points within max_distance
                candidate_indices = global_tree.query_ball_point(
                    ep_coord_scaled, 
                    r=max_distance_voxels
                )
                
                if len(candidate_indices) == 0:
                    continue
                
                # Filter out points from the same component
                candidate_coords = all_coords_xyz[candidate_indices]
                candidate_labels = labeled[
                    candidate_coords[:, 2],  # z
                    candidate_coords[:, 1],  # y
                    candidate_coords[:, 0]   # x
                ]
                
                # Find closest point from different component
                different_component_mask = candidate_labels != target_label
                if not np.any(different_component_mask):
                    continue
                
                # Get distances to filtered candidates
                valid_indices = np.array(candidate_indices)[different_component_mask]
                valid_coords_scaled = all_coords_scaled[valid_indices]
                
                # Compute distances
                distances = np.linalg.norm(
                    valid_coords_scaled - ep_coord_scaled[None, :],
                    axis=1
                )
                
                min_idx = np.argmin(distances)
                min_dist = distances[min_idx]
                
                if min_dist < best_distance and min_dist <= max_distance_voxels:
                    best_distance = min_dist
                    best_connection = (
                        (ep_z, ep_y, ep_x),
                        tuple(candidate_coords[different_component_mask][min_idx]),
                        min_dist
                    )
            
            if best_connection is not None:
                connections_to_make.append((target_label, best_connection))
        
        # Make all connections found in this iteration
        if len(connections_to_make) == 0:
            print(f"    No more connections possible (distance > {max_distance_voxels:.2f} voxels)")
            break
        
        # Sort connections by distance (connect closest first)
        connections_to_make.sort(key=lambda x: x[1][2])  # Sort by distance
        
        connections_made = 0
        for target_label, ((ep_z, ep_y, ep_x), (target_x, target_y, target_z), dist) in connections_to_make:
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
            if uf.find(target_idx) == uf.find(target_point_idx):
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
    
    # Final relabel to get accurate count
    labeled = label(skel, connectivity=3)
    final_components = len(np.unique(labeled)) - 1
    print(f"  Final: {final_components} connected component(s)")
    
    return skel


# Alias for backward compatibility
_connect_nearest_islands = _connect_nearest_islands_optimized


def _preprocess_vessel_data_with_island_connection(
    img: vtkImageData,
    max_connection_distance_mm: float = DEFAULT_MAX_CONNECTION_DISTANCE_MM,
    viz_dir: Path | None = None,
) -> vtkImageData:
    """Preprocess sparse vessel data by connecting islands before VMTK extraction.
    
    This extends the basic preprocessing with optimized island connection:
    1. Extract skeleton
    2. Connect sparse islands using optimized batch method (v05)
    3. Dilate connected skeleton back to vessel thickness
    
    Args:
        img: Input VTK ImageData
        max_connection_distance_mm: Maximum distance to connect islands (mm)
        viz_dir: Optional visualization directory for step-by-step visualizations
        
    Returns:
        Preprocessed VTK ImageData
    """
    scalars = img.GetPointData().GetScalars()
    data = vtknp.vtk_to_numpy(scalars)
    
    # Reshape to 3D
    dims = img.GetDimensions()
    binary_data = (
        (data > THRESHOLD_DEFAULT).astype(np.uint8).reshape((dims[2], dims[1], dims[0]))
    )
    
    print("Step 1: Extracting skeleton...")
    # Extract skeleton first
    skeleton = skeletonize(binary_data)
    print(f"  Skeleton points: {np.count_nonzero(skeleton)}")
    
    # Visualize skeleton before connection
    if viz_dir:
        try:
            _visualize_intermediate_stage(
                skeleton.astype(np.float32), viz_dir, "skeleton_before_connection",
                "Skeleton before island connection"
            )
        except Exception as e:
            print(f"Warning: Skeleton visualization failed: {e}")
    
    # Get spacing (VTK uses x,y,z, we need z,y,x)
    spacing_xyz = img.GetSpacing()
    spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    
    print("Step 2: Connecting sparse islands (optimized v05)...")
    # Connect islands using optimized method
    connected_skeleton = _connect_nearest_islands_optimized(
        skeleton, spacing_zyx, max_connection_distance_mm
    )
    
    # Visualize skeleton after connection
    if viz_dir:
        try:
            _visualize_intermediate_stage(
                connected_skeleton.astype(np.float32), viz_dir, "skeleton_after_connection",
                "Skeleton after island connection"
            )
        except Exception as e:
            print(f"Warning: Connected skeleton visualization failed: {e}")
    
    print("Step 3: Dilating connected skeleton to restore vessel thickness...")
    # Dilate to restore vessel thickness (similar to original preprocessing)
    dilated = sk_binary_dilation(connected_skeleton, footprint=np.ones((3, 3, 3)))
    
    # Check if dilated skeleton has enough voxels
    dilated_voxels = np.count_nonzero(dilated)
    total_voxels = dilated.size
    print(f"  Dilated skeleton: {dilated_voxels} voxels ({100*dilated_voxels/total_voxels:.4f}% of volume)")
    
    if dilated_voxels == 0:
        print("  ⚠️  WARNING: Dilated skeleton is empty - preprocessing may have failed")
    elif dilated_voxels < 100:
        print("  ⚠️  WARNING: Very few voxels in dilated skeleton - VMTK may fail")
    
    # Visualize final dilated result
    if viz_dir:
        try:
            _visualize_intermediate_stage(
                dilated.astype(np.float32), viz_dir, "skeleton_dilated",
                "Dilated skeleton (final preprocessing result)"
            )
        except Exception as e:
            print(f"Warning: Dilated skeleton visualization failed: {e}")
    
    # Convert back to VTK
    return _numpy_to_vtk_image(dilated.astype(np.float32), spacing_zyx)


# Import all visualization and other functions from v04 (they remain the same)
# For brevity, I'll include the key ones that are needed

def _visualize_3d_vessels_from_numpy(vessel_data: np.ndarray, viz_dir: Path) -> None:
    """Create 3D vessel visualization directly from numpy array (notebook style).
    
    This matches visualize_vessels_notebook_style.py exactly:
    - Directly wraps numpy array with pv.wrap()
    - No VTK conversion/reshaping that could cause issues
    """
    if not PYVISTA_AVAILABLE:
        print("Skipping 3D vessel visualization (PyVista not available)")
        return

    print("Creating 3D vessel visualization (notebook style - direct numpy)...")
    print(f"Data shape: {vessel_data.shape}")
    
    data_min, data_max = np.min(vessel_data), np.max(vessel_data)
    print(f"Data range: [{data_min:.4f}, {data_max:.4f}]")
    
    vessel_voxels = np.count_nonzero(vessel_data > 0)
    print(f"Vessel voxels: {vessel_voxels} ({100*vessel_voxels/vessel_data.size:.2f}% of volume)")

    try:
        # Directly wrap numpy array like visualize_vessels_notebook_style.py
        grid = pv.wrap(vessel_data.astype(np.float32))

        # Create plotter (off-screen for headless rendering)
        try:
            plotter = pv.Plotter(off_screen=True, window_size=VIZ_WINDOW_SIZE)
        except (SystemExit, KeyboardInterrupt):
            raise
        except Exception as plotter_error:
            error_msg = str(plotter_error).lower()
            print(f"⚠ Warning: Failed to create PyVista plotter: {plotter_error}")
            
            if "x server" in error_msg or "display" in error_msg or "bad x server connection" in error_msg:
                print("\n" + "="*60)
                print("OFFSCREEN RENDERING ERROR:")
                print("="*60)
                print("This script requires offscreen rendering for compute nodes.")
                print("Two common fixes:")
                print("  1) Launch the script under a virtual framebuffer:")
                print("       xvfb-run -s '-screen 0 1920x1080x24' python 3dslicer_v06_graph_based.py ...")
                print("  2) Install vtk-osmesa to enable pure offscreen rendering:")
                print("     conda install -c conda-forge vtk-osmesa")
                print("  3) Use --no-visualizations to skip all visualizations")
                print("="*60)
            
            print("  Continuing without visualization...")
            return

        # Use volume rendering with linear opacity (matching notebook exactly)
        print("  Using volume rendering with linear opacity and 'Blues' colormap (notebook style)")
        plotter.add_volume(grid, opacity="linear", cmap="Blues")

        # Set background to black for better contrast
        plotter.background_color = "black"
        plotter.show_axes()

        # Generate rotating video
        output_path = viz_dir / "vessels_3d.mp4"
        print(f"Saving 3D vessel visualization to: {output_path}")

        try:
            print(f"  Generating {VIZ_N_FRAMES} frame rotating video (framerate={VIZ_FRAMERATE})...")
            plotter.open_movie(str(output_path), framerate=VIZ_FRAMERATE)
            n_frames = VIZ_N_FRAMES
            progress_interval = max(1, n_frames // 4)
            
            # Precompute all camera angles vectorized (optimized)
            azimuth_angles = 180 + np.arange(n_frames, dtype=np.float32) * 360.0 / n_frames
            
            # Pre-set constant camera properties once
            plotter.camera_position = "yz"
            plotter.camera.elevation = 30

            # Optimized frame generation loop
            for i in range(n_frames):
                plotter.camera.azimuth = azimuth_angles[i]
                plotter.render()
                plotter.write_frame()
                
                # Progress reporting (only when needed)
                if (i + 1) % progress_interval == 0 or i == n_frames - 1:
                    print(f"    Progress: {i + 1}/{n_frames} frames ({100*(i+1)/n_frames:.1f}%)")

            plotter.close()
            
            # Verify file was created
            if output_path.exists() and output_path.stat().st_size > 0:
                file_size = output_path.stat().st_size / (1024 * 1024)
                print(f"✓ 3D vessel visualization saved: {output_path} ({file_size:.2f} MB)")
            else:
                print(f"⚠ Warning: MP4 file created but appears empty or missing")
        except (SystemExit, KeyboardInterrupt):
            raise
        except Exception as movie_error:
            print(f"⚠ Warning: Movie generation failed (continuing without visualization): {movie_error}")
            import traceback
            traceback.print_exc()
            try:
                plotter.close()
            except:
                pass
            return

    except (SystemExit, KeyboardInterrupt):
        raise
    except Exception as e:
        print(f"⚠ Warning: 3D visualization failed (continuing without visualization): {e}")
        import traceback
        traceback.print_exc()
        return


def _visualize_3d_vessels(img: vtkImageData, viz_dir: Path) -> None:
    """Create a 3D rotating volume visualization of VESSEL data only."""
    if not PYVISTA_AVAILABLE:
        print("Skipping 3D vessel visualization (PyVista not available)")
        return

    scalars = img.GetPointData().GetScalars()
    if not scalars:
        print("No scalars data available for 3D visualization")
        return

    try:
        data = vtknp.vtk_to_numpy(scalars)
        dims = img.GetDimensions()

        # Reshape to 3D
        if len(data.shape) == 1:
            data_3d = data.reshape((dims[2], dims[1], dims[0]))
        else:
            data_3d = data

        print("Creating 3D vessel visualization (VESSELS ONLY - no fibroglandular or background)...")
        print(f"Data shape: {data_3d.shape}")
        
        # Vectorized data analysis and conversion
        data_min, data_max = np.min(data_3d), np.max(data_3d)
        print(f"Data range: [{data_min:.4f}, {data_max:.4f}]")
        
        # Check if data is probability (0-1 range) or binary (0/1) - vectorized check
        is_probability = (data_max <= 1.0 and data_min >= 0.0 and 
                         not np.allclose(data_3d, data_3d.astype(bool)))
        
        # Use probability values directly for volume rendering (matching notebook approach)
        vessel_data = data_3d.astype(np.float32)
        if is_probability:
            print("  Detected probability data - using volume rendering with probability values (notebook style)")
        else:
            print("  Detected binary data - converting to float for volume rendering")
        
        # Vectorized voxel counting (match notebook style: count any value > 0)
        vessel_voxels = np.count_nonzero(vessel_data > 0)
        print(f"Vessel voxels: {vessel_voxels} ({100*vessel_voxels/vessel_data.size:.2f}% of volume)")
    except Exception as e:
        print(f"Warning: Failed to prepare data for visualization: {e}")
        return

    try:
        # Convert to PyVista grid
        grid = pv.wrap(vessel_data)

        # Create plotter (off-screen for headless rendering)
        try:
            plotter = pv.Plotter(off_screen=True, window_size=VIZ_WINDOW_SIZE)
        except (SystemExit, KeyboardInterrupt):
            raise
        except Exception as plotter_error:
            error_msg = str(plotter_error).lower()
            print(f"⚠ Warning: Failed to create PyVista plotter: {plotter_error}")
            
            if "x server" in error_msg or "display" in error_msg or "bad x server connection" in error_msg:
                print("\n" + "="*60)
                print("OFFSCREEN RENDERING ERROR:")
                print("="*60)
                print("This script requires offscreen rendering for compute nodes.")
                print("Two common fixes:")
                print("  1) Launch the script under a virtual framebuffer:")
                print("       xvfb-run -s '-screen 0 1920x1080x24' python 3dslicer_v05_optimized_island_connection.py ...")
                print("  2) Install vtk-osmesa to enable pure offscreen rendering:")
                print("     conda install -c conda-forge vtk-osmesa")
                print("  3) Use --no-visualizations to skip all visualizations")
                print("="*60)
            
            print("  Continuing without visualization...")
            return

        # Use volume rendering with linear opacity (matching notebook exactly)
        print("  Using volume rendering with linear opacity and 'Blues' colormap (notebook style)")
        plotter.add_volume(grid, opacity="linear", cmap="Blues")

        # Set background to black for better contrast
        plotter.background_color = "black"
        plotter.show_axes()

        # Generate rotating video
        output_path = viz_dir / "vessels_3d.mp4"
        print(f"Saving 3D vessel visualization to: {output_path}")

        try:
            print(f"  Generating {VIZ_N_FRAMES} frame rotating video (framerate={VIZ_FRAMERATE})...")
            plotter.open_movie(str(output_path), framerate=VIZ_FRAMERATE)
            n_frames = VIZ_N_FRAMES
            progress_interval = max(1, n_frames // 4)
            
            # Precompute all camera angles vectorized (optimized)
            azimuth_angles = 180 + np.arange(n_frames, dtype=np.float32) * 360.0 / n_frames
            
            # Pre-set constant camera properties once
            plotter.camera_position = "yz"
            plotter.camera.elevation = 30

            # Optimized frame generation loop
            for i in range(n_frames):
                plotter.camera.azimuth = azimuth_angles[i]
                plotter.render()
                plotter.write_frame()
                
                # Progress reporting (only when needed)
                if (i + 1) % progress_interval == 0 or i == n_frames - 1:
                    print(f"    Progress: {i + 1}/{n_frames} frames ({100*(i+1)/n_frames:.1f}%)")

            plotter.close()
            
            # Verify file was created
            if output_path.exists() and output_path.stat().st_size > 0:
                file_size = output_path.stat().st_size / (1024 * 1024)
                print(f"✓ 3D vessel visualization saved: {output_path} ({file_size:.2f} MB)")
            else:
                print(f"⚠ Warning: MP4 file created but appears empty or missing")
        except (SystemExit, KeyboardInterrupt):
            raise
        except Exception as movie_error:
            print(f"⚠ Warning: Movie generation failed (continuing without visualization): {movie_error}")
            import traceback
            traceback.print_exc()
            try:
                plotter.close()
            except:
                pass
            return

    except (SystemExit, KeyboardInterrupt):
        raise
    except Exception as e:
        print(f"⚠ Warning: 3D visualization failed (continuing without visualization): {e}")
        import traceback
        traceback.print_exc()
        return


def _visualize_intermediate_stage(
    data: np.ndarray, viz_dir: Path, stage: str, description: str = ""
) -> None:
    """Visualize intermediate processing stages as MP4."""
    if not PYVISTA_AVAILABLE:
        print("Skipping intermediate visualization (PyVista not available)")
        return

    if len(data.shape) != DIMENSIONS_3D:
        print(f"Expected 3D data, got shape {data.shape}")
        return

    print(f"Creating {stage} 3D visualization...")
    print(f"Data shape: {data.shape}")
    
    # Vectorized data analysis and conversion
    data_min, data_max = np.min(data), np.max(data)
    print(f"Data range: [{data_min:.4f}, {data_max:.4f}]")
    
    visualization_data = data.astype(np.float32)
    
    # Vectorized voxel counting
    non_zero_voxels = np.count_nonzero(visualization_data > 0.01)
    print(f"Non-zero voxels: {non_zero_voxels} ({100*non_zero_voxels/visualization_data.size:.2f}% of volume)")

    try:
        # Convert to PyVista grid
        grid = pv.wrap(visualization_data)

        # Create plotter (off-screen for headless rendering)
        try:
            plotter = pv.Plotter(off_screen=True, window_size=VIZ_WINDOW_SIZE)
        except (SystemExit, KeyboardInterrupt):
            raise
        except Exception as plotter_error:
            error_msg = str(plotter_error).lower()
            print(f"⚠ Warning: Failed to create PyVista plotter: {plotter_error}")
            
            if "x server" in error_msg or "display" in error_msg or "bad x server connection" in error_msg:
                print("\n" + "="*60)
                print("OFFSCREEN RENDERING ERROR:")
                print("="*60)
                print("This script requires offscreen rendering for compute nodes.")
                print("Two common fixes:")
                print("  1) Launch the script under a virtual framebuffer:")
                print("       xvfb-run -s '-screen 0 1920x1080x24' python 3dslicer_v05_optimized_island_connection.py ...")
                print("  2) Install vtk-osmesa to enable pure offscreen rendering:")
                print("     conda install -c conda-forge vtk-osmesa")
                print("  3) Use --no-visualizations to skip all visualizations")
                print("="*60)
            
            print("  Continuing without visualization...")
            return

        # Use volume rendering (notebook style)
        print("  Using volume rendering with linear opacity (notebook style)")
        if "skeleton" in stage.lower():
            cmap = "hot"
        elif "dilated" in stage.lower():
            cmap = "viridis"
        else:
            cmap = "magma"
        
        plotter.add_volume(grid, opacity="linear", cmap=cmap)

        plotter.background_color = "black"
        plotter.show_axes()

        # Generate rotating video
        output_path = viz_dir / f"{stage}_stage_visualization.mp4"
        print(f"Saving {stage} stage visualization to: {output_path}")

        try:
            print(f"  Generating {VIZ_N_FRAMES} frame rotating video (framerate={VIZ_FRAMERATE})...")
            plotter.open_movie(str(output_path), framerate=VIZ_FRAMERATE)
            n_frames = VIZ_N_FRAMES
            progress_interval = max(1, n_frames // 4)
            
            # Precompute all camera angles vectorized (optimized)
            azimuth_angles = 180 + np.arange(n_frames, dtype=np.float32) * 360.0 / n_frames
            
            # Pre-set constant camera properties once
            plotter.camera_position = "yz"
            plotter.camera.elevation = 30

            # Optimized frame generation loop
            for i in range(n_frames):
                plotter.camera.azimuth = azimuth_angles[i]
                plotter.render()
                plotter.write_frame()
                
                # Progress reporting (only when needed)
                if (i + 1) % progress_interval == 0 or i == n_frames - 1:
                    print(f"    Progress: {i + 1}/{n_frames} frames ({100*(i+1)/n_frames:.1f}%)")

            plotter.close()
            
            # Verify file was created
            if output_path.exists() and output_path.stat().st_size > 0:
                file_size = output_path.stat().st_size / (1024 * 1024)
                print(f"✓ {stage.title()} stage visualization saved: {output_path} ({file_size:.2f} MB)")
            else:
                print(f"⚠ Warning: MP4 file created but appears empty or missing")
        except Exception as movie_error:
            print(f"Error during movie generation: {movie_error}")
            import traceback
            traceback.print_exc()
            try:
                plotter.close()
            except:
                pass
            raise

    except Exception as e:
        print(f"Error creating {stage} visualization: {e}")
        import traceback
        traceback.print_exc()


def _visualize_centerlines(centerlines: vtkPolyData, viz_dir: Path) -> None:
    """Visualize extracted centerlines as 3D rotating MP4."""
    if not PYVISTA_AVAILABLE:
        print("Skipping centerlines visualization (PyVista not available)")
        return

    if centerlines.GetNumberOfPoints() == 0:
        print("⚠️  No centerlines detected - cannot create visualization")
        print("   This usually means VMTK network extraction failed or returned empty results.")
        print("   Possible causes:")
        print("     - Preprocessed data is too sparse or disconnected")
        print("     - Surface generation failed (empty surface)")
        print("     - VMTK network extraction could not find valid centerlines")
        return

    print("Creating centerlines 3D visualization...")
    print(f"Centerlines points: {centerlines.GetNumberOfPoints()}")
    print(f"Centerlines lines: {centerlines.GetNumberOfCells()}")

    try:
        # Create plotter (off-screen for headless rendering)
        plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])

        # Wrap VTK PolyData in PyVista and convert lines to tubes for better 3D visibility
        centerlines_pv = pv.wrap(centerlines)
        centerlines_tubes = centerlines_pv.tube(radius=0.3, n_sides=8)
        
        # Add centerlines to the plotter as tubes
        plotter.add_mesh(centerlines_tubes, color="cyan")

        # Generate rotating video
        output_path = viz_dir / "final_centerlines_3d.mp4"
        print(f"Saving centerlines visualization to: {output_path}")

        try:
            print(f"  Generating {VIZ_N_FRAMES} frame rotating video (framerate={VIZ_FRAMERATE})...")
            plotter.open_movie(str(output_path), framerate=VIZ_FRAMERATE)
            n_frames = VIZ_N_FRAMES
            progress_interval = max(1, n_frames // 4)
            
            # Precompute all camera angles vectorized (optimized)
            azimuth_angles = 180 + np.arange(n_frames, dtype=np.float32) * 360.0 / n_frames
            
            # Pre-set constant camera properties once
            plotter.camera_position = "yz"
            plotter.camera.elevation = 30

            # Optimized frame generation loop
            for i in range(n_frames):
                plotter.camera.azimuth = azimuth_angles[i]
                plotter.render()
                plotter.write_frame()
                
                # Progress reporting (only when needed)
                if (i + 1) % progress_interval == 0 or i == n_frames - 1:
                    print(f"    Progress: {i + 1}/{n_frames} frames ({100*(i+1)/n_frames:.1f}%)")

            plotter.close()
            
            # Verify file was created
            if output_path.exists() and output_path.stat().st_size > 0:
                file_size = output_path.stat().st_size / (1024 * 1024)
                print(f"✓ Centerlines visualization saved: {output_path} ({file_size:.2f} MB)")
            else:
                print(f"⚠ Warning: MP4 file created but appears empty or missing")
        except Exception as movie_error:
            print(f"Error during movie generation: {movie_error}")
            import traceback
            traceback.print_exc()
            try:
                plotter.close()
            except:
                pass
            raise

    except Exception as e:
        print(f"Error creating centerlines visualization: {e}")


def _analyze_input_data(img: vtkImageData) -> dict:
    """Analyze input data to determine the best centerline extraction method."""
    scalars = img.GetPointData().GetScalars()
    if not scalars:
        return {"method": "unknown", "sparsity": 1.0, "connectivity": "unknown"}

    data = vtknp.vtk_to_numpy(scalars)
    binary_data = (data > THRESHOLD_DEFAULT).astype(np.uint8)

    # Calculate sparsity
    non_zero_count = np.count_nonzero(binary_data)
    total_voxels = binary_data.size
    sparsity = 1.0 - (non_zero_count / total_voxels)

    # Analyze connectivity
    labeled_data = label(binary_data)
    num_components = len(np.unique(labeled_data)) - 1  # Exclude background

    # Determine method based on characteristics
    if sparsity > SPARSITY_THRESHOLD_HIGH:  # Very sparse (likely vessels)
        method = "vessel_preprocessing"
    elif sparsity < SPARSITY_THRESHOLD_LOW:  # Dense (likely solid regions)
        method = "skeletonization"
    elif num_components > COMPONENTS_THRESHOLD:  # Many small components
        method = "vessel_preprocessing"
    else:  # Few large components
        method = "skeletonization"

    return {
        "method": method,
        "sparsity": sparsity,
        "connectivity": f"{num_components} components",
        "non_zero_count": non_zero_count,
        "total_voxels": total_voxels,
    }


def _preprocess_vessel_data(img: vtkImageData) -> vtkImageData:
    """Preprocess sparse vessel data to improve VMTK network extraction."""
    scalars = img.GetPointData().GetScalars()
    data = vtknp.vtk_to_numpy(scalars)

    # Reshape to 3D
    dims = img.GetDimensions()
    binary_data = (
        (data > THRESHOLD_DEFAULT).astype(np.uint8).reshape((dims[2], dims[1], dims[0]))
    )

    # Apply morphological operations to connect nearby vessel segments
    # Dilation to thicken vessels
    dilated = sk_binary_dilation(binary_data, footprint=np.ones((3, 3, 3)))
    
    # Erosion to restore original thickness
    processed = binary_erosion(dilated, footprint=np.ones((2, 2, 2)))

    # Convert back to VTK
    return _numpy_to_vtk_image(processed, img.GetSpacing())


def _extract_skeleton_centerlines(
    img: vtkImageData, 
    use_island_connection: bool = False,
    max_connection_distance_mm: float = DEFAULT_MAX_CONNECTION_DISTANCE_MM,
    viz_dir: Path | None = None
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
        (data > 0).astype(np.uint8).reshape((dims[2], dims[1], dims[0]))
    )
    
    print(f"  Binary data for skeletonization: {np.count_nonzero(binary_data)} voxels")

    # Extract 3D skeleton
    skeleton = skeletonize(binary_data)
    skeleton_points = np.count_nonzero(skeleton)
    print(f"  Skeleton points: {skeleton_points}")
    
    # Visualize skeleton before connection (if requested)
    if viz_dir and use_island_connection:
        try:
            _visualize_intermediate_stage(
                skeleton.astype(np.float32), viz_dir, "skeleton_before_connection",
                "Skeleton before island connection"
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
                    skeleton.astype(np.float32), viz_dir, "skeleton_after_connection",
                    "Skeleton after island connection"
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
    min_branch_length: int = 10
) -> tuple[list[dict], list[dict]]:
    """Convert 3D skeleton to graph structure (nodes and links) - Python version of Skel2Graph3D.
    
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
        if num_neighbors > 2:
            junction_voxels.append(i)
        elif num_neighbors == 1:
            endpoint_voxels.append(i)
        elif num_neighbors == 2:
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
        
        def dfs_junction(start_idx: int, component: list[int]):
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
            nodes.append({
                'idx': component,
                'comx': comx,
                'comy': comy,
                'comz': comz,
                'ep': 0,  # Not an endpoint
                'links': [],
                'conn': []
            })
    else:
        junction_components = []
    
    # Create nodes from endpoints (each endpoint is its own node)
    num_junction_nodes = len(nodes)
    for i, point_idx in enumerate(endpoint_voxels):
        node_idx = num_junction_nodes + i
        node_idx_map[point_idx] = node_idx
        nodes.append({
            'idx': [point_idx],
            'comx': x_coords[point_idx],
            'comy': y_coords[point_idx],
            'comz': z_coords[point_idx],
            'ep': 1,  # Is an endpoint
            'links': [],
            'conn': []
        })
    
    # Create mapping for canal voxels (points with 2 neighbors)
    canal_to_neighbors = {}
    for point_idx in canal_voxels:
        neighbors = graph[point_idx]
        canal_to_neighbors[point_idx] = neighbors
    
    # Follow links from each node
    links = []
    link_idx = 0
    processed_links = set()  # Track processed links to avoid duplicates
    
    def follow_link(start_node_idx: int, start_point_idx: int, direction_point_idx: int) -> tuple[list[int], int, bool]:
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
                is_endpoint = nodes[end_node_idx]['ep'] == 1
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
        node_voxels = node['idx']
        
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
                            'n1': node_idx,
                            'n2': end_node_idx,
                            'point': [start_voxel, neighbor]
                        }
                        links.append(link)
                        node['links'].append(link_idx)
                        node['conn'].append(end_node_idx)
                        nodes[end_node_idx]['links'].append(link_idx)
                        nodes[end_node_idx]['conn'].append(node_idx)
                        link_idx += 1
            elif neighbor in canal_to_neighbors:
                # Follow link through canal voxels
                path, end_node_idx, is_endpoint = follow_link(node_idx, start_voxel, neighbor)
                
                if end_node_idx is not None and end_node_idx != node_idx and len(path) >= min_branch_length:
                    # Avoid duplicate links
                    link_key = tuple(sorted([node_idx, end_node_idx]))
                    if link_key not in processed_links:
                        processed_links.add(link_key)
                        link = {
                            'n1': node_idx,
                            'n2': end_node_idx,
                            'point': path
                        }
                        links.append(link)
                        node['links'].append(link_idx)
                        node['conn'].append(end_node_idx)
                        nodes[end_node_idx]['links'].append(link_idx)
                        nodes[end_node_idx]['conn'].append(node_idx)
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
    print(f"  Skeleton point ranges: z=[{z_coords.min()}, {z_coords.max()}], y=[{y_coords.min()}, {y_coords.max()}], x=[{x_coords.min()}, {x_coords.max()}]")
    print(f"  VTK spacing (x, y, z): {spacing}")
    
    # Create mapping from (z, y, x) to point index
    n_points = len(z_coords)
    coord_to_idx = {}
    for i in range(n_points):
        coord_to_idx[(z_coords[i], y_coords[i], x_coords[i])] = i

    # Convert to world coordinates (x, y, z) for VTK
    # CRITICAL: PyVista interprets numpy (z, y, x) arrays as world (x, y, z) where:
    #   numpy[0] (z dimension) → world x coordinate
    #   numpy[1] (y dimension) → world y coordinate  
    #   numpy[2] (x dimension) → world z coordinate
    points_world = np.column_stack([
        z_coords * spacing[0],  # numpy z → world x (PyVista convention)
        y_coords * spacing[1],  # numpy y → world y
        x_coords * spacing[2]   # numpy x → world z (PyVista convention)
    ])
    
    print(f"  World coordinate ranges (PyVista-aligned): x=[{points_world[:, 0].min():.2f}, {points_world[:, 0].max():.2f}], y=[{points_world[:, 1].min():.2f}, {points_world[:, 1].max():.2f}], z=[{points_world[:, 2].min():.2f}, {points_world[:, 2].max():.2f}]")

    # Create VTK points
    vtk_points = vtk.vtkPoints()
    for point in points_world:
        vtk_points.InsertNextPoint(point)

    # Build graph: use 26-connectivity (like Matlab Skel2Graph3D)
    offsets = [
        (dz, dy, dx) for dz in [-1, 0, 1]
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
            
            if (nz < 0 or nz >= skeleton_array.shape[0] or
                ny < 0 or ny >= skeleton_array.shape[1] or
                nx < 0 or nx >= skeleton_array.shape[2]):
                continue
            
            if skeleton_array[nz, ny, nx] > 0:
                neighbor_idx = coord_to_idx.get((nz, ny, nx))
                if neighbor_idx is not None:
                    graph[i].append(neighbor_idx)
    
    # Convert skeleton to graph structure (like Skel2Graph3D)
    print("  Converting skeleton to graph structure (Skel2Graph3D style)...")
    min_branch_length = 5  # Minimum branch length in voxels
    nodes, links = _skel2graph3d(
        skeleton_array, z_coords, y_coords, x_coords,
        coord_to_idx, graph, min_branch_length
    )
    
    print(f"  Graph structure: {len(nodes)} nodes, {len(links)} links")
    
    # Extract centerlines from links (each link.point is a centerline segment)
    lines = vtk.vtkCellArray()
    for link in links:
        point_indices = link['point']
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
    num_endpoints = sum(1 for node in nodes if node['ep'] == 1)
    print(f"  Created {num_lines} centerline segments from {num_nodes} nodes ({num_endpoints} endpoints)")
    
    return polydata


def _extract_vmtk_centerlines(
    img: vtkImageData, target_points: int = 15000
) -> vtkPolyData:
    """Extract centerlines using VMTK network extraction."""
    print("Step 1: Generating surface using marching cubes...")
    
    # Check input data first
    scalars = img.GetPointData().GetScalars()
    if scalars:
        data = vtknp.vtk_to_numpy(scalars)
        non_zero = np.count_nonzero(data > 0.5)
        total = data.size
        print(f"  Input data: {non_zero}/{total} voxels above threshold ({100*non_zero/total:.2f}%)")
        if non_zero == 0:
            print("  ⚠️  ERROR: No voxels above threshold - cannot generate surface")
            return vtkPolyData()
    
    # Generate surface using marching cubes
    surface = _mask_to_surface(img, 0.5)
    print(f"  Surface generated: {surface.GetNumberOfPoints()} points, {surface.GetNumberOfCells()} cells")

    if surface.GetNumberOfPoints() == 0:
        print("  ⚠️  ERROR: Empty surface - marching cubes failed")
        print("     Possible causes:")
        print("       - Input data is too sparse or has no connected components")
        print("       - Threshold (0.5) is too high for the data")
        return vtkPolyData()

    print(f"Step 2: Decimating surface to ~{target_points} points...")
    # Decimate surface
    decimated = _decimate_to_points(surface, target_points)
    print(f"  Decimated surface: {decimated.GetNumberOfPoints()} points, {decimated.GetNumberOfCells()} cells")

    if decimated.GetNumberOfPoints() == 0:
        print("  ⚠️  ERROR: Decimation resulted in empty surface")
        return vtkPolyData()

    # VMTK network extraction
    print("Step 3: Running VMTK network extraction (this may take several minutes)...")
    try:
        net = vmtkscripts.vmtkNetworkExtraction()
        net.Surface = decimated
        print("  Executing VMTK network extraction...")
        net.Execute()
        network = net.Network
        print(f"  Network extracted: {network.GetNumberOfPoints()} points, {network.GetNumberOfCells()} cells")

        if network.GetNumberOfPoints() == 0:
            print("  ⚠️  ERROR: VMTK network extraction returned empty network")
            print("     Possible causes:")
            print("       - Surface topology is not suitable for network extraction")
            print("       - Surface has no valid network structure (too fragmented)")
            print("       - VMTK parameters may need adjustment")
            return vtkPolyData()

        print("Step 4: Resampling centerlines...")
        # Resample centerlines
        res = vmtkscripts.vmtkCenterlineResampling()
        res.Centerlines = network
        res.Length = 0.6  # 0.6mm spacing
        res.Execute()
        final_points = res.Centerlines.GetNumberOfPoints()
        print(f"  Resampled centerlines: {final_points} points")

        if final_points == 0:
            print("  ⚠️  WARNING: Resampling resulted in 0 points")
            return vtkPolyData()

        return res.Centerlines

    except Exception as e:
        print(f"  ⚠️  ERROR: VMTK network extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return vtkPolyData()


def _mask_to_surface(img: vtkImageData, level: float) -> vtkPolyData:
    """Marching cubes surface generation from binary mask."""
    mc = vtkFlyingEdges3D()
    mc.SetInputData(img)
    mc.SetValue(0, level)
    mc.Update()

    tri = vtkTriangleFilter()
    tri.SetInputConnection(mc.GetOutputPort())
    tri.Update()

    clean = vtkCleanPolyData()
    clean.SetInputConnection(tri.GetOutputPort())
    clean.Update()

    conn = vtkPolyDataConnectivityFilter()
    conn.SetInputConnection(clean.GetOutputPort())
    conn.SetExtractionModeToLargestRegion()
    conn.Update()

    return conn.GetOutput()


def _decimate_to_points(surf: vtkPolyData, target_points: int) -> vtkPolyData:
    """Decimate surface to target number of points."""
    n0 = max(1, surf.GetNumberOfPoints())
    if target_points <= 0 or target_points >= n0:
        return surf

    target_reduction = float(max(0.0, min(0.99, 1.0 - (target_points / float(n0)))))

    dec = vtkDecimatePro()
    dec.SetInputData(surf)
    dec.SetTargetReduction(target_reduction)
    dec.PreserveTopologyOn()
    dec.BoundaryVertexDeletionOn()
    dec.Update()

    clean = vtkCleanPolyData()
    clean.SetInputConnection(dec.GetOutputPort())
    clean.Update()
    return clean.GetOutput()


def _numpy_to_vtk_image(
    vol_zyx: np.ndarray, spacing: tuple[float, float, float]
) -> vtkImageData:
    """Convert numpy array to VTK ImageData."""
    nz, ny, nx = map(int, vol_zyx.shape)
    img = vtkImageData()
    img.SetDimensions(nx, ny, nz)
    img.SetSpacing(spacing[2], spacing[1], spacing[0])  # VTK uses x,y,z order
    img.SetOrigin(0.0, 0.0, 0.0)
    scalars = vtknp.numpy_to_vtk(
        vol_zyx.ravel(order="F"),
        deep=True,
        array_type=vtkTypeFloat32Array().GetDataType(),
    )
    scalars.SetName("Scalars")
    img.GetPointData().SetScalars(scalars)
    return img


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
    force_method: str = None,  # "vmtk", "skeleton", or None for auto
    target_points: int = 15000,
    curve_sampling_mm: float = 0.6,
    enable_visualizations: bool = True,
    max_connection_distance_mm: float = DEFAULT_MAX_CONNECTION_DISTANCE_MM,
    use_island_connection: bool = True,
    extract_label: int = None,  # Specific label to extract from multi-label files
) -> None:
    """Extract centerlines using the most appropriate method for the input data.
    
    Version 05 adds optimized island connection preprocessing with major performance improvements:
    - Batch connection strategy (10-100x faster for many components)
    - Vectorized endpoint detection
    - Global KD-tree per iteration
    - Union-Find for component tracking
    
    All visualizations are saved as MP4 files (rotating 3D videos).
    
    IMPORTANT: This script is configured for headless/offscreen rendering.
    Safe for SLURM compute nodes - no windows will pop up. All visualizations
    are written directly to MP4 files without opening any display windows.

    Args:
        input_segmentation_path: Path to input mask (.nii.gz/.nrrd/.npy)
        output_centerline_path: Path to output centerlines (.vtp/.vtk)
        force_method: Force specific method ("vmtk" or "skeleton")
        target_points: Target points for surface decimation
        curve_sampling_mm: Resampling step for VMTK centerlines
        enable_visualizations: Whether to create intermediate visualizations
        max_connection_distance_mm: Maximum distance to connect islands (mm)
        use_island_connection: Whether to use optimized island connection preprocessing
        extract_label: Specific label to extract from multi-label NRRD files
    """
    ipath = Path(input_segmentation_path)
    opath = Path(output_centerline_path)

    # Create visualization directory (if visualizations are enabled)
    viz_dir = None
    if enable_visualizations:
        viz_dir = _create_visualization_dir(opath)

    # Read input image
    # Store original probability data for visualization (before binarization)
    original_probability_data = None
    
    if ipath.suffix.lower() == ".npy":
        # Handle .npy files (NumPy arrays)
        # Use EXACTLY the same loading logic as visualize_vessels_notebook_style.py
        data = np.load(str(ipath))
        
        print(f"Input shape: {data.shape}")
        print(f"Data range: [{np.min(data):.4f}, {np.max(data):.4f}]")
        
        # If 4D (channels, z, y, x), extract the vessel channel (same as notebook)
        if data.ndim == 4:
            channel = 1  # Channel 1 = blood vessels (same as notebook default)
            if channel >= data.shape[0]:
                raise ValueError(f"Channel {channel} not available. Input has {data.shape[0]} channels.")
            print(f"Extracting channel {channel} (blood vessels)")
            data = data[channel].astype(float)  # Convert to float (same as notebook)
            print(f"Vessel data shape: {data.shape}")
        elif data.ndim == 3:
            # Already 3D, use as-is (same as notebook)
            data = data.astype(float)
            print("Input is already 3D, using directly")
        else:
            raise ValueError(f"Expected 3D or 4D array, got shape {data.shape}")
        
        # Convert to float32 for processing (float16 can cause issues)
        if data.dtype == np.float16:
            data = data.astype(np.float32)
        
        # Check if this looks like probability data
        data_min, data_max = np.min(data), np.max(data)
        is_probability = (data_max <= 1.0 and data_min >= 0.0)
        
        if is_probability:
            print(f"Detected probability data (range: [{data_min:.4f}, {data_max:.4f}])")
            print(f"Non-zero voxels: {np.count_nonzero(data)} ({100*np.count_nonzero(data)/data.size:.2f}% of volume)")
            
            # Preserve original probability data for visualization (notebook style)
            # This is exactly what visualize_vessels_notebook_style.py uses
            original_probability_data = data.copy()
            
            # For centerline extraction: use a threshold that captures visible vessels
            # Volume rendering shows all non-zero values, but most are tiny (noise)
            # Use a threshold that captures the meaningful vessel structures
            # Threshold of 0.01 captures ~53k voxels (vs 36M with > 0)
            centerline_threshold = 0.01
            voxels_above_threshold = np.count_nonzero(data > centerline_threshold)
            print(f"  Binarizing for centerline extraction (threshold={centerline_threshold}, matching visible vessels)")
            print(f"  This captures {voxels_above_threshold} voxels ({voxels_above_threshold/data.size*100:.2f}% of volume)")
            print(f"  (Using > 0 would give {np.count_nonzero(data > 0)} voxels, which is too slow)")
            data = (data > centerline_threshold).astype(np.float32)
            print(f"  Binary voxels: {np.count_nonzero(data)}")
        else:
            # Not probability data, treat as binary
            print(f"Converting to binary (threshold={THRESHOLD_DEFAULT})")
            original_probability_data = None
            data = (data > THRESHOLD_DEFAULT).astype(np.float32)
        
        # Default spacing for .npy files (can be overridden)
        spacing_zyx = (1.0, 1.0, 1.0)
        img = _numpy_to_vtk_image(data, spacing_zyx)
        
    elif ipath.suffix.lower() in {".nii", ".gz"} and ipath.name.endswith(".nii.gz"):
        r = vtkNIFTIImageReader()
        r.SetFileName(str(ipath))
        r.Update()
        img = r.GetOutput()
    elif ipath.suffix.lower() in {".nii"}:
        r = vtkNIFTIImageReader()
        r.SetFileName(str(ipath))
        r.Update()
        img = r.GetOutput()
    elif ipath.suffix.lower() == ".nrrd":
        import nrrd

        data, header = nrrd.read(str(ipath))
        if data.ndim != DIMENSIONS_3D:
            raise ValueError(f"expected 3D nrrd, got shape {tuple(data.shape)}")

        # Check if this is a multi-label segmentation
        unique_values = np.unique(data)
        if len(unique_values) > 2:
            print(f"Detected multi-label NRRD with values: {unique_values}")
            
            # If user specified a label, use it
            if extract_label is not None:
                if extract_label not in unique_values:
                    raise ValueError(f"Label {extract_label} not found in NRRD. Available labels: {unique_values}")
                target_label = extract_label
                # Try to get label name from header
                label_name = f"Label_{target_label}"
                for key in header.keys():
                    if f'Segment{target_label}_Name' in key:
                        label_name = header[key]
                        break
                print(f"Extracting specified label {target_label} ({label_name})...")
            else:
                # Auto-detect: Try to find vessel label from header
                print("Checking header for vessel label...")
                vessel_label = None
                for key in header.keys():
                    if 'Segment' in key and 'Name' in key:
                        if 'vessel' in header[key].lower() or 'vessel' in key.lower():
                            # Extract segment number
                            seg_num = key.split('Segment')[1].split('_')[0]
                            vessel_label = int(seg_num)
                            print(f"Found vessel segment: {header[key]} (label {vessel_label})")
                            break
                
                if vessel_label is None:
                    # Default: use label 2 if it exists (common for vessel segmentations)
                    if 2 in unique_values:
                        vessel_label = 2
                        print(f"Using label 2 as vessels (default)")
                    else:
                        # Use highest non-zero label
                        vessel_label = int(unique_values[unique_values > 0].max())
                        print(f"Using highest non-zero label {vessel_label} as vessels")
                target_label = vessel_label
            
            # Extract only the specified label
            data = (data == target_label).astype(np.float32)
            print(f"Extracted label {target_label}: {np.count_nonzero(data)} voxels ({100*np.count_nonzero(data)/data.size:.2f}% of volume)")
        else:
            # Binary mask - convert to float
            data = data.astype(np.float32)

        vol_zyx = (
            rearrange(data, "x y z -> z y x")
            if header.get("dimension") == DIMENSIONS_3D
            else data
        )
        sdirs = header.get("space directions")
        if sdirs is None:
            raise ValueError("missing 'space directions' in nrrd header")
        spacing_xyz = tuple(float(np.linalg.norm(np.asarray(v))) for v in sdirs)
        spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
        img = _numpy_to_vtk_image(vol_zyx, spacing_zyx)
    else:
        raise ValueError(f"unsupported input: {ipath}")

    # 0. Create 3D vessel visualization first (before any processing)
    if enable_visualizations and viz_dir:
        try:
            print("Creating 3D vessel visualization...")
            print("  Note: On compute nodes, visualizations may fail due to rendering limitations.")
            print("  The script will continue with centerline extraction even if visualization fails.")
            
            # Use original probability data if available (for notebook-style visualization)
            # Directly use numpy array like visualize_vessels_notebook_style.py does
            if original_probability_data is not None:
                _visualize_3d_vessels_from_numpy(original_probability_data, viz_dir)
            else:
                # Extract from VTK ImageData
                _visualize_3d_vessels(img, viz_dir)
        except (SystemExit, KeyboardInterrupt):
            raise
        except Exception as e:
            print(f"⚠ Warning: Initial visualization failed (continuing with centerline extraction): {e}")

    # Analyze input data
    analysis = _analyze_input_data(img)
    print(f"Input analysis: {analysis}")

    # Choose method
    if force_method:
        # Map command-line argument names to internal method names
        if force_method == "skeleton":
            method = "skeletonization"
        elif force_method == "vmtk":
            method = "vessel_preprocessing"
        else:
            method = force_method
        print(f"Using forced method: {method}")
    else:
        method = analysis["method"]
        print(f"Auto-selected method: {method}")

    # Extract centerlines
    if method == "vessel_preprocessing":
        print("Using vessel preprocessing + VMTK...")
        
        if use_island_connection:
            print("  Using optimized island connection preprocessing (v05 feature)...")
            # Pass viz_dir to get step-by-step visualizations
            preprocessed_img = _preprocess_vessel_data_with_island_connection(
                img, max_connection_distance_mm, viz_dir=viz_dir if enable_visualizations else None
            )
        else:
            print("  Using standard preprocessing (v03 method)...")
            preprocessed_img = _preprocess_vessel_data(img)
            
            # Visualize standard preprocessing result
            if enable_visualizations and viz_dir:
                try:
                    print("Creating preprocessing visualization...")
                    scalars = preprocessed_img.GetPointData().GetScalars()
                    if scalars:
                        data = vtknp.vtk_to_numpy(scalars)
                        dims = preprocessed_img.GetDimensions()
                        if len(data.shape) == 1:
                            data_3d = data.reshape((dims[2], dims[1], dims[0]))
                        else:
                            data_3d = data
                        _visualize_intermediate_stage(
                            data_3d, viz_dir, "preprocessed", "Vessel preprocessing (standard method)"
                        )
                except (SystemExit, KeyboardInterrupt):
                    raise
                except Exception as e:
                    print(f"⚠ Warning: Preprocessing visualization failed (continuing with centerline extraction): {e}")

        centerlines = _extract_vmtk_centerlines(preprocessed_img, target_points)
    elif method == "skeletonization":
        print("Using skeletonization...")

        # 2. Visualize skeletonization process (optional, may fail on compute nodes)
        if enable_visualizations and viz_dir:
            try:
                print("Creating skeletonization visualization...")
                scalars = img.GetPointData().GetScalars()
                if scalars:
                    data = vtknp.vtk_to_numpy(scalars)
                    dims = img.GetDimensions()
                    if len(data.shape) == 1:
                        data_3d = data.reshape((dims[2], dims[1], dims[0]))
                    else:
                        data_3d = data
                    _visualize_intermediate_stage(
                        data_3d, viz_dir, "skeletonization", "3D skeletonization method"
                    )
            except (SystemExit, KeyboardInterrupt):
                raise
            except Exception as e:
                print(f"⚠ Warning: Skeletonization visualization failed (continuing with centerline extraction): {e}")

        centerlines = _extract_skeleton_centerlines(
            img, 
            use_island_connection=use_island_connection,
            max_connection_distance_mm=max_connection_distance_mm,
            viz_dir=viz_dir if enable_visualizations else None
        )
    else:
        print(f"Unknown method: {method}, falling back to skeletonization...")
        centerlines = _extract_skeleton_centerlines(
            img, 
            use_island_connection=use_island_connection,
            max_connection_distance_mm=max_connection_distance_mm,
            viz_dir=viz_dir if enable_visualizations else None
        )

    # 3. Visualize final centerlines (optional, may fail on compute nodes)
    if enable_visualizations and viz_dir:
        try:
            print("Creating final centerlines visualization...")
            _visualize_centerlines(centerlines, viz_dir)
        except (SystemExit, KeyboardInterrupt):
            raise
        except Exception as e:
            print(f"⚠ Warning: Centerlines visualization failed (centerline extraction completed successfully): {e}")

    # Write output (this should always happen, regardless of visualization success)
    _write_polydata(centerlines, opath)

    # Report results
    num_points = centerlines.GetNumberOfPoints()
    if num_points > 0:
        print(
            f"✅ Successfully extracted {centerlines.GetNumberOfPoints()} centerline points"
        )
        if enable_visualizations and viz_dir:
            print(f"\n📊 Step-by-step MP4 visualizations saved to: {viz_dir}")
            print("\n" + "="*70)
            print("MP4 PROGRESS CHECKER")
            print("="*70)
            
            # List all MP4 files that should have been created
            mp4_files = []
            
            # Input visualization
            mp4_files.append(("1. Input Data", "vessels_3d.mp4"))
            
            # Island connection steps (if used)
            if method == "vessel_preprocessing" and use_island_connection:
                mp4_files.append(("2. Skeleton (Before Connection)", "skeleton_before_connection_stage_visualization.mp4"))
                mp4_files.append(("3. Skeleton (After Connection)", "skeleton_after_connection_stage_visualization.mp4"))
                mp4_files.append(("4. Dilated Skeleton", "skeleton_dilated_stage_visualization.mp4"))
            elif method == "vessel_preprocessing":
                mp4_files.append(("2. Preprocessed Data", "preprocessed_stage_visualization.mp4"))
            elif method == "skeletonization":
                mp4_files.append(("2. Skeletonization", "skeletonization_stage_visualization.mp4"))
            
            # Final centerlines
            mp4_files.append((f"{len(mp4_files) + 1}. Final Centerlines", "final_centerlines_3d.mp4"))
            
            # Check progress
            progress = _check_mp4_progress(viz_dir, mp4_files)
            
            # Print detailed status
            print(f"\nProgress: {progress['created']}/{progress['total']} MP4 files created")
            print("\nDetailed Status:")
            for step_name, filename in mp4_files:
                file_info = progress["files"][step_name]
                status = file_info["status"]
                size = file_info["size_mb"]
                if "✓" in status:
                    print(f"  {status} {step_name}")
                    print(f"      File: {filename}")
                    print(f"      Size: {size:.2f} MB")
                elif "⚠" in status:
                    print(f"  {status} {step_name}")
                    print(f"      File: {filename}")
                    print(f"      Issue: {status.replace('⚠ ', '')}")
                else:
                    print(f"  {status} {step_name}")
                    print(f"      File: {filename}")
                    print(f"      Expected: {viz_dir / filename}")
            
            if progress["missing"]:
                print(f"\n⚠ Warning: {len(progress['missing'])} MP4 file(s) missing or empty:")
                for step in progress["missing"]:
                    print(f"    - {step}")
            else:
                print(f"\n✅ All {progress['total']} MP4 files successfully created!")
            
            print(f"\nAll MP4 files are rotating 3D visualizations ({VIZ_N_FRAMES} frames, {VIZ_FRAMERATE} fps)")
            print("="*70)
    else:
        print("\n" + "="*70)
        print("⚠️  CENTERLINE EXTRACTION FAILED")
        print("="*70)
        print(f"No centerlines extracted ({num_points} points)")
        print("\nPossible causes:")
        print("  1. Input data is too sparse or disconnected")
        print("  2. Surface generation failed (marching cubes returned empty surface)")
        print("  3. VMTK network extraction could not find valid centerlines")
        print("  4. Island connection preprocessing may have created invalid geometry")
        print("\nTroubleshooting:")
        print("  - Check the intermediate visualizations to see if preprocessing worked")
        print("  - Try reducing --max-connection-distance-mm")
        print("  - Try using --no-island-connection to use standard preprocessing")
        print("  - Try using --method skeleton instead of VMTK")
        print("="*70)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Adaptive centerline extraction with optimized island connection (v05)"
    )
    p.add_argument("input", help="input .nii.gz/.nrrd/.npy")
    p.add_argument("output", help="output .vtp or .vtk")
    p.add_argument(
        "--method",
        choices=["vmtk", "skeleton"],
        help="Force specific method (auto-detect if not specified)",
    )
    p.add_argument(
        "--target-points",
        type=int,
        default=15000,
        help="target vertex count for VMTK",
    )
    p.add_argument(
        "--curve-sampling-mm",
        type=float,
        default=0.6,
        help="resampling step for VMTK centerlines (mm)",
    )
    p.add_argument(
        "--max-connection-distance-mm",
        type=float,
        default=DEFAULT_MAX_CONNECTION_DISTANCE_MM,
        help=f"maximum distance to connect islands (mm, default: {DEFAULT_MAX_CONNECTION_DISTANCE_MM})",
    )
    p.add_argument(
        "--no-island-connection",
        action="store_true",
        help="disable island connection preprocessing (use v03 method)",
    )
    p.add_argument(
        "--no-visualizations",
        action="store_true",
        help="disable visualization generation",
    )
    p.add_argument(
        "--extract-label",
        type=int,
        default=None,
        help="Extract specific label from multi-label NRRD file (e.g., --extract-label 1 for label 1)",
    )
    args = p.parse_args()

    extract_adaptive_centerlines(
        args.input,
        args.output,
        force_method=args.method,
        target_points=args.target_points,
        curve_sampling_mm=args.curve_sampling_mm,
        enable_visualizations=not args.no_visualizations,
        max_connection_distance_mm=args.max_connection_distance_mm,
        use_island_connection=not args.no_island_connection,
        extract_label=args.extract_label,
    )

