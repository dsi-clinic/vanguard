"""Convert VTK centerlines to JSON format matching sample_json.json structure.

Extracts vessel metrics: radius, length, tortuosity, volume, curvature, and bifurcations.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
import vtk
from vtkmodules.util import numpy_support as vtknp
from vtkmodules.vtkCommonDataModel import vtkPolyData
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader
from vtkmodules.vtkIOLegacy import vtkPolyDataReader


def _load_polydata(filepath: Path) -> vtkPolyData:
    """Load VTK PolyData from file."""
    suffix = filepath.suffix.lower()
    if suffix == ".vtp":
        reader = vtkXMLPolyDataReader()
    elif suffix == ".vtk":
        reader = vtkPolyDataReader()
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
    
    reader.SetFileName(str(filepath))
    reader.Update()
    return reader.GetOutput()


def _calculate_segment_length(points: np.ndarray) -> float:
    """Calculate path length of a segment."""
    if len(points) < 2:
        return 0.0
    diffs = np.diff(points, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    return float(np.sum(distances))


def _calculate_tortuosity(points: np.ndarray) -> float:
    """Calculate tortuosity (path length / euclidean distance)."""
    if len(points) < 2:
        return 0.0
    path_length = _calculate_segment_length(points)
    euclidean_dist = np.linalg.norm(points[-1] - points[0])
    if euclidean_dist < 1e-6:
        return 0.0
    return float((path_length / euclidean_dist) - 1.0)


def _calculate_curvature(points: np.ndarray) -> np.ndarray:
    """Calculate curvature at each point along the segment.
    
    Uses finite differences to compute curvature: |dT/ds| where T is tangent vector.
    """
    if len(points) < 3:
        return np.zeros(len(points))
    
    # Compute tangent vectors
    diffs = np.diff(points, axis=0)
    ds = np.linalg.norm(diffs, axis=1, keepdims=True)
    ds = np.maximum(ds, 1e-6)  # Avoid division by zero
    tangents = diffs / ds
    
    # Compute curvature as change in tangent direction
    dT = np.diff(tangents, axis=0)
    ds_mid = (ds[:-1] + ds[1:]) / 2.0
    ds_mid = np.maximum(ds_mid, 1e-6)
    curvature = np.linalg.norm(dT, axis=1) / ds_mid.flatten()
    
    # Pad to match number of points (curvature undefined at endpoints)
    curvature_full = np.zeros(len(points))
    curvature_full[1:-1] = curvature
    
    return curvature_full


def _estimate_radius_from_segmentation(
    centerline_points: np.ndarray,
    segmentation: np.ndarray,
    spacing: tuple[float, float, float]
) -> np.ndarray:
    """Estimate radius at each centerline point from segmentation.
    
    Uses distance transform from centerline to nearest boundary.
    """
    from scipy.ndimage import distance_transform_edt
    
    # Convert centerline points to voxel coordinates
    # Centerline is in world coords (x, y, z), segmentation is (z, y, x)
    spacing_zyx = (spacing[2], spacing[1], spacing[0])
    
    radii = []
    for point in centerline_points:
        # Convert world (x,y,z) to voxel (z,y,x)
        voxel_z = int(round(point[2] / spacing_zyx[0]))
        voxel_y = int(round(point[1] / spacing_zyx[1]))
        voxel_x = int(round(point[0] / spacing_zyx[2]))
        
        # Check bounds
        if (0 <= voxel_z < segmentation.shape[0] and
            0 <= voxel_y < segmentation.shape[1] and
            0 <= voxel_x < segmentation.shape[2]):
            
            # Extract local region around point
            z_min = max(0, voxel_z - 5)
            z_max = min(segmentation.shape[0], voxel_z + 6)
            y_min = max(0, voxel_y - 5)
            y_max = min(segmentation.shape[1], voxel_y + 6)
            x_min = max(0, voxel_x - 5)
            x_max = min(segmentation.shape[2], voxel_x + 6)
            
            local_region = segmentation[z_min:z_max, y_min:y_max, x_min:x_max]
            if np.any(local_region > 0):
                # Distance transform from center
                center_local = (voxel_z - z_min, voxel_y - y_min, voxel_x - x_min)
                dt = distance_transform_edt(local_region > 0)
                radius = dt[center_local] * min(spacing_zyx)
                radii.append(radius)
            else:
                radii.append(0.0)
        else:
            radii.append(0.0)
    
    return np.array(radii)


def _find_bifurcations(
    centerlines: vtkPolyData,
    point_tolerance: float = 2.0
) -> list[dict]:
    """Find bifurcation points where 3+ segments meet.
    
    Returns list of bifurcation dicts with midpoint, connected segments, etc.
    """
    points = vtknp.vtk_to_numpy(centerlines.GetPoints().GetData())
    n_points = len(points)
    
    # Build connectivity: which points are close to each other
    # Use KD-tree for efficient neighbor search
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    
    # Find points that are close to multiple line segments
    lines = centerlines.GetLines()
    lines.InitTraversal()
    
    # Map each point to segments that pass through it
    point_to_segments = {}
    segment_id = 0
    id_list = vtk.vtkIdList()
    
    while lines.GetNextCell(id_list):
        n_cell_points = id_list.GetNumberOfIds()
        if n_cell_points < 2:
            continue
        
        segment_points = []
        for i in range(n_cell_points):
            pt_id = id_list.GetId(i)
            segment_points.append(pt_id)
        
        # Add this segment to all points it contains
        for pt_id in segment_points:
            if pt_id not in point_to_segments:
                point_to_segments[pt_id] = []
            point_to_segments[pt_id].append(segment_id)
        
        segment_id += 1
    
    # Find bifurcations: points where 3+ segments meet or are very close
    bifurcations = []
    processed_points = set()
    
    for pt_id, segments in point_to_segments.items():
        if len(segments) >= 3 and pt_id not in processed_points:
            # Find nearby points that might be part of same bifurcation
            point_coord = points[pt_id]
            neighbors = tree.query_ball_point(point_coord, point_tolerance)
            
            # Collect all segments connected to this region
            all_segments = set()
            region_points = []
            for neighbor_id in neighbors:
                if neighbor_id in point_to_segments:
                    all_segments.update(point_to_segments[neighbor_id])
                    region_points.append(neighbor_id)
                    processed_points.add(neighbor_id)
            
            if len(all_segments) >= 3:
                # Calculate midpoint of bifurcation region
                region_coords = points[region_points]
                midpoint_idx = region_points[np.argmin(cdist([point_coord], region_coords)[0])]
                
                bifurcations.append({
                    'midpoint': int(midpoint_idx),
                    'dist_angle': point_tolerance,
                    'points_angle': region_points[:3],  # First 3 points for angle calculation
                    'segments': list(all_segments),
                    'region_points': region_points
                })
    
    return bifurcations


def _calculate_bifurcation_angles(
    bifurcation: dict,
    centerlines: vtkPolyData,
    points: np.ndarray
) -> dict[str, float]:
    """Calculate angles between segments at a bifurcation."""
    lines = centerlines.GetLines()
    lines.InitTraversal()
    
    # Get direction vectors for each segment near the bifurcation
    segment_vectors = []
    segment_ids = set(bifurcation['segments'][:3])  # Use first 3 segments
    id_list = vtk.vtkIdList()
    
    segment_idx = 0
    while lines.GetNextCell(id_list):
        if segment_idx in segment_ids:
            n_points = id_list.GetNumberOfIds()
            if n_points >= 2:
                # Get direction vector (from first to second point)
                pt1_id = id_list.GetId(0)
                pt2_id = id_list.GetId(min(1, n_points - 1))
                vec = points[pt2_id] - points[pt1_id]
                vec_norm = np.linalg.norm(vec)
                if vec_norm > 1e-6:
                    segment_vectors.append(vec / vec_norm)
                else:
                    segment_vectors.append(None)
        segment_idx += 1
        if len(segment_vectors) >= len(segment_ids):
            break
    
    # Calculate angles between pairs of vectors
    angles = {}
    if len(segment_vectors) >= 3:
        v1, v2, v3 = segment_vectors[0], segment_vectors[1], segment_vectors[2]
        if v1 is not None and v2 is not None:
            angles['seg1/seg2'] = float(np.arccos(np.clip(np.dot(v1, v2), -1, 1)) * 180 / np.pi)
        if v1 is not None and v3 is not None:
            angles['seg1/seg3'] = float(np.arccos(np.clip(np.dot(v1, v3), -1, 1)) * 180 / np.pi)
        if v2 is not None and v3 is not None:
            angles['seg2/seg3'] = float(np.arccos(np.clip(np.dot(v2, v3), -1, 1)) * 180 / np.pi)
    
    return angles


def centerlines_to_json(
    centerline_path: str | Path,
    segmentation_path: str | Path | None = None,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> dict:
    """Convert VTK centerlines to JSON format.
    
    Args:
        centerline_path: Path to VTK PolyData file (.vtp or .vtk)
        segmentation_path: Optional path to original segmentation for radius estimation
        spacing: Voxel spacing (x, y, z) in mm
        
    Returns:
        Dictionary matching sample_json.json structure
    """
    centerline_path = Path(centerline_path)
    centerlines = _load_polydata(centerline_path)
    
    if centerlines.GetNumberOfPoints() == 0:
        return {}
    
    points = vtknp.vtk_to_numpy(centerlines.GetPoints().GetData())
    
    # Load segmentation if provided for radius estimation
    segmentation = None
    if segmentation_path:
        seg_path = Path(segmentation_path)
        if seg_path.suffix.lower() == ".npy":
            seg_data = np.load(str(seg_path))
            if seg_data.ndim == 4:
                seg_data = np.argmax(seg_data, axis=0)
            # Ensure (z, y, x) order
            if seg_data.shape[0] != min(seg_data.shape):
                seg_data = np.moveaxis(seg_data, -1, 0)
            segmentation = (seg_data > 0.5).astype(bool)
    
    # Extract segments from lines
    lines = centerlines.GetLines()
    lines.InitTraversal()
    
    segments = []
    segment_id = 0
    id_list = vtk.vtkIdList()
    
    while lines.GetNextCell(id_list):
        n_points = id_list.GetNumberOfIds()
        if n_points < 2:
            continue
        
        segment_points = np.array([points[id_list.GetId(i)] for i in range(n_points)])
        segments.append({
            'id': segment_id,
            'points': segment_points,
            'point_ids': [id_list.GetId(i) for i in range(n_points)]
        })
        segment_id += 1
    
    # Calculate metrics for each segment
    result = {}
    vessel_id = 1
    
    for seg in segments:
        seg_points = seg['points']
        
        # Basic metrics
        length = _calculate_segment_length(seg_points)
        tortuosity = _calculate_tortuosity(seg_points)
        curvature = _calculate_curvature(seg_points)
        
        # Radius estimation
        if segmentation is not None:
            radii = _estimate_radius_from_segmentation(seg_points, segmentation, spacing)
            radii = radii[radii > 0]  # Filter out invalid radii
        else:
            # Default radius if no segmentation
            radii = np.array([1.0] * len(seg_points))
        
        if len(radii) == 0:
            radii = np.array([1.0])
        
        # Volume estimation (assuming cylindrical: π * r² * length)
        volume = float(np.pi * np.mean(radii)**2 * length) if length > 0 else 0.0
        
        # Statistics
        radius_stats = {
            'mean': float(np.mean(radii)),
            'sd': float(np.std(radii)),
            'median': float(np.median(radii)),
            'min': float(np.min(radii)),
            'q1': float(np.percentile(radii, 25)),
            'q3': float(np.percentile(radii, 75)),
            'max': float(np.max(radii))
        }
        
        curvature_filtered = curvature[curvature > 0]
        if len(curvature_filtered) == 0:
            curvature_filtered = np.array([0.0])
        
        curvature_stats = {
            'mean': float(np.mean(curvature_filtered)),
            'sd': float(np.std(curvature_filtered)),
            'median': float(np.median(curvature_filtered)),
            'min': float(np.min(curvature_filtered)),
            'q1': float(np.percentile(curvature_filtered, 25)),
            'q3': float(np.percentile(curvature_filtered, 75)),
            'max': float(np.max(curvature_filtered))
        }
        
        # Create vessel entry (using blank name as requested)
        vessel_key = str(vessel_id)
        if vessel_key not in result:
            result[vessel_key] = {}
        
        segment_name = ""  # Blank as requested
        if segment_name not in result[vessel_key]:
            result[vessel_key][segment_name] = []
        
        result[vessel_key][segment_name].append({
            'segment': {
                'start': int(seg['point_ids'][0]),
                'end': int(seg['point_ids'][-1])
            },
            'radius': radius_stats,
            'length': length,
            'tortuosity': tortuosity,
            'volume': volume,
            'curvature': curvature_stats
        })
        
        vessel_id += 1
    
    # Find and add bifurcations
    bifurcations = _find_bifurcations(centerlines)
    for bif in bifurcations:
        # Find which vessel this bifurcation belongs to (use first segment)
        if bif['segments']:
            vessel_key = str(bif['segments'][0] + 1)  # +1 because segments are 0-indexed
            if vessel_key not in result:
                result[vessel_key] = {}
            
            bif_name = " bifurcation"
            if bif_name not in result[vessel_key]:
                result[vessel_key][bif_name] = []
            
            # Calculate angles
            angles = _calculate_bifurcation_angles(bif, centerlines, points)
            
            # Estimate radii at bifurcation (simplified)
            midpoint_radius = 1.0  # Default
            if segmentation is not None and bif['midpoint'] < len(points):
                point = points[bif['midpoint']]
                radii = _estimate_radius_from_segmentation([point], segmentation, spacing)
                if len(radii) > 0 and radii[0] > 0:
                    midpoint_radius = float(radii[0])
            
            result[vessel_key][bif_name].append({
                'bifurcation': {
                    'midpoint': bif['midpoint'],
                    'dist_angle': bif['dist_angle'],
                    'points_angle': bif['points_angle'][:3]
                },
                'angles': angles
            })
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert VTK centerlines to JSON format")
    parser.add_argument("centerline", help="Input VTK PolyData file (.vtp or .vtk)")
    parser.add_argument("output", help="Output JSON file")
    parser.add_argument("--segmentation", help="Original segmentation file for radius estimation")
    parser.add_argument("--spacing", nargs=3, type=float, default=[1.0, 1.0, 1.0],
                       help="Voxel spacing in mm (x y z)")
    
    args = parser.parse_args()
    
    result = centerlines_to_json(
        args.centerline,
        args.segmentation,
        tuple(args.spacing)
    )
    
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=4)
    
    print(f"✅ Converted centerlines to JSON: {args.output}")

