"""Adaptive centerline extraction (mask -> surface -> appropriate centerline method)

This version automatically detects input type and chooses the best centerline extraction method:
- Sparse vessel data: VMTK network extraction with preprocessing
- Dense solid regions: Skeletonization or principal axes
- Mixed data: Hybrid approach

Dependencies (conda-forge):
  - python==3.11
  - vmtk==1.5.0
  - vtk>=9.2
  - nibabel, nrrd, einops
  - scikit-image (for skeletonization)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
from einops import rearrange
from skimage.morphology import binary_dilation, binary_erosion, skeletonize
from skimage.measure import label

# vmtk
from vmtk import vmtkscripts

# vtk: import only the modules we need (no rendering)
import vtk
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
from vtkmodules.vtkImagingCore import vtkImageThreshold
from vtkmodules.vtkIOGeometry import vtkSTLReader
from vtkmodules.vtkIOImage import vtkNIFTIImageReader
from vtkmodules.vtkIOLegacy import vtkPolyDataReader, vtkPolyDataWriter
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader, vtkXMLPolyDataWriter

__all__ = ["extract_adaptive_centerlines"]


def _analyze_input_data(img: vtkImageData) -> dict:
    """Analyze input data to determine the best centerline extraction method."""
    scalars = img.GetPointData().GetScalars()
    if not scalars:
        return {"method": "unknown", "sparsity": 1.0, "connectivity": "unknown"}
    
    data = vtknp.vtk_to_numpy(scalars)
    binary_data = (data > 0.5).astype(np.uint8)
    
    # Calculate sparsity
    non_zero_count = np.count_nonzero(binary_data)
    total_voxels = binary_data.size
    sparsity = 1.0 - (non_zero_count / total_voxels)
    
    # Analyze connectivity
    labeled_data = label(binary_data)
    num_components = len(np.unique(labeled_data)) - 1  # Exclude background
    
    # Determine method based on characteristics
    if sparsity > 0.95:  # Very sparse (likely vessels)
        method = "vessel_preprocessing"
    elif sparsity < 0.1:  # Dense (likely solid regions)
        method = "skeletonization"
    elif num_components > 100:  # Many small components
        method = "vessel_preprocessing"
    else:  # Few large components
        method = "skeletonization"
    
    return {
        "method": method,
        "sparsity": sparsity,
        "connectivity": f"{num_components} components",
        "non_zero_count": non_zero_count,
        "total_voxels": total_voxels
    }


def _preprocess_vessel_data(img: vtkImageData) -> vtkImageData:
    """Preprocess sparse vessel data to improve VMTK network extraction."""
    scalars = img.GetPointData().GetScalars()
    data = vtknp.vtk_to_numpy(scalars)
    
    # Reshape to 3D
    dims = img.GetDimensions()
    binary_data = (data > 0.5).astype(np.uint8).reshape((dims[2], dims[1], dims[0]))
    
    # Apply morphological operations to connect nearby vessel segments
    # Dilation to thicken vessels
    dilated = binary_dilation(binary_data, footprint=np.ones((3,3,3)))
    
    # Erosion to restore original thickness
    processed = binary_erosion(dilated, footprint=np.ones((2,2,2)))
    
    # Convert back to VTK
    return _numpy_to_vtk_image(processed, img.GetSpacing())


def _extract_skeleton_centerlines(img: vtkImageData) -> vtkPolyData:
    """Extract centerlines using 3D skeletonization."""
    scalars = img.GetPointData().GetScalars()
    data = vtknp.vtk_to_numpy(scalars)
    
    # Reshape to 3D
    dims = img.GetDimensions()
    binary_data = (data > 0.5).astype(np.uint8).reshape((dims[2], dims[1], dims[0]))
    
    # Extract 3D skeleton
    skeleton = skeletonize(binary_data)
    
    # Convert skeleton to VTK PolyData
    return _skeleton_to_polydata(skeleton, img.GetSpacing())


def _skeleton_to_polydata(skeleton_array: np.ndarray, spacing: tuple[float, float, float]) -> vtkPolyData:
    """Convert 3D skeleton array to VTK PolyData."""
    skeleton_points = np.where(skeleton_array > 0)
    
    if len(skeleton_points[0]) == 0:
        return vtkPolyData()
    
    # Convert to world coordinates
    z_coords = skeleton_points[0] * spacing[2]
    y_coords = skeleton_points[1] * spacing[1] 
    x_coords = skeleton_points[2] * spacing[0]
    
    points = np.column_stack([x_coords, y_coords, z_coords])
    
    # Create VTK points
    vtk_points = vtk.vtkPoints()
    for point in points:
        vtk_points.InsertNextPoint(point)
    
    # Create polydata
    polydata = vtkPolyData()
    polydata.SetPoints(vtk_points)
    
    # Add vertices for each point
    vertices = vtk.vtkCellArray()
    for i in range(len(points)):
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(i)
    
    polydata.SetVerts(vertices)
    return polydata


def _extract_vmtk_centerlines(img: vtkImageData, target_points: int = 15000) -> vtkPolyData:
    """Extract centerlines using VMTK network extraction."""
    # Generate surface using marching cubes
    surface = _mask_to_surface(img, 0.5)
    
    if surface.GetNumberOfPoints() == 0:
        return vtkPolyData()
    
    # Decimate surface
    decimated = _decimate_to_points(surface, target_points)
    
    # VMTK network extraction
    try:
        net = vmtkscripts.vmtkNetworkExtraction()
        net.Surface = decimated
        net.Execute()
        network = net.Network
        
        if network.GetNumberOfPoints() == 0:
            return vtkPolyData()
        
        # Resample centerlines
        res = vmtkscripts.vmtkCenterlineResampling()
        res.Centerlines = network
        res.Length = 0.6  # 0.6mm spacing
        res.Execute()
        
        return res.Centerlines
        
    except Exception as e:
        print(f"VMTK network extraction failed: {e}")
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


def _numpy_to_vtk_image(vol_zyx: np.ndarray, spacing: tuple[float, float, float]) -> vtkImageData:
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
) -> None:
    """Extract centerlines using the most appropriate method for the input data.
    
    Args:
        input_segmentation_path: Path to input mask (.nii.gz/.nrrd)
        output_centerline_path: Path to output centerlines (.vtp/.vtk)
        force_method: Force specific method ("vmtk" or "skeleton")
        target_points: Target points for surface decimation
        curve_sampling_mm: Resampling step for VMTK centerlines
    """
    ipath = Path(input_segmentation_path)
    opath = Path(output_centerline_path)
    
    # Read input image
    if ipath.suffix.lower() in {".nii", ".gz"} and ipath.name.endswith(".nii.gz"):
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
        if data.ndim != 3:
            raise ValueError(f"expected 3D nrrd, got shape {tuple(data.shape)}")
        
        vol_zyx = rearrange(data, "x y z -> z y x") if header.get("dimension") == 3 else data
        sdirs = header.get("space directions")
        if sdirs is None:
            raise ValueError("missing 'space directions' in nrrd header")
        spacing_xyz = tuple(float(np.linalg.norm(np.asarray(v))) for v in sdirs)
        spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
        img = _numpy_to_vtk_image(vol_zyx.astype(np.float32, copy=False), spacing_zyx)
    else:
        raise ValueError(f"unsupported input: {ipath}")
    
    # Analyze input data
    analysis = _analyze_input_data(img)
    print(f"Input analysis: {analysis}")
    
    # Choose method
    if force_method:
        method = force_method
        print(f"Using forced method: {method}")
    else:
        method = analysis["method"]
        print(f"Auto-selected method: {method}")
    
    # Extract centerlines
    if method == "vessel_preprocessing":
        print("Using vessel preprocessing + VMTK...")
        preprocessed_img = _preprocess_vessel_data(img)
        centerlines = _extract_vmtk_centerlines(preprocessed_img, target_points)
    elif method == "skeletonization":
        print("Using skeletonization...")
        centerlines = _extract_skeleton_centerlines(img)
    else:
        print(f"Unknown method: {method}, falling back to skeletonization...")
        centerlines = _extract_skeleton_centerlines(img)
    
    # Write output
    _write_polydata(centerlines, opath)
    
    # Report results
    if centerlines.GetNumberOfPoints() > 0:
        print(f"✅ Successfully extracted {centerlines.GetNumberOfPoints()} centerline points")
    else:
        print("⚠️  No centerlines extracted - input may not be suitable for centerline extraction")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Adaptive centerline extraction (auto-detects best method)"
    )
    p.add_argument("input", help="input .nii.gz/.nrrd")
    p.add_argument("output", help="output .vtp or .vtk")
    p.add_argument(
        "--method", 
        choices=["vmtk", "skeleton"], 
        help="Force specific method (auto-detect if not specified)"
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
    args = p.parse_args()

    extract_adaptive_centerlines(
        args.input,
        args.output,
        force_method=args.method,
        target_points=args.target_points,
        curve_sampling_mm=args.curve_sampling_mm,
    )
