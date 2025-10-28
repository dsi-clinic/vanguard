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

import os
from pathlib import Path

import numpy as np

# VTK imports
import vtk
from einops import rearrange
from skimage.measure import label
from skimage.morphology import binary_dilation, binary_erosion, skeletonize
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

# PyVista for 3D vessel visualization
try:
    import pyvista as pv

    # Set backend to headless for offscreen rendering
    try:
        if not os.environ.get("DISPLAY"):
            import pyvista

            pyvista.start_xvfb()
        PYVISTA_AVAILABLE = True
    except Exception:
        PYVISTA_AVAILABLE = True  # Try anyway
except ImportError:
    PYVISTA_AVAILABLE = False
    print("Warning: PyVista not available. 3D vessel visualization will be skipped.")

__all__ = ["extract_adaptive_centerlines"]


def _check_display_available() -> bool:
    """Check if display is available for visualizations."""
    display = os.environ.get("DISPLAY")
    return display and display != ""


def _create_visualization_dir(output_path: Path) -> Path:
    """Create visualization directory based on output path."""
    viz_dir = output_path.parent / f"{output_path.stem}_visualizations"
    viz_dir.mkdir(exist_ok=True)
    return viz_dir


def _visualize_3d_vessels(img: vtkImageData, viz_dir: Path) -> None:
    """Create a 3D rotating volume visualization of vessel data."""
    if not PYVISTA_AVAILABLE or not _check_display_available():
        print("Skipping 3D vessel visualization (PyVista not available or no display)")
        return

    scalars = img.GetPointData().GetScalars()
    if not scalars:
        print("No scalars data available for 3D visualization")
        return

    data = vtknp.vtk_to_numpy(scalars)
    dims = img.GetDimensions()

    # Reshape to 3D
    if len(data.shape) == 1:
        data_3d = data.reshape((dims[2], dims[1], dims[0]))
    else:
        data_3d = data

    # Apply threshold to get binary vessel mask
    vessel_mask = (data_3d > THRESHOLD_DEFAULT).astype(float)

    print("Creating 3D vessel visualization...")
    print(f"Data shape: {data_3d.shape}")
    print(f"Vessel voxels: {np.count_nonzero(vessel_mask)}")

    try:
        # Convert to PyVista grid
        grid = pv.wrap(vessel_mask)

        # Create plotter (off-screen for headless rendering)
        plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])

        # Use threshold to extract vessel points
        thresholded = grid.threshold(0.5, invert=False)

        if thresholded.n_points > 0:
            # Use surface rendering for visible vessels with bright yellow color
            plotter.add_mesh(
                thresholded,
                color="yellow",
                opacity=1.0,
                show_edges=False,
                smooth_shading=True,
            )
        else:
            # Fall back to volume rendering with adjusted opacity
            print("Warning: No threshold surface found, using volume rendering")
            plotter.add_volume(grid, opacity="linear", cmap="hot")

        # Set background to black for better contrast
        plotter.background_color = "black"
        plotter.show_axes()

        # Generate rotating video
        output_path = viz_dir / "vessels_3d.mp4"
        print(f"Saving 3D vessel visualization to: {output_path}")

        plotter.open_movie(str(output_path), framerate=15)
        n_frames = 120

        for i in range(n_frames):
            plotter.camera_position = "yz"
            plotter.camera.elevation = 30
            plotter.camera.azimuth = 180 + i * 360 / n_frames
            plotter.render()
            plotter.write_frame()

        plotter.close()
        print(f"✓ 3D vessel visualization saved: {output_path}")

    except Exception as e:
        print(f"Error creating 3D visualization: {e}")
        import traceback

        traceback.print_exc()


def _visualize_input_data(
    img: vtkImageData, viz_dir: Path, stage: str = "input"
) -> None:
    """Visualize input data before processing as MP4 animation."""
    if not PYVISTA_AVAILABLE or not _check_display_available():
        print("Skipping 3D visualization (PyVista not available or no display)")
        return

    scalars = img.GetPointData().GetScalars()
    if not scalars:
        return

    data = vtknp.vtk_to_numpy(scalars)
    dims = img.GetDimensions()

    # Reshape to 3D
    if len(data.shape) == 1:
        data_3d = data.reshape((dims[2], dims[1], dims[0]))
    else:
        data_3d = data

    # Apply threshold to get binary vessel mask
    vessel_mask = (data_3d > THRESHOLD_DEFAULT).astype(float)

    print(f"Creating {stage} 3D visualization...")
    print(f"Data shape: {data_3d.shape}")
    print(f"Vessel voxels: {np.count_nonzero(vessel_mask)}")

    try:
        # Convert to PyVista grid
        grid = pv.wrap(vessel_mask)

        # Create plotter (off-screen for headless rendering)
        plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])

        # Use threshold to extract vessel points
        thresholded = grid.threshold(0.5, invert=False)

        if thresholded.n_points > 0:
            plotter.add_mesh(
                thresholded,
                color="cyan",
                opacity=1.0,
                show_edges=False,
                smooth_shading=True,
            )
        else:
            plotter.add_volume(grid, opacity="linear", cmap="hot")

        plotter.background_color = "black"
        plotter.show_axes()

        # Generate rotating video
        output_path = viz_dir / f"{stage}_data_visualization.mp4"
        print(f"Saving {stage} visualization to: {output_path}")

        plotter.open_movie(str(output_path), framerate=15)
        n_frames = 120

        for i in range(n_frames):
            plotter.camera_position = "yz"
            plotter.camera.elevation = 30
            plotter.camera.azimuth = 180 + i * 360 / n_frames
            plotter.render()
            plotter.write_frame()

        plotter.close()
        print(f"✓ {stage.title()} visualization saved: {output_path}")

    except Exception as e:
        print(f"Error creating {stage} visualization: {e}")
        import traceback

        traceback.print_exc()


def _visualize_intermediate_stage(
    data: np.ndarray, viz_dir: Path, stage: str, description: str = ""
) -> None:
    """Visualize intermediate processing stages as MP4."""
    if not PYVISTA_AVAILABLE or not _check_display_available():
        print(
            "Skipping intermediate visualization (PyVista not available or no display)"
        )
        return

    if len(data.shape) != DIMENSIONS_3D:
        print(f"Expected 3D data, got shape {data.shape}")
        return

    # Apply threshold to get binary mask
    vessel_mask = (data > THRESHOLD_DEFAULT).astype(float)

    print(f"Creating {stage} 3D visualization...")
    print(f"Data shape: {data.shape}")
    print(f"Vessel voxels: {np.count_nonzero(vessel_mask)}")

    try:
        # Convert to PyVista grid
        grid = pv.wrap(vessel_mask)

        # Create plotter (off-screen for headless rendering)
        plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])

        # Use threshold to extract vessel points
        thresholded = grid.threshold(0.5, invert=False)

        if thresholded.n_points > 0:
            plotter.add_mesh(
                thresholded,
                color="magenta",
                opacity=1.0,
                show_edges=False,
                smooth_shading=True,
            )
        else:
            plotter.add_volume(grid, opacity="linear", cmap="hot")

        plotter.background_color = "black"
        plotter.show_axes()

        # Generate rotating video
        output_path = viz_dir / f"{stage}_stage_visualization.mp4"
        print(f"Saving {stage} stage visualization to: {output_path}")

        plotter.open_movie(str(output_path), framerate=15)
        n_frames = 120

        for i in range(n_frames):
            plotter.camera_position = "yz"
            plotter.camera.elevation = 30
            plotter.camera.azimuth = 180 + i * 360 / n_frames
            plotter.render()
            plotter.write_frame()

        plotter.close()
        print(f"✓ {stage.title()} stage visualization saved: {output_path}")

    except Exception as e:
        print(f"Error creating {stage} visualization: {e}")
        import traceback

        traceback.print_exc()


def _visualize_centerlines(centerlines: vtkPolyData, viz_dir: Path) -> None:
    """Visualize extracted centerlines as 3D rotating MP4."""
    if not PYVISTA_AVAILABLE or not _check_display_available():
        print(
            "Skipping centerlines visualization (PyVista not available or no display)"
        )
        return

    if centerlines.GetNumberOfPoints() == 0:
        print("No centerlines to visualize")
        return

    print("Creating centerlines 3D visualization...")
    print(f"Centerlines points: {centerlines.GetNumberOfPoints()}")

    try:
        # Create plotter (off-screen for headless rendering)
        plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])

        # Add centerlines to the plotter
        plotter.add_mesh(centerlines, color="cyan", line_width=3)

        # Generate rotating video
        output_path = viz_dir / "final_centerlines_3d.mp4"
        print(f"Saving centerlines visualization to: {output_path}")

        plotter.open_movie(str(output_path))
        n_frames = 120

        for i in range(n_frames):
            plotter.camera_position = "yz"
            plotter.camera.elevation = 30
            plotter.camera.azimuth = 180 + i * 360 / n_frames
            plotter.render()
            plotter.write_frame()

        plotter.close()
        print(f"✓ Centerlines visualization saved: {output_path}")

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
    binary_data = (data > THRESHOLD_DEFAULT).astype(np.uint8).reshape((dims[2], dims[1], dims[0]))

    # Apply morphological operations to connect nearby vessel segments
    # Dilation to thicken vessels
    dilated = binary_dilation(binary_data, footprint=np.ones((3, 3, 3)))

    # Erosion to restore original thickness
    processed = binary_erosion(dilated, footprint=np.ones((2, 2, 2)))

    # Convert back to VTK
    return _numpy_to_vtk_image(processed, img.GetSpacing())


def _extract_skeleton_centerlines(img: vtkImageData) -> vtkPolyData:
    """Extract centerlines using 3D skeletonization."""
    scalars = img.GetPointData().GetScalars()
    data = vtknp.vtk_to_numpy(scalars)

    # Reshape to 3D
    dims = img.GetDimensions()
    binary_data = (data > THRESHOLD_DEFAULT).astype(np.uint8).reshape((dims[2], dims[1], dims[0]))

    # Extract 3D skeleton
    skeleton = skeletonize(binary_data)

    # Convert skeleton to VTK PolyData
    return _skeleton_to_polydata(skeleton, img.GetSpacing())


def _skeleton_to_polydata(
    skeleton_array: np.ndarray, spacing: tuple[float, float, float]
) -> vtkPolyData:
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


def _extract_vmtk_centerlines(
    img: vtkImageData, target_points: int = 15000
) -> vtkPolyData:
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
) -> None:
    """Extract centerlines using the most appropriate method for the input data.

    Args:
        input_segmentation_path: Path to input mask (.nii.gz/.nrrd)
        output_centerline_path: Path to output centerlines (.vtp/.vtk)
        force_method: Force specific method ("vmtk" or "skeleton")
        target_points: Target points for surface decimation
        curve_sampling_mm: Resampling step for VMTK centerlines
        enable_visualizations: Whether to create intermediate visualizations
    """
    ipath = Path(input_segmentation_path)
    opath = Path(output_centerline_path)

    # Create visualization directory (if visualizations are enabled)
    viz_dir = None
    if enable_visualizations:
        viz_dir = _create_visualization_dir(opath)

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
        if data.ndim != DIMENSIONS_3D:
            raise ValueError(f"expected 3D nrrd, got shape {tuple(data.shape)}")

        vol_zyx = (
            rearrange(data, "x y z -> z y x") if header.get("dimension") == DIMENSIONS_3D else data
        )
        sdirs = header.get("space directions")
        if sdirs is None:
            raise ValueError("missing 'space directions' in nrrd header")
        spacing_xyz = tuple(float(np.linalg.norm(np.asarray(v))) for v in sdirs)
        spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
        img = _numpy_to_vtk_image(vol_zyx.astype(np.float32, copy=False), spacing_zyx)
    else:
        raise ValueError(f"unsupported input: {ipath}")

    # 0. Create 3D vessel visualization first (before any processing)
    if enable_visualizations and viz_dir:
        print("Creating 3D vessel visualization...")
        _visualize_3d_vessels(img, viz_dir)

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

        # 2. Visualize preprocessed data
        if enable_visualizations and viz_dir:
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
                    data_3d, viz_dir, "preprocessed", "Vessel preprocessing applied"
                )

        centerlines = _extract_vmtk_centerlines(preprocessed_img, target_points)
    elif method == "skeletonization":
        print("Using skeletonization...")

        # 2. Visualize skeletonization process
        if enable_visualizations and viz_dir:
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

        centerlines = _extract_skeleton_centerlines(img)
    else:
        print(f"Unknown method: {method}, falling back to skeletonization...")
        centerlines = _extract_skeleton_centerlines(img)

    # 3. Visualize final centerlines
    if enable_visualizations and viz_dir:
        print("Creating final centerlines visualization...")
        _visualize_centerlines(centerlines, viz_dir)

    # Write output
    _write_polydata(centerlines, opath)

    # Report results
    if centerlines.GetNumberOfPoints() > 0:
        print(
            f"✅ Successfully extracted {centerlines.GetNumberOfPoints()} centerline points"
        )
        if enable_visualizations and viz_dir:
            print(f"📊 All visualizations saved to: {viz_dir}")
    else:
        print(
            "⚠️  No centerlines extracted - input may not be suitable for centerline extraction"
        )


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
        "--no-visualizations",
        action="store_true",
        help="disable visualization generation",
    )
    args = p.parse_args()

    extract_adaptive_centerlines(
        args.input,
        args.output,
        force_method=args.method,
        target_points=args.target_points,
        curve_sampling_mm=args.curve_sampling_mm,
        enable_visualizations=not args.no_visualizations,
    )
