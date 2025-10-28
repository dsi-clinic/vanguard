"""headless centerline extraction (mask -> surface -> seedless network -> resample)

Dependencies (conda-forge):
  - python==3.11
  - vmtk==1.5.0
  - vtk>=9.2
  - nibabel, nrrd, einops

This file only imports non-rendering VTK modules so it runs on headless HPC nodes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk
from einops import rearrange

# vmtk
from vmtk import vmtkscripts

# vtk: import only the modules we need (no rendering)
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
from vtkmodules.vtkIOGeometry import vtkSTLReader
from vtkmodules.vtkIOImage import vtkNIFTIImageReader
from vtkmodules.vtkIOLegacy import vtkPolyDataReader, vtkPolyDataWriter
from vtkmodules.vtkIOXML import vtkXMLPolyDataReader, vtkXMLPolyDataWriter

__all__ = ["extract_centerlines"]

# Constants
DIMENSIONS_3D = 3
THRESHOLD_DEFAULT = 0.5


def _read_surface(surface_path: Path) -> vtkPolyData:
    """Read a surface (.vtp/.vtk/.stl) into vtkPolyData"""
    ext = surface_path.suffix.lower()
    if ext == ".vtp":
        r = vtkXMLPolyDataReader()
        r.SetFileName(str(surface_path))
        r.Update()
        return r.GetOutput()
    if ext == ".vtk":
        r = vtkPolyDataReader()
        r.SetFileName(str(surface_path))
        r.Update()
        return r.GetOutput()
    if ext == ".stl":
        r = vtkSTLReader()
        r.SetFileName(str(surface_path))
        r.Update()
        return r.GetOutput()
    raise ValueError(f"unsupported surface: {surface_path}")


def _nifti_to_vtk_image(nifti_path: Path) -> vtkImageData:
    """Read .nii/.nii.gz via vtkNIFTIImageReader to preserve spacing/origin"""
    r = vtkNIFTIImageReader()
    r.SetFileName(str(nifti_path))
    r.Update()
    return r.GetOutput()


def _nrrd_to_vtk_image(nrrd_path: Path) -> vtkImageData:
    """Read .nrrd via pynrrd -> numpy, then map to vtkImageData with correct spacing.

    volumes: (Z, Y, X) binary mask in {0,1}
    """
    import nrrd  # local import to keep top-level clean

    data, header = nrrd.read(str(nrrd_path))
    if data.ndim != DIMENSIONS_3D:
        raise ValueError(f"expected 3D nrrd, got shape {tuple(data.shape)}")

    # most nrrd images are stored (X, Y, Z); rearrange to (Z, Y, X) for clarity
    vol_zyx = (
        rearrange(data, "x y z -> z y x")
        if header.get("dimension") == DIMENSIONS_3D
        else data
    )

    # infer voxel spacing (mm) from space directions; take norms, then map to (Z,Y,X)
    sdirs = header.get("space directions")
    if sdirs is None:
        raise ValueError("missing 'space directions' in nrrd header")
    spacing_xyz = tuple(float(np.linalg.norm(np.asarray(v))) for v in sdirs)
    spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])

    return _numpy_to_vtk_image(vol_zyx.astype(np.float32, copy=False), spacing_zyx)


def _numpy_to_vtk_image(
    vol_zyx: np.ndarray, spacing_zyx: tuple[float, float, float]
) -> vtkImageData:
    """Map a (Z, Y, X) numpy array to vtkImageData.

    use Fortran-order ravel to match VTK's x-fastest memory layout
    """
    nz, ny, nx = map(int, vol_zyx.shape)
    img = vtkImageData()
    img.SetDimensions(nx, ny, nz)
    img.SetSpacing(spacing_zyx[2], spacing_zyx[1], spacing_zyx[0])
    img.SetOrigin(0.0, 0.0, 0.0)
    scalars = vtknp.numpy_to_vtk(
        vol_zyx.ravel(order="F"),
        deep=True,
        array_type=vtkTypeFloat32Array().GetDataType(),
    )
    scalars.SetName("Scalars")
    img.GetPointData().SetScalars(scalars)
    return img


def _mask_to_surface(img: vtkImageData, level: float) -> vtkPolyData:
    """Marching cubes (flying edges) on a binary image (0/1) at threshold = level.

    cleans, triangles, and keeps largest connected component
    """
    # For binary data, we don't need thresholding - use the image directly
    # The marching cubes will work directly on the binary data

    mc = vtkFlyingEdges3D()
    mc.SetInputData(img)  # Use the original image directly
    mc.SetValue(0, 0.5)  # Use 0.5 as the isovalue for binary data
    mc.Update()

    tri = vtkTriangleFilter()
    tri.SetInputConnection(mc.GetOutputPort())
    tri.Update()

    clean = vtkCleanPolyData()
    clean.SetInputConnection(tri.GetOutputPort())
    clean.Update()

    # keep the largest connected region to avoid speckles
    conn = vtkPolyDataConnectivityFilter()
    conn.SetInputConnection(clean.GetOutputPort())
    conn.SetExtractionModeToLargestRegion()
    conn.Update()

    return conn.GetOutput()


def _decimate_to_points(surf: vtkPolyData, target_points: int) -> vtkPolyData:
    """Decimate using vtkDecimatePro; choose TargetReduction from current point count"""
    n0 = max(1, surf.GetNumberOfPoints())
    if target_points <= 0 or target_points >= n0:
        return surf

    # fraction of polygons to remove to hit ~target_points
    target_reduction = float(max(0.0, min(0.99, 1.0 - (target_points / float(n0)))))

    dec = vtkDecimatePro()
    dec.SetInputData(surf)
    dec.SetTargetReduction(target_reduction)
    dec.PreserveTopologyOn()  # avoids hole creation; may cap achievable reduction
    dec.BoundaryVertexDeletionOn()  # no-op on closed surfaces; ok to keep
    # dec.SplittingOn()               # optional: allowed; irrelevant if PreserveTopologyOn
    dec.Update()

    clean = vtkCleanPolyData()
    clean.SetInputConnection(dec.GetOutputPort())
    clean.Update()
    return clean.GetOutput()


def _write_polydata(poly: vtkPolyData, out_path: Path) -> None:
    """Write .vtp via vtkXMLPolyDataWriter or .vtk via vtkPolyDataWriter"""
    sx = out_path.suffix.lower()
    if sx == ".vtp":
        w = vtkXMLPolyDataWriter()
        w.SetFileName(str(out_path))
        w.SetInputData(poly)
        w.SetDataModeToBinary()
        if w.Write() != 1:
            raise RuntimeError(f"failed to write {out_path}")
        return
    if sx == ".vtk":
        w = vtkPolyDataWriter()
        w.SetFileName(str(out_path))
        w.SetInputData(poly)
        w.SetFileTypeToBinary()
        if w.Write() != 1:
            raise RuntimeError(f"failed to write {out_path}")
        return
    raise ValueError(f"unsupported output extension: {out_path.suffix}")


def extract_centerlines(
    input_segmentation_path: str,
    output_centerline_path: str,
    *,
    marching_level: float = 0.5,
    target_points: int = 15000,
    curve_sampling_mm: float = 0.6,
) -> None:
    """Given a binary vessel mask (.nii.gz/.nrrd) or a surface (.vtp/.vtk/.stl),

    convert masks to a surface (marching cubes), decimate to ~target_points,
    run VMTK seedless network extraction, resample polylines to ~curve_sampling_mm,
    and write a VTK PolyData file (.vtp/.vtk) to output_centerline_path.

    shapes: volumes are (Z, Y, X); vertices are (N, 3) in mm
    """
    ipath = Path(input_segmentation_path)
    opath = Path(output_centerline_path)

    # surface or image path dispatch
    if ipath.suffix.lower() in {".vtp", ".vtk", ".stl"}:
        surface = _read_surface(ipath)
    elif ipath.suffix.lower() in {".nii", ".gz"} and ipath.name.endswith(".nii.gz"):
        surface = _mask_to_surface(_nifti_to_vtk_image(ipath), marching_level)
    elif ipath.suffix.lower() in {".nii"}:
        surface = _mask_to_surface(_nifti_to_vtk_image(ipath), marching_level)
    elif ipath.suffix.lower() == ".nrrd":
        surface = _mask_to_surface(_nrrd_to_vtk_image(ipath), marching_level)
    else:
        raise ValueError(f"unsupported input: {ipath}")

    decimated = _decimate_to_points(surface, target_points)

    # seedless network extraction (no endpoints required)
    net = vmtkscripts.vmtkNetworkExtraction()
    net.Surface = decimated
    net.Execute()
    network = net.Network

    # resample centerlines to uniform spacing (~curve_sampling_mm, mm units)
    res = vmtkscripts.vmtkCenterlineResampling()
    res.Centerlines = network
    res.Length = float(curve_sampling_mm)
    res.Execute()
    resampled = res.Centerlines

    _write_polydata(resampled, opath)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="headless seedless centerline extraction (mask/surface -> VMTK network -> resample)"
    )
    p.add_argument("input", help="input .nii.gz/.nrrd or .vtp/.vtk/.stl")
    p.add_argument("output", help="output .vtp or .vtk")
    p.add_argument(
        "--marching-level", type=float, default=0.5, help="iso-value for marching cubes"
    )
    p.add_argument(
        "--target-points",
        type=int,
        default=15000,
        help="target vertex count before VMTK",
    )
    p.add_argument(
        "--curve-sampling-mm",
        type=float,
        default=0.6,
        help="resampling step along centerlines (mm)",
    )
    args = p.parse_args()

    extract_centerlines(
        args.input,
        args.output,
        marching_level=args.marching_level,
        target_points=args.target_points,
        curve_sampling_mm=args.curve_sampling_mm,
    )


# Example function for saving NIfTI outputs in segmentation scripts
def save_nifti_outputs(
    preprocessed_array: np.ndarray,
    breast_mask: np.ndarray,
    final_image: np.ndarray,
    output_dir: Path,
) -> None:
    """Save segmentation outputs as NIfTI files.

    Converts from x,y,z format (used internally) back to z,x,y (SimpleITK/ITK convention).
    """
    print("\nSaving NIfTI outputs...")

    # convert x,y,z back to z,x,y for SimpleITK
    # reverse the preprocessing transformation: x,y,z → y,x,z → z,x,y, then unflip
    def to_zxy(array: np.ndarray) -> np.ndarray:
        """Convert from x,y,z to z,x,y and undo flip."""
        return array[::-1].transpose(2, 1, 0)

    # save breast mask (convert to uint8 for mask, SimpleITK doesn't support float16)
    breast_mask_zxy = to_zxy(breast_mask.astype(np.uint8))
    breast_img = sitk.GetImageFromArray(breast_mask_zxy)
    sitk.WriteImage(breast_img, str(output_dir / "breast_mask.nii.gz"))
    print("  Saved breast_mask.nii.gz")

    # save argmax labels (0=background, 1=vessels, 2=FGT)
    labels_xyz = np.argmax(final_image, axis=0).astype(np.uint8)
    labels_zxy = to_zxy(labels_xyz)
    labels_img = sitk.GetImageFromArray(labels_zxy)
    sitk.WriteImage(labels_img, str(output_dir / "segmentation_labels.nii.gz"))
    print("  Saved segmentation_labels.nii.gz (0=background, 1=vessels, 2=FGT)")

    # save individual channel probabilities
    for i, channel_name in enumerate(["fgt", "vessels", "background"]):
        channel_xyz = final_image[i].astype(np.float32)
        channel_zxy = to_zxy(channel_xyz)
        channel_img = sitk.GetImageFromArray(channel_zxy)
        sitk.WriteImage(
            channel_img, str(output_dir / f"segmentation_{channel_name}.nii.gz")
        )
        print(f"  Saved segmentation_{channel_name}.nii.gz")
