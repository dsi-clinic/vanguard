#!/usr/bin/env python3
"""
Create 3D visualizations for vessels (label 2) in the NRRD file.
Can run locally with display or attempt offscreen rendering.
"""

import argparse
from pathlib import Path
import os
import numpy as np
import nrrd

# Detect if we're running without a DISPLAY (remote/headless)
NO_DISPLAY = not os.environ.get("DISPLAY")

# Force offscreen rendering for remote servers (set BEFORE importing pyvista)
if NO_DISPLAY:
    os.environ.setdefault('PYVISTA_OFF_SCREEN', 'true')
    os.environ.setdefault('PYVISTA_USE_PANEL', 'false')
    os.environ.setdefault('MESA_GL_VERSION_OVERRIDE', '3.3')
    print("No DISPLAY detected; enabling offscreen mode (Xvfb if available)")
else:
    print("DISPLAY detected; interactive rendering available")

try:
    import pyvista as pv
    
    # Always use offscreen for video generation
    pv.OFF_SCREEN = True

    if NO_DISPLAY and hasattr(pv, "start_xvfb"):
        try:
            pv.start_xvfb()
            print("Started Xvfb for offscreen rendering")
        except Exception as xvfb_err:
            print(f"Warning: could not start Xvfb automatically: {xvfb_err}")
            print("If rendering fails, install vtk-osmesa or run inside xvfb-run.")

    # Try to reduce rendering overhead (may not be available in all PyVista versions)
    try:
        if hasattr(pv.global_theme, 'anti_aliasing'):
            pv.global_theme.anti_aliasing = False
    except (AttributeError, Exception):
        pass  # Ignore if attribute doesn't exist or can't be set
    
    PYVISTA_AVAILABLE = True
    print("PyVista loaded with offscreen rendering enabled")
except ImportError:
    PYVISTA_AVAILABLE = False
    print("Error: PyVista not available. Please install it to visualize masks.")


def load_nrrd_mask(nrrd_path: Path, label_value: int):
    """Load NRRD mask and extract specific label."""
    print(f"Loading: {nrrd_path}")
    data, header = nrrd.read(str(nrrd_path))
    
    print(f"  Shape: {data.shape}")
    print(f"  Dtype: {data.dtype}")
    unique_values = np.unique(data)
    print(f"  Unique values: {unique_values}")
    
    # Extract specific label
    if label_value == 0:
        # Background - show everything that's NOT the other labels
        mask = (data == 0).astype(np.float32)
        label_name = "Background"
    else:
        mask = (data == label_value).astype(np.float32)
        # Try to get label name from header
        label_name = f"Label_{label_value}"
        for key in header.keys():
            if 'Segment' in key and 'Name' in key:
                seg_num = key.split('Segment')[1].split('_')[0]
                try:
                    if int(seg_num) == label_value:
                        label_name = header[key]
                        break
                except ValueError:
                    pass
    
    print(f"  Extracted {label_name}: {np.count_nonzero(mask)} voxels")
    
    # Get spacing from header
    sdirs = header.get("space directions")
    if sdirs is not None:
        spacing_xyz = tuple(float(np.linalg.norm(np.asarray(v))) for v in sdirs)
        spacing = spacing_xyz
    else:
        spacing = (1.0, 1.0, 1.0)
    
    print(f"  Spacing: {spacing}")
    
    return mask, spacing, label_name


def create_3d_visualization(
    nrrd_path: str,
    output_path: str,
    label_value: int,
    color: str = "cyan",
    opacity: float = 1.0,
    n_frames: int = 120,
    framerate: int = 15,
):
    """Create a 3D rotating visualization of the mask."""
    if not PYVISTA_AVAILABLE:
        raise ImportError("PyVista is required for visualization")
    
    nrrd_path = Path(nrrd_path)
    output_path = Path(output_path)
    
    # Load mask
    mask, spacing, label_name = load_nrrd_mask(nrrd_path, label_value)
    
    if np.count_nonzero(mask) == 0:
        print(f"  Warning: No voxels for {label_name}, skipping visualization")
        return False
    
    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCreating 3D visualization for {label_name}...")
    print(f"Output: {output_path}")
    
    try:
        print("  Creating PyVista grid...")
        # Convert to PyVista grid
        grid = pv.wrap(mask)
        grid.spacing = spacing
        
        print("  Creating plotter...")
        # Always use offscreen for video generation (requires vtk-osmesa for servers without X)
        plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
        
        print("  Extracting surface...")
        # Extract surface using threshold
        thresholded = grid.threshold(0.5, invert=False)
        
        if thresholded.n_points > 0:
            print(f"  Surface has {thresholded.n_points:,} points")
            plotter.add_mesh(
                thresholded,
                color=color,
                opacity=opacity,
                show_edges=False,
                smooth_shading=True,
            )
        else:
            print("  Warning: No surface found, using volume rendering")
            plotter.add_volume(grid, opacity="linear", cmap="hot")
        
        plotter.background_color = "black"
        plotter.show_axes()
        
        # Always generate rotating video (offscreen mode)
        print(f"  Generating {n_frames} frame rotating video (framerate={framerate})...")
        plotter.open_movie(str(output_path), framerate=framerate)
        
        for i in range(n_frames):
            plotter.camera_position = "yz"
            plotter.camera.elevation = 30
            plotter.camera.azimuth = 180 + i * 360 / n_frames
            plotter.render()
            plotter.write_frame()
            
            if (i + 1) % 30 == 0:
                print(f"    Progress: {i + 1}/{n_frames} frames ({100*(i+1)/n_frames:.1f}%)")
        
        plotter.close()
        print(f"\n✓ 3D visualization saved: {output_path}")
        
        # Verify file was created
        if output_path.exists():
            file_size = output_path.stat().st_size / (1024 * 1024)  # MB
            print(f"  File size: {file_size:.2f} MB")
        else:
            print(f"  WARNING: Output file not found at {output_path}")
        
    except Exception as e:
        error_msg = str(e).lower()
        print(f"\nERROR creating visualization: {e}")
        
        # Provide helpful guidance for X server errors
        if "x server" in error_msg or "display" in error_msg or "bad x server connection" in error_msg:
            print("\n" + "="*60)
            print("OFFSCREEN RENDERING ERROR:")
            print("="*60)
            print("This script requires offscreen rendering for remote servers.")
            print("Two common fixes:")
            print("  1) Launch the script under a virtual framebuffer:")
            print("       xvfb-run -s '-screen 0 1920x1080x24' python visualize_all_labels_3d.py ...")
            print("  2) Install vtk-osmesa to enable pure offscreen rendering:")
            print("  conda install -c conda-forge vtk-osmesa")
            print("  or")
            print("  pip install vtk-osmesa")
            print("="*60)
        
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Create 3D rotating visualization for vessels (label 2) in NRRD mask"
    )
    parser.add_argument(
        "input",
        help="Input NRRD file",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="centerline_outputs",
        help="Output directory for visualization videos (default: centerline_outputs)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=120,
        help="Number of frames in video (default: 120)",
    )
    parser.add_argument(
        "--framerate",
        type=int,
        default=15,
        help="Framerate for video (default: 15)",
    )
    
    args = parser.parse_args()
    
    # Load the file to check if label 2 exists
    nrrd_path = Path(args.input)
    data, header = nrrd.read(str(nrrd_path))
    unique_values = np.unique(data)
    
    print(f"Found labels: {unique_values}")
    
    # Only visualize label 2 (vessels)
    label_value = 2
    if label_value not in unique_values:
        print(f"\nERROR: Label {label_value} (vessels) not found in the NRRD file!")
        print(f"Available labels: {unique_values}")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualization for vessels (label 2)
    color = "red"  # Vessels color
    output_path = output_dir / f"vessels_3d.mp4"
    
    create_3d_visualization(
        str(nrrd_path),
        str(output_path),
        label_value=label_value,
        color=color,
        opacity=1.0,
        n_frames=args.frames,
        framerate=args.framerate,
    )
    
    print(f"\n✓ Visualization completed!")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()

