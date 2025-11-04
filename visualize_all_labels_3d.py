#!/usr/bin/env python3
"""
Create 3D visualizations for each label in the NRRD file separately.
Can run locally with display or attempt offscreen rendering.
"""

import argparse
from pathlib import Path
import os
import numpy as np
import nrrd

try:
    import pyvista as pv
    
    # Only force offscreen if no display
    if not os.environ.get("DISPLAY"):
        os.environ['PYVISTA_OFF_SCREEN'] = 'true'
        os.environ['PYVISTA_USE_PANEL'] = 'false'
        os.environ['VTK_USE_X'] = '0'
        pv.OFF_SCREEN = True
        print("No DISPLAY detected, using offscreen rendering")
    else:
        pv.OFF_SCREEN = False
        print("DISPLAY detected, using interactive rendering")
    
    PYVISTA_AVAILABLE = True
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
        plotter = pv.Plotter(off_screen=pv.OFF_SCREEN, window_size=[1920, 1080])
        
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
        
        if pv.OFF_SCREEN:
            # Generate rotating video
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
        else:
            # Interactive mode - show the plotter
            print(f"  Displaying interactive 3D visualization...")
            print(f"  Close the window to continue...")
            plotter.show()
            print(f"  Interactive visualization closed")
        
    except Exception as e:
        print(f"\nERROR creating visualization: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Create 3D rotating visualizations for each label in NRRD mask"
    )
    parser.add_argument(
        "input",
        help="Input NRRD file",
    )
    parser.add_argument(
        "output_dir",
        help="Output directory for visualization videos",
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
    parser.add_argument(
        "--skip-background",
        action="store_true",
        help="Skip background label (label 0)",
    )
    
    args = parser.parse_args()
    
    # Load the file to find all labels
    nrrd_path = Path(args.input)
    data, header = nrrd.read(str(nrrd_path))
    unique_values = np.unique(data)
    
    print(f"Found labels: {unique_values}")
    
    # Define colors for each label
    label_colors = {
        0: "gray",      # Background
        1: "yellow",    # Dense tissue
        2: "red",       # Vessels
    }
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualization for each label
    for label in unique_values:
        if label == 0 and args.skip_background:
            print(f"\nSkipping background label (0)")
            continue
        
        color = label_colors.get(label, "cyan")
        output_path = output_dir / f"label_{label}_3d.mp4"
        
        create_3d_visualization(
            str(nrrd_path),
            str(output_path),
            label_value=int(label),
            color=color,
            opacity=1.0,
            n_frames=args.frames,
            framerate=args.framerate,
        )
    
    print(f"\n✓ All visualizations completed!")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()

