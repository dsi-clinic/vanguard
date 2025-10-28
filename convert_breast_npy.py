#!/usr/bin/env python3
"""Convert breast tissue .npy files to .nii.gz format with proper spacing."""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

# Constants
DIMENSIONS_3D = 3
DIMENSIONS_4D = 4
THRESHOLD_DEFAULT = 0.5


def convert_breast_npy_to_nifti(
    npy_path: Path,
    output_path: Path,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    channel: int = 1,
) -> None:
    """Convert a breast tissue .npy file to .nii.gz format."""
    # Load the numpy array
    data = np.load(npy_path)

    # Handle vessel segmentation (3 channels)
    if data.ndim == DIMENSIONS_4D:
        print(f"Detected 4D vessel segmentation with shape {data.shape}")
        print(f"Using channel {channel}")
        data = data[channel]  # Extract vessel channel (channel 1)
        print(f"Shape after channel extraction: {data.shape}")

    # Ensure it's 3D
    if data.ndim != DIMENSIONS_3D:
        raise ValueError(f"Expected 3D array after processing, got shape {data.shape}")

    # Convert to binary mask (threshold at 0.5)
    binary_data = (data > THRESHOLD_DEFAULT).astype(np.uint8)

    # Create NIfTI image with specified spacing
    nii_img = nib.Nifti1Image(binary_data, affine=np.eye(4))
    nii_img.header.set_zooms(spacing)

    # Save as .nii.gz
    nib.save(nii_img, output_path)
    print(f"Converted {npy_path} -> {output_path}")
    print(f"  Shape: {binary_data.shape}")
    print(f"  Non-zero voxels: {np.count_nonzero(binary_data)}")
    print(
        f"  Non-zero percentage: {100 * np.count_nonzero(binary_data) / binary_data.size:.2f}%"
    )


def main() -> None:
    """Main function to convert breast tissue .npy files to .nii.gz format."""
    parser = argparse.ArgumentParser(
        description="Convert breast tissue .npy files to .nii.gz"
    )
    parser.add_argument("input_dir", help="Directory containing .npy files")
    parser.add_argument("output_dir", help="Output directory for .nii.gz files")
    parser.add_argument(
        "--spacing",
        nargs=3,
        type=float,
        default=[1.0, 1.0, 1.0],
        help="Voxel spacing (x y z)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=2,
        help="Maximum number of files to convert for testing",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Find all .npy files
    npy_files = list(input_dir.glob("*.npy"))
    print(f"Found {len(npy_files)} .npy files")

    # Convert up to max_files for testing
    for _i, npy_file in enumerate(npy_files[: args.max_files]):
        output_file = output_dir / f"{npy_file.stem}.nii.gz"
        try:
            convert_breast_npy_to_nifti(npy_file, output_file, args.spacing, channel=1)
        except Exception as e:
            print(f"Error converting {npy_file}: {e}")


if __name__ == "__main__":
    main()
