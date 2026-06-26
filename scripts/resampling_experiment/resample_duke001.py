#!/usr/bin/env python3
"""STEP 1 of the voxel-spacing experiment: build four versions of DUKE_001.

We start from the original five-timepoint DUKE_001 series and produce four
"versions" that differ only in their through-plane (z) voxel spacing.  The goal
is to later segment each version and see how z spacing affects vessel
segmentation quality.

Why resample *all five* timepoints the same way?  Because downstream we compute
a subtraction image (post-contrast minus pre-contrast).  That subtraction is
only valid if every timepoint sits on the exact same voxel grid, so whatever we
do to one timepoint we must do identically to all of them.

The four versions
------------------
  Version 1 (z=2.0mm, thick / out-of-distribution):
      Downsample z from native (~1.1mm) to 2.0mm to mimic an ISPY1/NACT scan.
      Before downsampling we Gaussian-blur ALONG z only.  This is "anti-aliasing":
      if you throw away samples without first removing the fine detail they
      carry, that fine detail folds back as fake structure (aliasing), like the
      wagon-wheel-spinning-backwards effect in old films.
  Version 2 (z=1.4mm):  take Version 1 and upsample z back up to 1.4mm.
  Version 3 (z=1.0mm):  take Version 1 and upsample z back up to 1.0mm.
  Version 4 (original): the untouched native files, just copied across.

Versions 2 and 3 deliberately start from the already-degraded 2.0mm Version 1.
That mimics the real situation: you receive a thick-slice scan and resample it
to the resolution your model was trained on.  Upsampling cannot recover detail
that the 2.0mm step destroyed, which is exactly the effect we want to study.

All resampling uses SimpleITK with LINEAR interpolation.
"""

import argparse
import shutil
from pathlib import Path

import SimpleITK as sitk

# The five timepoints that make up one DUKE series.
TIMEPOINTS = ["0000", "0001", "0002", "0003", "0004"]

# Folder name -> target z spacing in millimetres (None means "leave untouched").
VERSIONS = {
    "version1_z2.0mm": 2.0,
    "version2_z1.4mm": 1.4,
    "version3_z1.0mm": 1.0,
    "version4_original": None,
}


def resample_z(
    image: sitk.Image,
    new_z_spacing: float,
    antialias: bool,
) -> sitk.Image:
    """Resample a 3-D image to a new z spacing, keeping x and y untouched.

    Args:
        image: input volume (any pixel type; we cast to float32 internally).
        new_z_spacing: desired spacing along the z axis, in millimetres.
        antialias: if True, Gaussian-blur along z BEFORE resampling.  Use this
            only when *down*-sampling (going to a coarser spacing), where
            aliasing is a risk.  Upsampling does not need it.

    Returns:
        A new float32 SimpleITK image on the requested z grid.
    """
    # SimpleITK spacing/size are ordered (x, y, z).
    spacing = list(image.GetSpacing())
    size = list(image.GetSize())

    # Blurring and interpolation need a continuous (float) pixel type.
    image = sitk.Cast(image, sitk.sitkFloat32)

    if antialias:
        # Blur ONLY along z (direction index 2).  We pick a Gaussian width of
        # half the new spacing (in mm): roughly the finest scale the coarser
        # grid can still represent, so finer detail is smoothed away before we
        # drop samples.  RecursiveGaussian takes sigma in physical units (mm).
        sigma_mm = new_z_spacing / 2.0
        image = sitk.RecursiveGaussian(image, sigma=sigma_mm, direction=2)

    # New number of z slices = old physical extent / new spacing.  We keep the
    # total physical length of the volume the same; only the sampling changes.
    new_size = list(size)
    new_size[2] = int(round(size[2] * spacing[2] / new_z_spacing))

    new_spacing = list(spacing)
    new_spacing[2] = new_z_spacing

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    # Keep the volume anchored in the same physical space.
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    return resampler.Execute(image)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default="/gpfs/data/karczmar-lab/MAMA-MIA-syn60868042/images/DUKE_001",
        help="Directory holding the original DUKE_001_000N.nii.gz files.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Where to write the four version subdirectories.",
    )
    parser.add_argument("--case-id", default="DUKE_001")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_root = Path(args.output_root)

    # --- Pre-build Version 1 (2.0mm) per timepoint, since 2 and 3 derive from it.
    print("=" * 70)
    print("Building Version 1 (z=2.0mm) from native, with anti-alias z-blur")
    print("=" * 70)
    v1_images: dict[str, sitk.Image] = {}
    native_spacing = None
    for tp in TIMEPOINTS:
        src = input_dir / f"{args.case_id}_{tp}.nii.gz"
        img = sitk.ReadImage(str(src))
        if native_spacing is None:
            native_spacing = img.GetSpacing()
        v1_images[tp] = resample_z(img, 2.0, antialias=True)
        print(f"  {tp}: native size {img.GetSize()} sp {tuple(round(s,3) for s in img.GetSpacing())}"
              f"  ->  {v1_images[tp].GetSize()} sp {tuple(round(s,3) for s in v1_images[tp].GetSpacing())}")

    print(f"\nNative DUKE_001 spacing (x, y, z) = "
          f"{tuple(round(s,4) for s in native_spacing)} mm\n")

    # --- Now write every version.
    for version, target_z in VERSIONS.items():
        case_out = out_root / version / args.case_id
        case_out.mkdir(parents=True, exist_ok=True)
        print("-" * 70)
        print(f"Writing {version}  (target z = {target_z} mm)")

        for tp in TIMEPOINTS:
            dst = case_out / f"{args.case_id}_{tp}.nii.gz"

            if version == "version4_original":
                # No resampling: copy the original file byte-for-byte.
                shutil.copy2(input_dir / f"{args.case_id}_{tp}.nii.gz", dst)
                out_img = sitk.ReadImage(str(dst))
            elif version == "version1_z2.0mm":
                out_img = v1_images[tp]
                sitk.WriteImage(out_img, str(dst))
            else:
                # Versions 2 and 3: upsample FROM the 2.0mm Version 1.
                out_img = resample_z(v1_images[tp], target_z, antialias=False)
                sitk.WriteImage(out_img, str(dst))

            print(f"  {tp}: size {out_img.GetSize()} "
                  f"sp {tuple(round(s,3) for s in out_img.GetSpacing())}  -> {dst}")

    print("\nSTEP 1 complete: four versions written under", out_root)


if __name__ == "__main__":
    main()
