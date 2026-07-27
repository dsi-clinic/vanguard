"""Top-level vessel segmentation, ported from:
  KineticAnalysisFunctions/SegVessel.m         (HR + ultrafast path)
  KineticAnalysisFunctions/SegVessel_U.m        (ultrafast-only path)
  KineticAnalysisFunctions/VesselAnalysis/seg_vessel.m
  KineticAnalysisFunctions/VesselAnalysis/segmentVessels3D.m

All arrays are kept in MATLAB (dim1, dim2, dim3) = (row, col, slice) order so index
logic matches the source; numpy axis 0/1/2 == MATLAB dim 1/2/3.

Ported into vanguard (2026-07-27) from the standalone validation workspace
``matlab-conv-2/vessel_pipeline`` (workspace saritbose), where this port was validated
against the lab's production ``merged_tc4d`` skeleton across 6 UChicago exams: the
DYN-branch quality-gated complement (see ``preprocessing.complement``) adds real,
interior, vessel-like voxels the merge misses at ~0% mask-edge-hug. See
``preprocessing/complement.py`` for how this module's output is combined with the
HR+UFAST merged skeleton in the production pipeline.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from .jerman import vesselness3d
from .matlab_ops import (
    bwareaopen,
    edge3_approxcanny,
    graythresh,
    imdilate_cube,
    imdilate_disk2_perslice,
    imdilate_sphere1,
    imfill_holes,
    imreconstruct,
    imresize3_nearest,
)
from .skeleton_graph import vessel_morph


def _imgaussfilt3(x, sigma):
    """imgaussfilt3(x,sigma): isotropic 3-D Gaussian, replicate padding."""
    return ndi.gaussian_filter(
        x.astype(np.float64), sigma, mode="nearest", truncate=3.0
    )


def seg_vessel(sub, aspect, sigmas=(0.2, 0.4, 0.6, 0.8, 1.0), verbose=False):
    """Port of seg_vessel.m: contrast-normalise then run the Jerman vesselness.

    Normalisation (exact port): shift so min=0; divide by the 95th percentile of the
    voxels brighter than half the (shifted) max; clip to 1. This stretches vessel-level
    contrast into [0,1] before the Hessian filter sees it.
    """
    I = np.asarray(sub, dtype=np.float64)
    I = I - I.min()
    m = I.max()
    bright = I[I > 0.5 * m]
    if bright.size > 0:
        denom = np.percentile(bright, 95)
    else:
        denom = m if m > 0 else 1.0
    if denom == 0:
        denom = 1.0
    I = I / denom
    I[I > 1] = 1.0
    return vesselness3d(I, sigmas, aspect, tau=1.0, brightondark=True, verbose=verbose)


def segment_vessels_3d(
    vol,
    centerline,
    max_radius=8.0,
    thresh_scale=0.9,
    gaussian_sigma=0.8,
    min_component_voxels=50,
    verbose=True,
):
    """Port of segmentVessels3D.m — grow the final mask around the centerline seed.

    Otsu threshold (scaled) on a lightly smoothed volume gives a candidate bright mask;
    keep only candidate voxels within `max_radius` voxels of the centerline; then
    morphologically reconstruct (flood) outward from the centerline seed; drop small
    components. `centerline` is a logical 3-D array here (skel==1).
    """
    vol0 = np.asarray(vol, dtype=np.float64)
    sz = vol0.shape
    clVol = np.asarray(centerline, dtype=bool)
    if clVol.shape != sz:
        raise ValueError("centerline must be a logical volume the size of vol")
    if not clVol.any():
        raise ValueError("No centerline voxels found. Check centerline input.")

    volSmooth = _imgaussfilt3(vol0, gaussian_sigma) if gaussian_sigma > 0 else vol0

    vmin, vmax = float(volSmooth.min()), float(volSmooth.max())
    if vmax > vmin:
        vnorm = (volSmooth - vmin) / (vmax - vmin)
        tOtsu = graythresh(vnorm)  # Otsu on [0,1]
        T = tOtsu * (vmax - vmin) + vmin
    else:
        T = vmin
    T = T * thresh_scale
    if verbose:
        print(f"  segmentVessels3D threshold (after scaling) = {T:.4g}")

    candidateMask = volSmooth >= T

    # bwdist is in VOXEL units (no spacing) -> plain EDT; radius 2.5 is in voxels.
    D = ndi.distance_transform_edt(~clVol)
    radiusMask = D <= max_radius
    candidateMask = candidateMask & radiusMask

    marker = clVol.copy()
    if not np.all(marker <= candidateMask):
        # some seeds are below threshold: dilate seeds slightly, keep within radius
        marker = imdilate_sphere1(marker) & radiusMask

    seg = imreconstruct(marker, candidateMask)

    # remove small components (26-connectivity), matching bwconncomp(seg,26)
    seg = bwareaopen(seg, min_component_voxels, 26)
    return seg


def _combine_and_clean(BW1_1, BW1_2, roi, drop_large_blobs=True, keep_thin_large=False):
    """The skeleton-combine + hole-fill + blob-removal block shared by SegVessel.

    drop_large_blobs=True reproduces the MATLAB exactly: keep only the SMALL (<30-voxel)
    connected pieces of the *filled* map, discarding large structures.
    drop_large_blobs=False (old denser experiment): OR back post-imfill large CCs. Those
    are often SOLID GLOBS because imfill_holes fills vessel loops — rejected by visual QC.
    keep_thin_large=True (new): OR back large CCs of the UNFILLED combined skeleton BW1,
    which stay 1-voxel thin trees. Prefer this over drop_large_blobs=False.
    """
    # combine HR (BW1_1) and DYN (BW1_2) skeletons: keep HR voxels away from DYN, plus DYN
    temp = imdilate_cube(BW1_2, 6)
    BW1 = BW1_1.astype(float) * (~temp) + BW1_2.astype(float)
    BW1 = BW1 > 0
    BW1 = BW1 & (roi == 1)

    BW2 = imfill_holes(BW1)  # fill holes (can thicken loops into solid regions)
    BW3 = bwareaopen(BW2, 30, 6)  # large connected components (often filled)
    BW4 = BW2 & ~BW3  # small fragments
    BW5 = imdilate_cube(BW4, 4)  # thicken fragments so nearby ones merge
    BW6 = skeletonize(BW5.astype(bool))  # re-thin -> connected short segments
    if not drop_large_blobs:
        # OLD denser path: post-fill large CCs (glob risk) — keep for ablation only.
        BW6 = BW6 | BW3
    if keep_thin_large:
        # NEW: large CCs of the thin unfilled skeleton (no imfill thickening).
        BW6 = BW6 | bwareaopen(BW1, 30, 6)
    BW6 = BW6 & (roi == 1)
    return BW6


def SegVessel(
    sub_dyn,
    sub_hr,
    spacing,
    verbose=True,
    drop_large_blobs=True,
    dyn_edge_thresh=0.4,
    hr_edge_thresh=0.5,
    keep_thin_large=False,
):
    """Port of SegVessel.m (HR + ultrafast). Returns (vmask, morph_result).

    Parameters
    ----------
    sub_dyn : 3-D ultrafast subtraction (peak - pre), MATLAB (row,col,slice) order.
    sub_hr  : 3-D high-resolution subtraction (peak - pre), same order.
    spacing : [row_mm, col_mm, slice_mm] from the ultrafast DICOM header.
    drop_large_blobs : True = faithful MATLAB; False = keep post-fill large CCs (globs).
    dyn_edge_thresh : edge3 high thresh on DYN vesselness (MATLAB default 0.4; SegVessel_U
        uses 0.15 — main sparsity lever vs ultrafast-only).
    hr_edge_thresh : edge3 high thresh on HR vesselness (MATLAB default 0.5).
    keep_thin_large : OR large unfilled thin skeleton CCs back (density without fill-globs).
    """
    spacing = np.asarray(spacing, dtype=np.float64)
    SIZE = sub_dyn.shape
    aspectDYN = np.array([1.0, 1.0, round(spacing[2] / spacing[0], 2)])
    aspectHR = np.array(
        [1.0, 1.0, round(aspectDYN[2] * sub_dyn.shape[2] / sub_hr.shape[2], 2)]
    )
    roi = np.ones(
        SIZE, dtype=int
    )  # ROIs = ones(...) -> restriction disabled, as MATLAB

    # ---- HR branch ----
    if verbose:
        print("SegVessel: HR vesselness")
    Subfilt = sub_hr - _imgaussfilt3(sub_hr, 5)  # unsharp / high-pass
    Subfilt[Subfilt < 0] = 0
    vness_hr = seg_vessel(Subfilt, aspectHR, verbose=verbose)
    BW = edge3_approxcanny(vness_hr, hr_edge_thresh, 1)
    BW = imdilate_disk2_perslice(BW, 2)  # 2-D disk per slice
    BW1 = imresize3_nearest(BW, SIZE)  # HR grid -> DYN grid
    BW1_1 = skeletonize(BW1.astype(bool))  # Skeleton3D

    # ---- ultrafast branch ----
    if verbose:
        print(f"SegVessel: ultrafast vesselness (edge3 thresh={dyn_edge_thresh})")
    Subfilt = sub_dyn - _imgaussfilt3(sub_dyn, 5)
    Subfilt[Subfilt < 0] = 0
    vness_dyn = seg_vessel(Subfilt, aspectDYN, verbose=verbose)
    BW = edge3_approxcanny(vness_dyn, dyn_edge_thresh, 1)
    BW = imdilate_disk2_perslice(BW, 2)
    BW1_2 = skeletonize(BW.astype(bool))

    BW6 = _combine_and_clean(
        BW1_1,
        BW1_2,
        roi,
        drop_large_blobs=drop_large_blobs,
        keep_thin_large=keep_thin_large,
    )

    if verbose:
        print("SegVessel: skeleton graph cleanup (Vessel_morph)")
    morph = vessel_morph(BW6, spacing)
    morph["vness_dyn"] = vness_dyn.astype(
        np.float32
    )  # DYN Jerman map (for offline QC/filter)
    skel = morph["skel_label"] > 0

    if not skel.any():
        # Degenerate case (no centerline survived cleanup): MATLAB would error in
        # segmentVessels3D; for batch robustness we return an empty mask instead.
        if verbose:
            print("SegVessel: empty centerline after cleanup -> empty mask")
        return np.zeros(SIZE, dtype=bool), morph

    if verbose:
        print("SegVessel: growing final mask (segmentVessels3D)")
    vmask = segment_vessels_3d(sub_dyn, skel, max_radius=2.5, verbose=verbose)
    return vmask, morph


def SegVessel_U(sub_dyn, spacing, verbose=True, drop_large_blobs=True):
    """Port of SegVessel_U.m (ultrafast only). Returns (vmask, morph_result).

    Used when there is no separate high-resolution dynamic series. Note the different
    final-mask construction: dilate the skeleton and OR it with a thresholded vesselness
    region, instead of the geodesic-reconstruction grow used in SegVessel.
    drop_large_blobs : True = faithful MATLAB; False = keep connected trees.
    """
    spacing = np.asarray(spacing, dtype=np.float64)
    SIZE = sub_dyn.shape
    aspectDYN = np.array([1.0, 1.0, round(spacing[2] / spacing[0], 2)])
    roi = np.ones(SIZE, dtype=int)

    Subfilt = sub_dyn - _imgaussfilt3(sub_dyn, 5)
    Subfilt[Subfilt < 0] = 0
    skel2 = seg_vessel(Subfilt, aspectDYN, verbose=verbose)  # vesselness map (0..1)
    BW = edge3_approxcanny(skel2, 0.15, 1)  # lower thresh (0.15)
    BW = imdilate_disk2_perslice(BW, 2)
    BW1 = skeletonize(BW.astype(bool))
    BW1 = BW1 & (roi == 1)

    BW2 = imfill_holes(BW1)
    BW3 = bwareaopen(BW2, 30, 6)
    BW4 = BW2 & ~BW3
    BW5 = imdilate_cube(BW4, 4)
    BW6 = skeletonize(BW5.astype(bool))
    if not drop_large_blobs:
        BW6 = BW6 | BW3  # keep large connected vessel trees (deviation)
    BW6 = BW6 & (roi == 1)

    morph = vessel_morph(BW6, spacing)
    skel = morph["skel_label"] > 0

    # final mask: dilate skeleton (cube 2) OR (thresholded vesselness within dilated area)
    vmask = imdilate_cube(skel, 2)
    temp = skel2.copy()
    varea = imdilate_cube(skel, 4)
    temp = temp >= 0.15
    temp = bwareaopen(temp, 30, 6)
    vmask = vmask | (temp & varea)
    vmask = vmask & (roi == 1)
    return vmask, morph
