"""Jerman 3-D vesselness filter — faithful Python port.

Ports:
  Analysis Pipeline/KineticAnalysisFunctions/Jerman_Filter/vesselness3D.m
  (functions: vesselness3D, volumeEigenvalues, Hessian3D, gradient3, imgaussian)

Method provenance: part of the Python translation of Zhen Ren's lab
MATLAB vessel pipeline, which builds on Wu et al., Magn Reson Med 2019
(PMID 30368906; doi:10.1002/mrm.27529).

WHY THIS FILE MATTERS (root-cause context)
------------------------------------------
The MATLAB pipeline enhances vessels with the *Jerman* tubularity filter, NOT the
Frangi/Sato filter. They look similar (both use Hessian eigenvalues) but the response
formula is different: Jerman regularises the largest-magnitude eigenvalue with a `tau`
term so the response is much more *uniform* across vessels of different contrast.
Substituting `skimage.filters.frangi` here is the single most likely reason a previous
translation came out sparse. So we port the exact formula rather than call a library.

A note on axes: we keep the volume in MATLAB's own (dim1, dim2, dim3) order, i.e.
numpy axis 0 <-> MATLAB dim 1, axis 1 <-> dim 2, axis 2 <-> dim 3. Eigenvalues of the
Hessian are rotation-invariant, so the *names* x/y/z do not matter as long as the
Hessian is built symmetrically and consistently, which it is. `spacing` is the per-axis
voxel-size ratio (aspect) — here [1, 1, slice/inplane] — used to shrink the Gaussian
kernel along the anisotropic (slice) axis exactly as MATLAB's imgaussian does.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d


def _gradient3(F: np.ndarray, axis: int) -> np.ndarray:
    """Central differences with one-sided differences at the edges.

    This reproduces MATLAB's gradient3.m (which is the same as MATLAB's `gradient`
    but one axis at a time). numpy's np.gradient uses exactly this convention
    (edge_order=1 -> forward/backward at the ends), so we just call it per-axis.
    """
    return np.gradient(F, axis=axis, edge_order=1)


def _imgaussian(I: np.ndarray, sigma: float, spacing) -> np.ndarray:
    """Separable Gaussian smoothing, kernel shrunk per-axis by voxel spacing.

    MATLAB's imgaussian builds a 1-D kernel per dimension with standard deviation
    `sigma/spacing[d]` and half-width `ceil((sigma*6)/spacing[d]/2)`. We reproduce the
    per-axis effective sigma. MATLAB pads with 'replicate' -> scipy mode='nearest'.
    (scipy's gaussian_filter1d truncates the kernel at `truncate` sigmas; MATLAB uses a
    6*sigma full width, i.e. truncate=3. We set truncate=3.0 to match the support.)
    """
    if sigma <= 0:
        return I
    out = I
    for d in range(3):
        eff_sigma = sigma / spacing[d]
        out = gaussian_filter1d(out, eff_sigma, axis=d, mode="nearest", truncate=3.0)
    return out


def _hessian3d(volume: np.ndarray, sigma: float, spacing):
    """Gaussian-smooth then take 2nd-order central differences => Hessian components.

    Returns Dxx, Dyy, Dzz, Dxy, Dxz, Dyz (each same shape as `volume`), matching
    Hessian3D.m. 'x','y','z' below are just MATLAB's names for axes 0,1,2.
    """
    if sigma > 0:
        F = _imgaussian(volume, sigma, spacing)
    else:
        F = volume

    Dz = _gradient3(F, 2)
    Dzz = _gradient3(Dz, 2)

    Dy = _gradient3(F, 1)
    Dyy = _gradient3(Dy, 1)
    Dyz = _gradient3(Dy, 2)

    Dx = _gradient3(F, 0)
    Dxx = _gradient3(Dx, 0)
    Dxy = _gradient3(Dx, 1)
    Dxz = _gradient3(Dx, 2)

    return Dxx, Dyy, Dzz, Dxy, Dxz, Dyz


def _volume_eigenvalues(V, sigma, spacing, brightondark):
    """Per-voxel Hessian eigenvalues, magnitude-sorted |L1|<=|L2|<=|L3|.

    Ports volumeEigenvalues.m, including its speed trick: only compute eigenvalues at
    voxels that can possibly be tubular (the T-mask via B1/B2/B3), which is a big
    speedup on a 528x528x80 volume.

    IMPORTANT (a HIGH-risk mapping item): MATLAB's eig3volume MEX returns eigenvalues
    sorted by *magnitude*. numpy.linalg.eigvalsh returns them in *signed ascending*
    order, so we must re-sort each voxel's triple by absolute value. Getting this wrong
    silently breaks the whole filter.
    """
    Hxx, Hyy, Hzz, Hxy, Hxz, Hyz = _hessian3d(V, sigma, spacing)

    # Correct for scale (gamma-normalisation with gamma=2): multiply by sigma^2.
    c = sigma**2
    Hxx = c * Hxx
    Hxy = c * Hxy
    Hxz = c * Hxz
    Hyy = c * Hyy
    Hyz = c * Hyz
    Hzz = c * Hzz

    # B1,B2,B3 are the coefficients of the characteristic polynomial (trace, etc.).
    # Used only to skip voxels that cannot be tubular -> matches MATLAB exactly.
    B1 = -(Hxx + Hyy + Hzz)
    B2 = Hxx * Hyy + Hxx * Hzz + Hyy * Hzz - Hxy * Hxy - Hxz * Hxz - Hyz * Hyz
    B3 = (
        Hxx * Hyz * Hyz
        + Hxy * Hxy * Hzz
        + Hxz * Hyy * Hxz
        - Hxx * Hyy * Hzz
        - Hxy * Hyz * Hxz
        - Hxz * Hxy * Hyz
    )

    T = np.ones(B1.shape, dtype=bool)
    if brightondark:
        T[B1 <= 0] = False
        T[(B2 <= 0) & (B3 == 0)] = False
        T[(B1 > 0) & (B2 > 0) & (B1 * B2 < B3)] = False
    else:
        T[B1 >= 0] = False
        T[(B2 >= 0) & (B3 == 0)] = False
        T[(B1 < 0) & (B2 < 0) & ((-B1) * (-B2) < (-B3))] = False

    idx = np.where(T.ravel())[0]

    Lambda1 = np.zeros(T.shape).ravel()
    Lambda2 = np.zeros(T.shape).ravel()
    Lambda3 = np.zeros(T.shape).ravel()

    if idx.size > 0:
        hxx = Hxx.ravel()[idx]
        hyy = Hyy.ravel()[idx]
        hzz = Hzz.ravel()[idx]
        hxy = Hxy.ravel()[idx]
        hxz = Hxz.ravel()[idx]
        hyz = Hyz.ravel()[idx]

        # Build (N,3,3) symmetric Hessians and get eigenvalues (ascending, signed).
        n = idx.size
        H = np.empty((n, 3, 3), dtype=np.float64)
        H[:, 0, 0] = hxx
        H[:, 1, 1] = hyy
        H[:, 2, 2] = hzz
        H[:, 0, 1] = H[:, 1, 0] = hxy
        H[:, 0, 2] = H[:, 2, 0] = hxz
        H[:, 1, 2] = H[:, 2, 1] = hyz
        w = np.linalg.eigvalsh(H)  # shape (N,3), signed ascending

        # Re-sort each row by ABSOLUTE value -> |L1|<=|L2|<=|L3|.
        order = np.argsort(np.abs(w), axis=1)
        w = np.take_along_axis(w, order, axis=1)

        Lambda1[idx] = w[:, 0]
        Lambda2[idx] = w[:, 1]
        Lambda3[idx] = w[:, 2]

    Lambda1 = Lambda1.reshape(T.shape)
    Lambda2 = Lambda2.reshape(T.shape)
    Lambda3 = Lambda3.reshape(T.shape)

    for L in (Lambda1, Lambda2, Lambda3):
        L[~np.isfinite(L)] = 0
        L[np.abs(L) < 1e-4] = 0

    return Lambda1, Lambda2, Lambda3


def vesselness3d(
    I, sigmas, spacing, tau=1.0, brightondark=True, verbose=False, roi=None
):
    """Jerman multiscale vesselness. Faithful port of vesselness3D.m.

    Parameters
    ----------
    I : 3-D float array (already contrast-normalised by the caller, see seg_vessel).
    sigmas : iterable of scales (MATLAB default here: 0.2,0.4,0.6,0.8,1.0).
    spacing : per-axis voxel-size ratio (aspect), e.g. [1,1,slice/inplane].
    tau : response-uniformity control in [0.5,1]; MATLAB call uses 1.
    brightondark : vessels bright on dark background (True in the MATLAB call).
    roi : optional boolean volume restricting where the two global maxima below are
        measured. Responses are still computed everywhere; only the normalising
        statistics change. See the ROI note below for why this matters.

    Returns the max response over scales, normalised to its own max, with values
    below 1e-2 zeroed (exactly as MATLAB).

    THE ROI ARGUMENT (why a filter needs to be told where to look)
    --------------------------------------------------------------
    Two lines below take a maximum over the entire volume: the Lambda_rho floor and
    the final rescale. With tau=1 the first one sets Lambda_rho to the *same constant*
    at every voxel with positive Lambda3, so the whole output is a function of
    Lambda2 / max(Lambda3) -- every vessel in the image is scored against one voxel.
    The response formula is homogeneous of degree 0 in image intensity, so brightness
    alone cannot cause trouble; only the relative size of that one Lambda3 can.

    That makes the filter acutely sensitive to a single sharp non-vessel structure.
    Callers that hand this function a volume multiplied by a hard binary mask create
    exactly such a structure: the mask boundary is a cliff, a cliff has a very large
    second derivative, and it wins the max. Measured on four UChicago exams (see the
    2026-07-27 holistic-review LAB_NOTEBOOK entry), the Lambda3 argmax landed 1.0-2.2
    voxels inside the breast mask at every scale of every exam, and 32-66% of the
    voxels on those exams' independently-derived HR-TC4D skeletons scored exactly
    zero vesselness as a result.

    Passing an eroded mask as `roi` keeps the literal max rule but denies the boundary
    the chance to set it. Note that this is deliberately not a percentile: percentile
    floors also raise recall, but on every exam tested they did so at monotonically
    worse marginal precision, i.e. they are a global threshold loosening rather than
    the removal of a specific artifact.
    """
    I = np.asarray(I, dtype=np.float64)
    I[~np.isfinite(I)] = 0
    spacing = np.asarray(spacing, dtype=np.float64)

    if roi is not None:
        roi = np.asarray(roi, dtype=bool)
        if roi.shape != I.shape:
            raise ValueError(f"roi shape {roi.shape} != volume shape {I.shape}")
        if not roi.any():
            raise ValueError("roi is empty; cannot measure normalising statistics")

    vesselness = None
    for j, sigma in enumerate(sigmas):
        if verbose:
            print(f"  Jerman sigma = {sigma}")
        _, Lambda2, Lambda3 = _volume_eigenvalues(I, sigma, spacing, brightondark)
        if brightondark:
            Lambda2 = -Lambda2
            Lambda3 = -Lambda3

        # Regularise Lambda3 (the "proposed filter" step). Lambda_rho clamps small
        # positive Lambda3 up to tau*max(Lambda3), which is what makes the Jerman
        # response uniform (this is the term Frangi lacks).
        max_l3 = Lambda3.max() if roi is None else Lambda3[roi].max()
        Lambda_rho = Lambda3.copy()
        Lambda_rho[(Lambda3 > 0) & (Lambda3 <= tau * max_l3)] = tau * max_l3
        Lambda_rho[Lambda3 <= 0] = 0

        with np.errstate(divide="ignore", invalid="ignore"):
            response = (
                Lambda2
                * Lambda2
                * (Lambda_rho - Lambda2)
                * 27.0
                / (Lambda2 + Lambda_rho) ** 3
            )

        response[(Lambda2 >= Lambda_rho / 2.0) & (Lambda_rho > 0)] = 1.0
        response[(Lambda2 <= 0) | (Lambda_rho <= 0)] = 0.0
        response[~np.isfinite(response)] = 0.0

        vesselness = response if j == 0 else np.maximum(vesselness, response)

    m = vesselness.max() if roi is None else vesselness[roi].max()
    if m > 0:
        # Voxels outside the roi can exceed the roi's own maximum, so clip to keep
        # the documented [0,1] output range that the 1e-2 cutoff below assumes.
        vesselness = np.clip(vesselness / m, 0.0, 1.0)
    vesselness[vesselness < 1e-2] = 0.0
    return vesselness
