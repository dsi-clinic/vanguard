"""MATLAB Image Processing Toolbox operations, ported with matched semantics.

Each function documents the MATLAB call it reproduces and any behavioural difference
we had to handle. Several of these are HIGH-risk for the "sparse segmentation" bug:
  * strel('disk',2) is a 2-D structuring element; on a 3-D volume MATLAB applies it
    per axial slice, NOT as a 3-D ball.
  * bwareaopen connectivity (6 vs 26) and the keep->=k / drop-<k convention.
  * edge3 'approxcanny' has no library equivalent (approximated here, documented).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import reconstruction as _sk_reconstruction


# ---------------------------------------------------------------------------
# Structuring elements
# ---------------------------------------------------------------------------
def strel_disk2(radius: int = 2) -> np.ndarray:
    """MATLAB strel('disk',2) neighbourhood (the default octagonal approximation).

    MATLAB's strel('disk',R) is NOT a Euclidean disk; for R=2 its .Neighborhood is
    the 5x5 octagon below. We hardcode R=2 because that is the only radius used
    (imdilate(...,strel('disk',2)) in SegVessel). This is a 2-D SE.
    """
    if radius != 2:
        raise ValueError("only strel('disk',2) is used by this pipeline")
    return np.array(
        [
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
        ],
        dtype=bool,
    )


def imdilate_disk2_perslice(bw: np.ndarray, radius: int = 2) -> np.ndarray:
    """imdilate(bw, strel('disk',2)) where bw is 3-D.

    MATLAB broadcasts a 2-D SE across the 3rd dimension => each axial slice is dilated
    independently in-plane, with NO dilation across slices. Using a 3-D ball here would
    over-dilate in z and change the skeleton. We loop over slices (axis 2 = MATLAB
    dim3 = slice) to match exactly.
    """
    se = strel_disk2(radius)
    out = np.zeros_like(bw, dtype=bool)
    for k in range(bw.shape[2]):
        out[:, :, k] = ndi.binary_dilation(bw[:, :, k].astype(bool), structure=se)
    return out


def imdilate_cube(bw: np.ndarray, n: int) -> np.ndarray:
    """imdilate(bw, strel('cube',n)) -> full n x n x n 3-D box SE.

    For even n, MATLAB centres the SE at floor((n+1)/2) (1-based). We set scipy's
    `origin` so the centre lands on the same voxel (otherwise even cubes shift by 1).
    """
    se = np.ones((n, n, n), dtype=bool)
    # scipy default centre = n//2; MATLAB centre (0-based) = ceil(n/2)-1.
    matlab_centre = int(np.ceil(n / 2)) - 1
    origin = matlab_centre - (n // 2)  # 0 for odd n, -? for even; keep centres aligned
    return ndi.binary_dilation(bw.astype(bool), structure=se, origin=origin)


def imdilate_sphere1(bw: np.ndarray) -> np.ndarray:
    """imdilate(bw, strel('sphere',1)) -> 6-connective ball (radius-1 sphere)."""
    se = ndi.generate_binary_structure(3, 1)
    return ndi.binary_dilation(bw.astype(bool), structure=se)


# ---------------------------------------------------------------------------
# Region / connectivity ops
# ---------------------------------------------------------------------------
def _conn_structure(conn: int) -> np.ndarray:
    if conn == 6:
        return ndi.generate_binary_structure(3, 1)
    if conn == 18:
        return ndi.generate_binary_structure(3, 2)
    if conn == 26:
        return ndi.generate_binary_structure(3, 3)
    raise ValueError(f"unsupported connectivity {conn}")


def bwareaopen(bw: np.ndarray, k: int, conn: int) -> np.ndarray:
    """bwareaopen(bw, k, conn): remove connected components with FEWER than k voxels.

    Keeps components with size >= k. Connectivity must match the MATLAB call site
    (SegVessel uses conn=6). scipy.ndimage.label's default is 6-conn, but we pass the
    structure explicitly to be unambiguous.
    """
    bw = bw.astype(bool)
    lab, n = ndi.label(bw, structure=_conn_structure(conn))
    if n == 0:
        return np.zeros_like(bw)
    sizes = np.bincount(lab.ravel())
    keep = sizes >= k
    keep[0] = False  # background
    return keep[lab]


def imfill_holes(bw: np.ndarray) -> np.ndarray:
    """imfill(bw,'holes') -> fill cavities fully enclosed by foreground (3-D)."""
    return ndi.binary_fill_holes(bw.astype(bool))


def imreconstruct(marker: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """imreconstruct(marker, mask): morphological reconstruction by dilation.

    Grows `marker` within `mask` until stable (geodesic dilation). skimage requires
    marker <= mask elementwise; the caller (segment_vessels_3d) guarantees this after a
    seed dilation, exactly as the MATLAB code does.
    """
    marker = marker.astype(bool)
    mask = mask.astype(bool)
    seed = marker & mask
    if not seed.any():
        return np.zeros_like(mask)
    rec = _sk_reconstruction(
        seed.astype(np.uint8), mask.astype(np.uint8), method="dilation"
    )
    return rec.astype(bool)


def graythresh(vol_norm: np.ndarray) -> float:
    """graythresh(I): Otsu threshold on data already scaled to [0,1].

    We reproduce MATLAB's 256-bin Otsu on [0,1] rather than skimage's default (which
    picks bin count from the data), so the number matches MATLAB's segmentVessels3D.
    """
    x = vol_norm.ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    hist, edges = np.histogram(x, bins=256, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist / total
    omega = np.cumsum(p)
    centres = (edges[:-1] + edges[1:]) / 2.0
    mu = np.cumsum(p * centres)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b2 = (mu_t * omega - mu) ** 2 / denom
    sigma_b2[~np.isfinite(sigma_b2)] = 0
    return float(centres[np.argmax(sigma_b2)])


def imresize3_nearest(bw: np.ndarray, out_shape) -> np.ndarray:
    """imresize3(bw, out_shape, 'nearest') with an exact output size.

    Uses the pixel-centre ('half') sampling convention so it matches MATLAB's nearest
    resize, while guaranteeing the requested shape (scipy.ndimage.zoom can round the
    size off by one, which we avoid by mapping indices directly).
    """
    bw = np.asarray(bw)
    out_shape = tuple(int(s) for s in out_shape)
    coords = []
    for d in range(3):
        in_n = bw.shape[d]
        out_n = out_shape[d]
        scale = in_n / out_n
        dst = np.arange(out_n)
        src = np.floor((dst + 0.5) * scale - 0.5 + 0.5).astype(int)  # centred nearest
        src = np.clip(src, 0, in_n - 1)
        coords.append(src)
    ii, jj, kk = np.meshgrid(coords[0], coords[1], coords[2], indexing="ij")
    return bw[ii, jj, kk]


# ---------------------------------------------------------------------------
# 3-D Canny edge (edge3 'approxcanny') — no library equivalent, approximated
# ---------------------------------------------------------------------------
def edge3_approxcanny(vol: np.ndarray, thresh: float, sigma: float) -> np.ndarray:
    """edge3(vol, 'approxcanny', thresh, sigma) -> logical edge map.

    MATLAB's 'approxcanny' is the *fast approximation* of 3-D Canny: it skips the fine
    sub-voxel non-maximum suppression and works from the Gaussian-smoothed gradient
    magnitude with hysteresis thresholding. We reproduce that:

      1. Gaussian-smooth with the given sigma (isotropic, as edge3 does).
      2. Gradient magnitude via central differences.
      3. Normalise magnitude to [0,1] by its max (MATLAB scales thresholds to the
         gradient range).
      4. Hysteresis: strong = mag > thresh; weak = mag > 0.4*thresh; keep weak voxels
         26-connected to a strong voxel. (MATLAB's default low/high ratio is 0.4.)

    This is an ACKNOWLEDGED APPROXIMATION of the toolbox routine — flagged in the Phase
    1 mapping table. Because SegVessel immediately dilates and re-skeletonises this
    edge map, the lack of exact NMS thinning has little downstream effect.
    """
    v = ndi.gaussian_filter(vol.astype(np.float64), sigma, mode="nearest", truncate=3.0)
    gx = np.gradient(v, axis=0, edge_order=1)
    gy = np.gradient(v, axis=1, edge_order=1)
    gz = np.gradient(v, axis=2, edge_order=1)
    mag = np.sqrt(gx * gx + gy * gy + gz * gz)
    m = mag.max()
    if m > 0:
        mag = mag / m

    high = thresh
    low = 0.4 * thresh
    strong = mag > high
    weak = mag > low

    # hysteresis: label weak components (26-conn), keep those containing a strong voxel
    lab, n = ndi.label(weak, structure=np.ones((3, 3, 3), dtype=bool))
    if n == 0:
        return np.zeros_like(vol, dtype=bool)
    keep_labels = np.unique(lab[strong])
    keep_labels = keep_labels[keep_labels != 0]
    out = np.isin(lab, keep_labels)
    return out
