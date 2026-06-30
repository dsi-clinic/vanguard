"""Faster, in-process inference for breast + vessel segmentation.

Drop-in-compatible replacement for the per-stage ``predict.py`` CLI. The key
change is the **vessel** stage: instead of streaming 192 sub-volume patches per
subject through the GPU at ``batch_size=1`` (``model_utils.pred_and_save_masks_3d_divided``),
we reuse the *exact same* ``Dataset3DDivided`` patch extraction but feed patches
to the model in mini-batches and scatter the results back per subject.

Mathematical equivalence
-------------------------
- The patches, their order, and their placement are produced by the unchanged
  ``Dataset3DDivided`` / ``box_indicies_dict``.
- Softmax is per-voxel and the model runs in ``eval`` mode (BatchNorm uses
  running stats), so a patch's output is independent of which batch it rides in.
- Per-voxel accumulation happens in the same patch order as the original loop.
  With ``accum_dtype=np.float16`` the batched path is therefore *bit-identical*
  to ``pred_and_save_masks_3d_divided`` (verified in ``tests/test_batching_equiv.py``).
- Production uses ``accum_dtype=np.float32`` (strictly more accurate; the final
  vessel channel is still cast to float16 on save to match the existing
  ground-truth ``.npz`` format).

AMP
---
When ``use_amp`` is set and CUDA is available, the forward pass runs under
``torch.cuda.amp.autocast``. On CPU it is a no-op so the unit tests are
unaffected. AMP only changes intermediate precision; outputs are stored as
float16 either way.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# --- Make the vendored segmentation package importable ----------------------
_SUBMODULE = (
    Path(__file__).resolve().parent.parent / "vanguard-blood-vessel-segmentation"
)
if str(_SUBMODULE) not in sys.path:
    sys.path.insert(0, str(_SUBMODULE))

# Reuse the exact helpers from the original implementation so behaviour matches.
from model_utils import (  # noqa: E402
    _center_align_prediction,
    _save_vessel_probability,
    _slice_bounds,
)


def _align_skip_connection_spatial(
    skip_connection: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    """Center-crop/pad skip tensor so spatial dims match x exactly.

    Copied from ``predict.py`` — required so odd input dimensions don't break
    the U-Net skip connections. Must be applied before any inference.
    """
    spatial_skip = skip_connection.shape[2:]
    spatial_x = x.shape[2:]

    slices = [slice(None), slice(None)]
    pad_pairs = []

    for current, target in zip(spatial_skip, spatial_x):
        delta = current - target
        if delta >= 0:
            start = delta // 2
            end = start + target
            slices.append(slice(start, end))
            pad_pairs.append((0, 0))
        else:
            slices.append(slice(None))
            deficit = -delta
            pad_before = deficit // 2
            pad_after = deficit - pad_before
            pad_pairs.append((pad_before, pad_after))

    out = skip_connection[tuple(slices)]
    if any(p != (0, 0) for p in pad_pairs):
        pad_flat = tuple(p for pair in reversed(pad_pairs) for p in pair)
        out = F.pad(out, pad_flat, mode="constant", value=0)
    return out


def apply_shape_patch() -> None:
    """Apply the odd-dimension skip-connection monkeypatch (idempotent)."""
    from unet.decoding import DecodingBlock

    if not getattr(DecodingBlock, "_vanguard_shape_patch", False):

        def _patched_center_crop(
            self: Any, skip_connection: torch.Tensor, x: torch.Tensor
        ) -> torch.Tensor:
            return _align_skip_connection_spatial(skip_connection, x)

        DecodingBlock.center_crop = _patched_center_crop
        DecodingBlock._vanguard_shape_patch = True


def build_unet(target_tissue: str) -> tuple[nn.Module, int, int]:
    """Construct the UNet3D for ``breast`` (1->1) or ``dv`` (2->3)."""
    from unet import UNet3D

    if target_tissue == "breast":
        n_channels, n_classes = 1, 1
    elif target_tissue == "dv":
        n_channels, n_classes = 2, 3
    else:
        raise ValueError("target_tissue must be 'breast' or 'dv'")

    unet = UNet3D(
        in_channels=n_channels,
        out_classes=n_classes,
        num_encoding_blocks=3,
        padding=True,
        normalization="batch",
    )
    return unet, n_channels, n_classes


def load_model(
    unet: nn.Module, saved_model_path: str, device: torch.device
) -> nn.Module:
    """Wrap in DataParallel + load weights (matches the original loaders)."""
    unet = nn.DataParallel(unet)
    unet.load_state_dict(torch.load(saved_model_path, map_location=device))
    unet.to(device)
    unet.eval()
    return unet


def _autocast(use_amp: bool, device: torch.device) -> Any:
    """Return an autocast context manager (no-op unless CUDA + use_amp)."""
    enabled = bool(use_amp) and device.type == "cuda"
    return torch.cuda.amp.autocast(enabled=enabled)


def predict_breast_batched(
    unet: nn.Module,
    dataset: Any,
    save_masks_dir: str,
    device: torch.device | None = None,
    batch_size: int = 8,
    num_workers: int = 1,
    use_amp: bool = True,
) -> None:
    """Batched equivalent of ``pred_and_save_masks_3d_simple`` (breast stage).

    The breast dataset (``Dataset3DSimple`` + ``tio.Resize((144,144,96))``) yields
    a *uniform* input shape, so subjects can be batched directly. The per-subject
    resize back to native shape and the float16 save match the original exactly.
    """
    import torchio as tio

    save_masks_dir = Path(save_masks_dir)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    global_index = 0
    for batch in loader:
        image = batch["image"].to(device, dtype=torch.float32)

        with torch.no_grad(), _autocast(use_amp, device):
            pred = unet(image)
            pred = torch.sigmoid(pred)

        pred = pred.float().cpu().numpy()
        for b in range(pred.shape[0]):
            subject_id = dataset.subject_id_list[global_index]
            x_length, y_length, z_length = dataset.image_shape_list[global_index]
            global_index += 1

            resize_transform = tio.Resize((x_length, y_length, z_length))
            out = resize_transform(pred[b])  # (1, x, y, z)
            out = np.squeeze(out).astype(np.half)
            np.save(save_masks_dir / f"{subject_id}.npy", out)


def predict_vessel_batched(
    unet: nn.Module,
    dataset: Any,
    n_classes: int,
    save_masks_dir: str,
    device: torch.device | None = None,
    batch_size: int = 16,
    num_workers: int = 1,
    use_amp: bool = True,
    accum_dtype: type = np.float32,
) -> None:
    """Batched equivalent of ``pred_and_save_masks_3d_divided``.

    Parameters mirror the original where they overlap. ``unet`` must already be
    loaded onto ``device`` and in ``eval`` mode (use :func:`load_model`).

    ``accum_dtype`` controls the per-voxel accumulator precision. Production
    default ``np.float32``; pass ``np.float16`` to reproduce the original
    bit-for-bit.
    """
    save_masks_dir = Path(save_masks_dir)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    pred_volume_numer = None
    pred_volume_denom = None
    current_list_index = None

    def _finalize(list_index: int) -> None:
        """Divide accumulators and save the vessel channel for one subject."""
        assert pred_volume_denom is not None
        assert (pred_volume_denom != 0).all(), (
            f"{dataset.subject_id_list[list_index]} has regions that weren't analyzed; configure `x_y_divisions` or "
            "`z_division`"
        )
        volume_pred_array = (pred_volume_numer / pred_volume_denom).astype(np.half)
        _save_vessel_probability(
            save_masks_dir / f"{dataset.subject_id_list[list_index]}.npz",
            volume_pred_array,
        )

    global_index = 0
    for batch in loader:
        image = batch["image"].to(device, dtype=torch.float32)

        with torch.no_grad(), _autocast(use_amp, device):
            pred = unet(image)
            if n_classes == 1:
                pred = torch.sigmoid(pred)
            else:
                pred = F.softmax(pred, dim=1)

        # Back to float32 numpy (matches original, which cast to np.half per
        # patch; we keep float32 here and cast once at save time).
        pred = pred.float().cpu().numpy()

        for b in range(pred.shape[0]):
            list_index, box_index = dataset.box_indicies_dict[global_index]
            global_index += 1
            x_index, y_index, z_index = box_index

            # Subject boundary: finalize the previous subject, start a new one.
            if list_index != current_list_index:
                if current_list_index is not None:
                    _finalize(current_list_index)
                current_list_index = list_index
                x_len, y_len, z_len = dataset.image_array_list[list_index].shape
                if n_classes == 1:
                    pred_volume_numer = np.zeros(
                        (x_len, y_len, z_len), dtype=accum_dtype
                    )
                    pred_volume_denom = np.zeros(
                        (x_len, y_len, z_len), dtype=accum_dtype
                    )
                else:
                    pred_volume_numer = np.zeros(
                        (n_classes, x_len, y_len, z_len), dtype=accum_dtype
                    )
                    pred_volume_denom = np.zeros(
                        (n_classes, x_len, y_len, z_len), dtype=accum_dtype
                    )

            x_length, y_length, z_length = dataset.image_array_list[list_index].shape
            x_start, x_stop = _slice_bounds(x_index, dataset.input_dim, x_length)
            y_start, y_stop = _slice_bounds(y_index, dataset.input_dim, y_length)
            z_start, z_stop = _slice_bounds(z_index, dataset.input_dim, z_length)
            target_shape = (x_stop - x_start, y_stop - y_start, z_stop - z_start)

            patch = np.squeeze(pred[b])
            patch = _center_align_prediction(patch, target_shape)

            if n_classes == 1:
                pred_volume_numer[x_start:x_stop, y_start:y_stop, z_start:z_stop] += (
                    patch.astype(accum_dtype, copy=False)
                )
                pred_volume_denom[x_start:x_stop, y_start:y_stop, z_start:z_stop] += 1
            else:
                pred_volume_numer[
                    :, x_start:x_stop, y_start:y_stop, z_start:z_stop
                ] += patch.astype(accum_dtype, copy=False)
                pred_volume_denom[
                    :, x_start:x_stop, y_start:y_stop, z_start:z_stop
                ] += 1

    # Finalize the last subject.
    if current_list_index is not None:
        _finalize(current_list_index)
