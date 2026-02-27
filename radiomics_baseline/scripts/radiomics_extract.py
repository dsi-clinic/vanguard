#!/usr/bin/env python3
"""Run PyRadiomics over all patients in a split and write feature tables to disk.

Stage 1: Run PyRadiomics over all patients in a split and
write feature tables to disk.

Inputs:
- --images : Directory path containing patient image files
- --masks  : Directory path containing patient mask files
- --labels : CSV with at least columns: patient_id,pcr[,subtype]
- --splits : CSV with at least columns: patient_id,split
- --output : output directory to write metrics, plots, and model
- --params : PyRadiomics YAML configuration
- --image-pattern  : Comma-separated template(s) for image paths relative
- --peri-radius-mm : Optional peritumoral shell width in millimeters (0 = tumor only).
- --peri-mode      : '3d' (isotropic, original) or '2d' (in-plane only, Braman-style).
- --force-2d       : Enable 2D texture extraction via PyRadiomics force2D.
- --force-2d-dimension : Slice dimension for force2D (0=axial, 1=coronal, 2=sagittal).
- --n-jobs         : Number of worker processes.
- --no-resume      : Ignore cached patient checkpoints and recompute.
- --clear-checkpoint : Delete output/_checkpoint before running.

What this script does:
1) Loads labels and train/test split
2) Checks that each patient has (at least) the first image phase + mask
3) For each patient:
    - For each requested image phase pattern (comma-separated)
        - Run PyRadiomics on tumor mask
        - Optionally build a peritumor shell (dilation in mm) and run again
4) Flattens PyRadiomics dicts into a single row per patient
5) Saves:
    - features_train.csv
    - features_test.csv
    - train_labels_split.csv
    - test_labels_split.csv

You can then feed these files to radiomics_train.py to build models.

Example Usage:

# 1 peri 5mm, multi-phase
python $SCRIPTS/radiomics_extract.py \
  --images "$IMAGES" \
  --masks  "$MASKS" \
  --labels "$LABELS" \
  --splits "$SPLITS" \
  --output "$OUTROOT/extract_peri5_multiphase" \
  --params "$PARAMS" \
  --image-pattern "{pid}/{pid}_0001.nii.gz,{pid}/{pid}_0002.nii.gz" \
  --mask-pattern  "{pid}.nii.gz" \
  --peri-radius-mm 5 \
  --peri-mode 2d \
  --force-2d \
  --n-proc 8

"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import shutil
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import SimpleITK as sitk
from joblib import Parallel, delayed
from tqdm import tqdm

try:
    from radiomics import featureextractor
except Exception as err:

    def featureextractor(*args, **kwargs):  # noqa: ANN201, D103
        raise err


warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.WARNING)
logging.getLogger("radiomics").setLevel(logging.ERROR)

# Tracks which non-scalar feature keys have already been logged in this process
# so the debug message fires once per unique key rather than once per patient.
# Note: joblib Parallel spawns separate processes, so each worker logs its first
# patient independently — that is expected and acceptable.
_LOGGED_NON_SCALAR_KEYS: set[str] = set()
_LOGGED_MISSING_IMAGE_PATTERNS: set[str] = set()
_CHECKPOINT_VERSION = 1
_CHECKPOINT_ROOT_NAME = "_checkpoint"
_CHECKPOINT_MANIFEST = "manifest.json"
_PHASE_TOKEN_RE = re.compile(r"_(\d{4})\.nii(?:\.gz)?")


# small helpers
def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file and ensure it contains a ``patient_id`` column."""
    csv_path = Path(path)
    data = pd.read_csv(csv_path, comment="#")
    if "patient_id" not in data.columns:
        msg = f"{csv_path} must have patient_id"
        raise ValueError(msg)
    return data


def ensure_unique_patient_ids(data: pd.DataFrame, tag: str) -> None:
    """Fail fast if a metadata table contains duplicate patient IDs."""
    dup = data["patient_id"][data["patient_id"].duplicated()].astype(str).unique()
    if len(dup) > 0:
        preview = ", ".join(dup[:5])
        msg = (
            f"{tag} has duplicate patient_id values "
            f"(n={len(dup)}): {preview}{' ...' if len(dup) > 5 else ''}"
        )
        raise ValueError(msg)


def path_from_pattern(root: str, pid: str, pattern: str | None) -> Path:
    """Resolve an image or mask path from a root directory and an optional pattern.

    If ``pattern`` is None, assume a simple ``{pid}.nii.gz`` layout under ``root``.
    Otherwise, interpret ``pattern`` as either a format string with ``{pid}``
    or a literal relative/absolute path.
    """
    if not pattern:
        return Path(root) / f"{pid}.nii.gz"
    if "{pid}" in pattern:
        rel = pattern.format(pid=pid)
    else:
        rel = pattern
    pth = Path(rel)
    if pth.is_absolute():
        return pth
    return Path(root) / rel


def file_exists(path: Path) -> bool:
    """Return True if ``path`` exists on disk."""
    return Path(path).exists()


def ensure_exists(path: Path, what: str) -> None:
    """Raise a FileNotFoundError if ``path`` does not exist."""
    if not file_exists(path):
        msg = f"{what} not found: {path}"
        raise FileNotFoundError(msg)


def _sequence_hash(values: list[str]) -> str:
    """Return a stable SHA1 for an ordered sequence of strings."""
    joined = "\n".join(values)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def _path_signature(path: str | None) -> dict[str, Any] | None:
    """Return a lightweight signature for a file or directory path."""
    if path is None:
        return None
    p = Path(path)
    sig: dict[str, Any] = {"path": str(p.resolve()), "exists": p.exists()}
    if p.exists():
        st = p.stat()
        sig["is_dir"] = p.is_dir()
        sig["size"] = st.st_size
        sig["mtime_ns"] = st.st_mtime_ns
    return sig


def build_checkpoint_fingerprint(
    *,
    images: str,
    masks: str,
    labels: str,
    splits: str,
    params: str | None,
    image_patterns: list[str],
    mask_pattern: str,
    peri_radius_mm: float,
    peri_mode: str,
    force_2d: bool,
    force_2d_dimension: int,
    allow_missing_image_patterns: bool,
    aggregate_phase_features: bool,
    phase_aggregate_stats: list[str],
    label_override: int | None,
    n_jobs: int,
    non_scalar_handling: str,
    aggregate_stats: list[str],
    hybrid_concat_threshold: int,
    train_ids: list[str],
    test_ids: list[str],
) -> tuple[str, dict[str, Any]]:
    """Build a stable fingerprint for the exact extraction setup."""
    train_ids_s = [str(pid) for pid in train_ids]
    test_ids_s = [str(pid) for pid in test_ids]
    payload: dict[str, Any] = {
        "version": _CHECKPOINT_VERSION,
        "paths": {
            "images": _path_signature(images),
            "masks": _path_signature(masks),
            "labels": _path_signature(labels),
            "splits": _path_signature(splits),
            "params": _path_signature(params),
        },
        "extract": {
            "image_patterns": image_patterns,
            "mask_pattern": mask_pattern,
            "peri_radius_mm": peri_radius_mm,
            "peri_mode": peri_mode,
            "force_2d": force_2d,
            "force_2d_dimension": force_2d_dimension,
            "allow_missing_image_patterns": allow_missing_image_patterns,
            "aggregate_phase_features": aggregate_phase_features,
            "phase_aggregate_stats": phase_aggregate_stats,
            "label_override": label_override,
            # Intentionally exclude n_jobs from cache identity.
            # Parallelism changes runtime only; extracted values should match.
            "non_scalar_handling": non_scalar_handling,
            "aggregate_stats": aggregate_stats,
            "hybrid_concat_threshold": hybrid_concat_threshold,
        },
        "splits": {
            "n_train": len(train_ids_s),
            "n_test": len(test_ids_s),
            "train_ids_sha1": _sequence_hash(train_ids_s),
            "test_ids_sha1": _sequence_hash(test_ids_s),
        },
    }

    payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha1(payload_json.encode("utf-8")).hexdigest()
    return fingerprint, payload


def prepare_checkpoint_root(outdir: Path, clear_checkpoint: bool) -> Path:
    """Create the checkpoint root directory (optionally clearing old state)."""
    checkpoint_root = outdir / _CHECKPOINT_ROOT_NAME
    if clear_checkpoint and checkpoint_root.exists():
        print(f"[CHECKPOINT] clearing {checkpoint_root}", file=sys.stderr)
        shutil.rmtree(checkpoint_root)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    return checkpoint_root


def ensure_checkpoint_manifest(
    checkpoint_root: Path,
    fingerprint: str,
    fingerprint_payload: dict[str, Any],
) -> None:
    """Ensure checkpoint cache matches the current extraction fingerprint."""
    manifest_path = checkpoint_root / _CHECKPOINT_MANIFEST
    should_write_manifest = True

    if manifest_path.exists():
        try:
            with manifest_path.open(encoding="utf-8") as fh:
                old_manifest = json.load(fh)
        except Exception:
            old_manifest = {}

        if old_manifest.get("fingerprint") != fingerprint:
            print(
                "[CHECKPOINT] fingerprint mismatch; clearing stale checkpoint cache",
                file=sys.stderr,
            )
            shutil.rmtree(checkpoint_root)
            checkpoint_root.mkdir(parents=True, exist_ok=True)
        else:
            should_write_manifest = False

    if should_write_manifest:
        manifest = {
            "version": _CHECKPOINT_VERSION,
            "fingerprint": fingerprint,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "fingerprint_payload": fingerprint_payload,
        }
        with manifest_path.open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)


def checkpoint_rows_dir(checkpoint_root: Path, split_name: str) -> Path:
    """Return the row-cache directory for one split."""
    return checkpoint_root / f"{split_name}_rows"


def _checkpoint_row_path(rows_dir: Path, pid: str) -> Path:
    """Return the per-patient checkpoint path for a split cache directory."""
    safe_pid = re.sub(r"[^\w.\-]", "_", str(pid))
    suffix = hashlib.sha1(str(pid).encode("utf-8")).hexdigest()[:10]
    return rows_dir / f"{safe_pid}__{suffix}.json"


def write_checkpoint_row(rows_dir: Path, row: dict[str, Any]) -> None:
    """Atomically write one patient feature row to checkpoint storage."""
    pid = str(row.get("patient_id", ""))
    if not pid:
        msg = "checkpoint row missing patient_id"
        raise ValueError(msg)

    rows_dir.mkdir(parents=True, exist_ok=True)
    target = _checkpoint_row_path(rows_dir, pid)
    tmp = target.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(row, fh, allow_nan=True, separators=(",", ":"))
    tmp.replace(target)


def load_checkpoint_rows(rows_dir: Path) -> dict[str, dict[str, Any]]:
    """Load all cached per-patient rows for one split."""
    rows: dict[str, dict[str, Any]] = {}
    if not rows_dir.exists():
        return rows

    for path in sorted(rows_dir.glob("*.json")):
        try:
            with path.open(encoding="utf-8") as fh:
                row = json.load(fh)
        except Exception as exc:
            print(
                f"[CHECKPOINT] skipping unreadable row cache {path}: {exc}",
                file=sys.stderr,
            )
            continue

        pid = str(row.get("patient_id", ""))
        if not pid:
            print(
                f"[CHECKPOINT] skipping row cache without patient_id: {path}",
                file=sys.stderr,
            )
            continue
        rows[pid] = row

    return rows


def clear_checkpoint_rows(rows_dir: Path) -> int:
    """Delete all cached row files for one split and return count removed."""
    if not rows_dir.exists():
        return 0
    removed = 0
    for path in rows_dir.iterdir():
        if path.is_file():
            path.unlink()
            removed += 1
    return removed


# radiomics extractor builder
def build_extractor(
    params_path: str | None = None,
    label_override: int | None = None,
    force_2d: bool = False,
    force_2d_dimension: int = 0,
) -> featureextractor.RadiomicsFeatureExtractor:
    """Build a :class:`RadiomicsFeatureExtractor` from a PyRadiomics YAML file.

    Parameters:
    params_path:
        Path to a PyRadiomics YAML parameter file.  When *None* the extractor
        is created with default settings.  The file is parsed by PyRadiomics
        itself, so ``setting``, ``imageType``, and ``featureClass`` are all
        honoured automatically.
    label_override:
        Optional integer label to force as the mask value, overriding whatever
        ``label`` is set in the YAML ``setting`` block.
    force_2d:
        If True, set ``force2D`` in extractor settings so that texture
        matrices are computed per-slice rather than in 3D.  This is often
        more stable when slice thickness >> in-plane spacing.
    force_2d_dimension:
        The dimension perpendicular to the extraction plane when
        ``force_2d`` is True.  0 = axial (z), 1 = coronal (y),
        2 = sagittal (x).  Default is 0 (axial).
    """
    if params_path:
        extractor = featureextractor.RadiomicsFeatureExtractor(paramsFile=params_path)
    else:
        extractor = featureextractor.RadiomicsFeatureExtractor()

    if label_override is not None:
        extractor.settings["label"] = label_override

    # --- Issue #1: force2D support ---
    if force_2d:
        extractor.settings["force2D"] = True
        extractor.settings["force2Ddimension"] = force_2d_dimension
        print(
            f"[INFO] force2D enabled — extraction dimension = {force_2d_dimension} "
            f"(0=axial, 1=coronal, 2=sagittal)",
            file=sys.stderr,
        )

    return extractor


# ---------------------------------------------------------------------------
# Peritumor mask creation
# ---------------------------------------------------------------------------

def make_peritumor_mask(mask_path: Path, radius_mm: float) -> sitk.Image | None:
    """Dilate a binary tumor mask to create a peritumor shell of given radius.

    Uses mean 3D spacing for isotropic dilation (original behaviour).
    Returns a SimpleITK image with dtype UInt8 containing the shell region, or
    ``None`` if the radius is non-positive or the mask cannot be read.
    """
    try:
        mask = sitk.ReadImage(str(mask_path))
    except Exception as exc:  # pragma: no cover - I/O defensive
        print(f"[WARN] could not read mask {mask_path}: {exc}", file=sys.stderr)
        return None

    spacing = mask.GetSpacing()  # (sx, sy, sz)
    mean_spacing = float(np.mean(spacing))
    r_vox = int(round(radius_mm / mean_spacing))
    if r_vox <= 0:
        return None

    dilated = sitk.BinaryDilate(
        sitk.Cast(mask, sitk.sitkUInt8),
        [r_vox] * len(spacing),
    )
    shell = sitk.Subtract(dilated, sitk.Cast(mask, sitk.sitkUInt8))
    shell = sitk.BinaryThreshold(
        shell,
        lowerThreshold=1,
        upperThreshold=255,
        insideValue=1,
        outsideValue=0,
    )
    shell.CopyInformation(mask)
    return shell


def make_peritumor_mask_2d(mask_path: Path, radius_mm: float) -> sitk.Image | None:
    """Create a peritumor shell by dilating **only in the x/y plane**.

    This implements the Braman-style 2D peritumor ring: the dilation kernel
    is computed from in-plane (x, y) spacing and the z-component is set to 0,
    so the shell never extends beyond the original tumor's slice range.

    Parameters:
    mask_path:
        Path to the binary tumor mask (NIfTI/MHA).
    radius_mm:
        Desired ring width in millimetres.

    Returns:
        SimpleITK UInt8 image of the peritumor shell, or ``None`` if the
        mask cannot be read or the requested radius is too small to produce
        any dilation voxels.
    """
    try:
        mask = sitk.ReadImage(str(mask_path))
    except Exception as exc:  # pragma: no cover - I/O defensive
        print(f"[WARN] could not read mask {mask_path}: {exc}", file=sys.stderr)
        return None

    spacing = mask.GetSpacing()  # (sx, sy, sz)
    sx, sy = spacing[0], spacing[1]

    # Compute per-axis voxel radii from in-plane spacing only
    r_vox_x = int(round(radius_mm / sx))
    r_vox_y = int(round(radius_mm / sy))

    if r_vox_x <= 0 and r_vox_y <= 0:
        print(
            f"[WARN] 2D peri dilation radius too small for spacing "
            f"({sx:.2f}, {sy:.2f}) mm — skipping",
            file=sys.stderr,
        )
        return None

    # Kernel: [x, y, z] — z is 0 so no expansion along the slice axis
    kernel_radius = [max(r_vox_x, 1), max(r_vox_y, 1), 0]

    print(
        f"[DEBUG] 2D peri ring: requested {radius_mm:.1f} mm -> "
        f"kernel [x={kernel_radius[0]}, y={kernel_radius[1]}, z=0] voxels "
        f"(spacing x={sx:.2f}, y={sy:.2f} mm)",
        file=sys.stderr,
    )

    mask_uint8 = sitk.Cast(mask, sitk.sitkUInt8)
    dilated = sitk.BinaryDilate(mask_uint8, kernelRadius=kernel_radius)
    shell = sitk.Subtract(dilated, mask_uint8)
    shell = sitk.BinaryThreshold(
        shell,
        lowerThreshold=1,
        upperThreshold=255,
        insideValue=1,
        outsideValue=0,
    )
    shell.CopyInformation(mask)
    return shell


# ---------------------------------------------------------------------------
# Flatten radiomics output
# ---------------------------------------------------------------------------

def _is_number(value: object) -> bool:
    """Return True if ``value`` can be cast to ``float`` without error."""
    try:
        float(value)
        return True
    except Exception:
        return False


def flatten_radiomics_result(
    res: dict[str, Any],
    prefix: str = "",
    non_scalar_handling: str = "concat",
    aggregate_stats: list[str] | None = None,
    hybrid_concat_threshold: int = 5,
) -> dict[str, Any]:
    """Flatten the (possibly nested) PyRadiomics result dict into a 1D mapping.

    Diagnostics keys are skipped.  Nested dicts are always expanded by sub-key.
    The treatment of list/tuple values (the 13-element angle vectors returned by
    GLCM, GLRLM, and GLDM) is controlled by *non_scalar_handling*:

    * ``"concat"``    – each element becomes its own column
      (``feature_0`` … ``feature_12``).  This is the original behaviour and the
      default.
    * ``"aggregate"`` – each vector is summarised into one column per stat in
      *aggregate_stats* (e.g. ``feature_mean``, ``feature_std``).
    * ``"hybrid"``    – vectors with length ≤ *hybrid_concat_threshold* are
      concatenated; longer vectors are aggregated.

    Parameters:
    res:
        Raw dict returned by ``RadiomicsFeatureExtractor.execute``.
    prefix:
        String prepended to every output key.
    non_scalar_handling:
        Strategy for list/tuple values (see above).
    aggregate_stats:
        Stats to compute in aggregate/hybrid mode.  Defaults to
        ``["mean", "std", "min", "max"]``.
    hybrid_concat_threshold:
        Length cutoff for hybrid mode.
    """
    if aggregate_stats is None:
        aggregate_stats = ["mean", "std", "min", "max"]

    _STAT_FUNCS: dict[str, Any] = {
        "mean": np.mean,
        "std": np.std,
        "min": np.min,
        "max": np.max,
    }

    out: dict[str, Any] = {}
    for key, value in res.items():
        if key.startswith("diagnostics_"):
            continue

        colkey = f"{prefix}{key}"

        # nested dict (named sub-features) — always expand by sub-key
        if isinstance(value, dict):
            for subkey, subval in value.items():
                full_key = f"{colkey}_{subkey}"
                out[full_key] = float(subval) if _is_number(subval) else subval
            continue

        if value is None:
            continue

        # list / tuple (angle vectors from GLCM/GLRLM/GLDM)
        if isinstance(value, list | tuple):
            # Log once per unique key so the user can see which features are
            # non-scalar without drowning in repeated lines.
            if colkey not in _LOGGED_NON_SCALAR_KEYS:
                print(
                    f"[DEBUG] non-scalar feature: {colkey} "
                    f"-> list of length {len(value)}",
                    file=sys.stderr,
                )
                _LOGGED_NON_SCALAR_KEYS.add(colkey)

            # Decide whether to aggregate this particular vector
            use_aggregate = False
            if non_scalar_handling == "aggregate":
                use_aggregate = True
            elif non_scalar_handling == "hybrid":
                use_aggregate = len(value) > hybrid_concat_threshold

            if use_aggregate:
                # Only aggregate if every element is numeric; otherwise fall
                # back to concat so we don't silently drop data.
                numeric_vals = [float(v) for v in value if _is_number(v)]
                if len(numeric_vals) == len(value):
                    arr = np.array(numeric_vals)
                    for stat_name in aggregate_stats:
                        if stat_name in _STAT_FUNCS:
                            out[f"{colkey}_{stat_name}"] = float(
                                _STAT_FUNCS[stat_name](arr),
                            )
                else:
                    # Non-numeric elements present — fall back to concat
                    for idx, elem in enumerate(value):
                        out[f"{colkey}_{idx}"] = (
                            float(elem) if _is_number(elem) else elem
                        )
            else:
                # concat: original behaviour
                for idx, elem in enumerate(value):
                    out[f"{colkey}_{idx}"] = float(elem) if _is_number(elem) else elem

        # scalar
        else:
            out[colkey] = float(value) if _is_number(value) else value

    return out


# feature extraction for one patient (multi-phase)
def extract_for_pid(
    pid: str,
    images_dir: str,
    masks_dir: str,
    image_patterns: list[str],
    mask_pattern: str,
    extractor: featureextractor.RadiomicsFeatureExtractor,
    peri_radius_mm: float = 0.0,
    peri_mode: str = "3d",
    allow_missing_image_patterns: bool = False,
    non_scalar_handling: str = "concat",
    aggregate_stats: list[str] | None = None,
    hybrid_concat_threshold: int = 5,
) -> dict[str, Any]:
    """Extract radiomics features for a single patient across all phases.

    Parameters:
    peri_mode:
        ``"3d"`` — original isotropic dilation using mean spacing.
        ``"2d"`` — in-plane (x/y) dilation only (Braman-style ring).
    """
    base_mask_path = path_from_pattern(masks_dir, pid, mask_pattern)
    ensure_exists(base_mask_path, "Mask")
    try:
        mask_img = sitk.ReadImage(str(base_mask_path))
    except Exception as exc:  # pragma: no cover - I/O defensive
        msg = f"Could not read mask for {pid}: {exc}"
        raise RuntimeError(msg) from exc

    flatten_kwargs: dict[str, Any] = {
        "non_scalar_handling": non_scalar_handling,
        "aggregate_stats": aggregate_stats,
        "hybrid_concat_threshold": hybrid_concat_threshold,
    }

    out_row: dict[str, Any] = {"patient_id": pid}

    extracted_any = False
    for pat in image_patterns:
        img_path = path_from_pattern(images_dir, pid, pat)
        if not file_exists(img_path):
            if allow_missing_image_patterns:
                if pat not in _LOGGED_MISSING_IMAGE_PATTERNS:
                    print(
                        "[WARN] optional image pattern missing for some patients; "
                        f"skipping missing files for pattern '{pat}' "
                        f"(example pid={pid})",
                        file=sys.stderr,
                    )
                    _LOGGED_MISSING_IMAGE_PATTERNS.add(pat)
                continue
            ensure_exists(img_path, f"Image pattern '{pat}'")
        try:
            img = sitk.ReadImage(str(img_path))
        except Exception as exc:  # pragma: no cover - I/O defensive
            msg = f"Could not read image for {pid} ({pat}): {exc}"
            raise RuntimeError(msg) from exc
        extracted_any = True

        # Run on tumor mask
        res_tumor = extractor.execute(img, mask_img)
        tumor_prefix = f"{pat}_tumor_"
        out_row.update(
            flatten_radiomics_result(res_tumor, prefix=tumor_prefix, **flatten_kwargs),
        )

        # Optional peritumor shell
        if peri_radius_mm > 0:
            # --- Issue #2: route to 2D or 3D peritumor builder ---
            if peri_mode == "2d":
                peri_mask = make_peritumor_mask_2d(base_mask_path, peri_radius_mm)
            else:
                peri_mask = make_peritumor_mask(base_mask_path, peri_radius_mm)

            if peri_mask is not None:
                res_peri = extractor.execute(img, peri_mask)
                peri_prefix = f"{pat}_peri{int(peri_radius_mm)}mm_"
                out_row.update(
                    flatten_radiomics_result(
                        res_peri, prefix=peri_prefix, **flatten_kwargs
                    ),
                )

    if not extracted_any:
        msg = (
            f"No image patterns were available for patient {pid}; "
            "cannot extract features."
        )
        raise RuntimeError(msg)

    return out_row


def extract_for_pid_with_checkpoint(
    pid: str,
    images_dir: str,
    masks_dir: str,
    image_patterns: list[str],
    mask_pattern: str,
    extractor: featureextractor.RadiomicsFeatureExtractor,
    peri_radius_mm: float = 0.0,
    peri_mode: str = "3d",
    allow_missing_image_patterns: bool = False,
    non_scalar_handling: str = "concat",
    aggregate_stats: list[str] | None = None,
    hybrid_concat_threshold: int = 5,
    checkpoint_rows_dir: str | None = None,
) -> dict[str, Any]:
    """Extract one patient row and persist it to checkpoint storage."""
    row = extract_for_pid(
        pid,
        images_dir,
        masks_dir,
        image_patterns,
        mask_pattern,
        extractor,
        peri_radius_mm=peri_radius_mm,
        peri_mode=peri_mode,
        allow_missing_image_patterns=allow_missing_image_patterns,
        non_scalar_handling=non_scalar_handling,
        aggregate_stats=aggregate_stats,
        hybrid_concat_threshold=hybrid_concat_threshold,
    )
    if checkpoint_rows_dir:
        write_checkpoint_row(Path(checkpoint_rows_dir), row)
    return row


# extract for split
def extract_split_features(
    pids: list[str],
    images_dir: str,
    masks_dir: str,
    image_patterns: list[str],
    mask_pattern: str,
    extractor: featureextractor.RadiomicsFeatureExtractor,
    peri_radius_mm: float = 0.0,
    peri_mode: str = "3d",
    allow_missing_image_patterns: bool = False,
    n_jobs: int = 1,
    non_scalar_handling: str = "concat",
    aggregate_stats: list[str] | None = None,
    hybrid_concat_threshold: int = 5,
    checkpoint_rows_dir: Path | None = None,
    resume: bool = True,
) -> pd.DataFrame:
    """Extract features for a split with optional per-patient checkpointing."""
    pid_list = [str(pid) for pid in pids]
    rows_by_pid: dict[str, dict[str, Any]] = {}
    pending_pids = list(pid_list)

    if checkpoint_rows_dir is not None:
        checkpoint_rows_dir.mkdir(parents=True, exist_ok=True)
        if resume:
            rows_by_pid = load_checkpoint_rows(checkpoint_rows_dir)
        else:
            removed = clear_checkpoint_rows(checkpoint_rows_dir)
            if removed:
                print(
                    f"[CHECKPOINT] cleared {removed} cached rows in {checkpoint_rows_dir}",
                    file=sys.stderr,
                )

        # Keep only rows for this split's patient IDs.
        rows_by_pid = {pid: rows_by_pid[pid] for pid in pid_list if pid in rows_by_pid}
        pending_pids = [pid for pid in pid_list if pid not in rows_by_pid]
        print(
            "[CHECKPOINT] "
            f"{checkpoint_rows_dir.name}: cached={len(rows_by_pid)} "
            f"pending={len(pending_pids)} total={len(pid_list)}",
            file=sys.stderr,
        )

    if n_jobs == 1:
        desc = "Extracting radiomics (serial)"
        if checkpoint_rows_dir is not None:
            desc = f"Extracting radiomics (serial, pending={len(pending_pids)})"
        for pid in tqdm(pending_pids, desc=desc):
            row = extract_for_pid(
                pid,
                images_dir,
                masks_dir,
                image_patterns,
                mask_pattern,
                extractor,
                peri_radius_mm=peri_radius_mm,
                peri_mode=peri_mode,
                allow_missing_image_patterns=allow_missing_image_patterns,
                non_scalar_handling=non_scalar_handling,
                aggregate_stats=aggregate_stats,
                hybrid_concat_threshold=hybrid_concat_threshold,
            )
            if checkpoint_rows_dir is not None:
                write_checkpoint_row(checkpoint_rows_dir, row)
            rows_by_pid[str(row["patient_id"])] = row
    else:
        # Parallel execution. Each finished patient is checkpointed inside the
        # worker so progress survives interruptions/timeouts.
        checkpoint_dir_str = str(checkpoint_rows_dir) if checkpoint_rows_dir else None
        func = delayed(extract_for_pid_with_checkpoint)
        new_rows = Parallel(n_jobs=n_jobs)(
            func(
                pid,
                images_dir,
                masks_dir,
                image_patterns,
                mask_pattern,
                extractor,
                peri_radius_mm=peri_radius_mm,
                peri_mode=peri_mode,
                allow_missing_image_patterns=allow_missing_image_patterns,
                non_scalar_handling=non_scalar_handling,
                aggregate_stats=aggregate_stats,
                hybrid_concat_threshold=hybrid_concat_threshold,
                checkpoint_rows_dir=checkpoint_dir_str,
            )
            for pid in tqdm(
                pending_pids,
                desc=f"Extracting radiomics (n_jobs={n_jobs}, pending={len(pending_pids)})",
            )
        )
        for row in new_rows:
            rows_by_pid[str(row["patient_id"])] = row

    missing = [pid for pid in pid_list if pid not in rows_by_pid]
    if missing:
        preview = ", ".join(missing[:5])
        msg = (
            "Extraction finished with missing patient rows: "
            f"{preview}{' ...' if len(missing) > 5 else ''}"
        )
        raise RuntimeError(msg)

    ordered_rows = [rows_by_pid[pid] for pid in pid_list]
    if not ordered_rows:
        return pd.DataFrame(index=pd.Index([], name="patient_id"))
    return pd.DataFrame(ordered_rows).set_index("patient_id")


# simple numeric sanitizer for extractor (so trainer finds *_final.csv)
def sanitize_numeric(data: pd.DataFrame, tag: str) -> pd.DataFrame:
    """Build a train-time numeric feature matrix (train-only schema selection).

    All columns are coerced to numeric (non-numeric values -> NaN), then
    all-NaN and zero-variance columns are dropped. The resulting column set is
    the reference schema that should be applied to test data.
    """
    raw_shape = data.shape
    coerced = data.apply(pd.to_numeric, errors="coerce")
    num = coerced.copy()
    # drop all-NaN (train-only)
    all_nan = num.columns[num.isna().all()].tolist()
    num = num.drop(columns=all_nan, errors="ignore")
    # drop zero-var (train-only)
    nunique = num.nunique(dropna=True)
    zero_var = nunique[nunique <= 1].index.tolist()
    num = num.drop(columns=zero_var, errors="ignore")
    print(
        f"[DEBUG] {tag}: raw={raw_shape} -> numeric={num.shape} "
        f"(all-NaN={len(all_nan)}, zero-var={len(zero_var)})",
    )
    return num


def align_numeric_to_reference(
    data: pd.DataFrame,
    reference_columns: list[str],
    tag: str,
) -> pd.DataFrame:
    """Coerce to numeric and align to a train-derived feature schema.

    This avoids dropping informative columns based only on the test split's
    variance. Missing reference columns are filled with NaN and imputed later.
    """
    raw_shape = data.shape
    coerced = data.apply(pd.to_numeric, errors="coerce")
    ref = list(reference_columns)
    extra_cols = [c for c in coerced.columns if c not in ref]
    missing_cols = [c for c in ref if c not in coerced.columns]
    aligned = coerced.reindex(columns=ref, fill_value=np.nan)
    all_nan_after_align = int(aligned.isna().all(axis=0).sum())
    print(
        f"[DEBUG] {tag}: raw={raw_shape} -> aligned={aligned.shape} "
        f"(missing={len(missing_cols)}, extra_ignored={len(extra_cols)}, "
        f"all-NaN-after-align={all_nan_after_align})",
    )
    return aligned


def aggregate_multiphase_features(
    df: pd.DataFrame,
    *,
    stats: list[str],
    drop_original_phase_columns: bool = True,
) -> pd.DataFrame:
    """Aggregate raw phase-specific feature columns into phase-blind summaries.

    Phase-specific columns are detected via a ``_0001.nii.gz``-style token in
    the column name and grouped by replacing the phase token with
    ``_PHASE.nii.gz``. For each group, row-wise statistics are computed across
    available phases (NaNs are ignored).
    """
    valid_stats = [s.strip().lower() for s in stats if s.strip()]
    supported = {"mean", "std", "min", "max"}
    invalid = [s for s in valid_stats if s not in supported]
    if invalid:
        msg = f"Unsupported phase aggregate stats: {invalid}"
        raise ValueError(msg)
    if not valid_stats:
        msg = "--phase-aggregate-stats must include at least one stat"
        raise ValueError(msg)

    groups: dict[str, list[str]] = {}
    for col in df.columns:
        m = _PHASE_TOKEN_RE.search(str(col))
        if not m:
            continue
        phase_idx = m.group(1)
        if phase_idx == "0000":
            continue
        key = _PHASE_TOKEN_RE.sub("_PHASE.nii.gz", str(col), count=1)
        groups.setdefault(key, []).append(str(col))

    if not groups:
        print("[DEBUG] no phase-specific columns found for aggregation")
        return df

    out = df.copy()
    total_phase_cols = 0
    for base_key, cols in groups.items():
        if len(cols) < 2:
            # Skip single-phase groups; no count-related confound to remove.
            continue
        total_phase_cols += len(cols)
        block = out[cols].apply(pd.to_numeric, errors="coerce")
        for stat in valid_stats:
            if stat == "mean":
                agg = block.mean(axis=1, skipna=True)
            elif stat == "std":
                agg = block.std(axis=1, skipna=True, ddof=0)
            elif stat == "min":
                agg = block.min(axis=1, skipna=True)
            else:  # max
                agg = block.max(axis=1, skipna=True)
            out[f"{base_key}__phaseagg_{stat}"] = agg

    if drop_original_phase_columns:
        drop_cols = sorted(
            {
                c
                for cols in groups.values()
                if len(cols) >= 2
                for c in cols
            }
        )
        out = out.drop(columns=drop_cols, errors="ignore")
        print(
            "[DEBUG] phase aggregation: "
            f"groups={len(groups)} dropped_phase_cols={len(drop_cols)} "
            f"added_cols={len(out.columns) - len(df.columns) + len(drop_cols)}",
        )
    else:
        print(
            "[DEBUG] phase aggregation (kept original phase cols): "
            f"groups={len(groups)} phase_cols={total_phase_cols}",
        )

    return out


# main
def main() -> None:
    """Entry point: run extraction for train and test splits and write CSVs."""
    ap = argparse.ArgumentParser(description="Extract radiomics features only.")
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--params",
        default=None,
        help="Optional YAML with PyRadiomics params.",
    )
    ap.add_argument(
        "--image-pattern",
        default="{pid}/{pid}_0001.nii.gz",
        help="Python format string or relative path for image files.",
    )
    ap.add_argument(
        "--allow-missing-image-patterns",
        action="store_true",
        help=(
            "If set, missing image files for any requested pattern are skipped "
            "per-patient instead of failing extraction. At least one pattern "
            "must still be available for each patient."
        ),
    )
    ap.add_argument(
        "--aggregate-phase-features",
        action="store_true",
        help=(
            "Aggregate raw phase-specific features across available phases "
            "(e.g. 0001..0005) into phase-blind summary columns, then drop the "
            "original per-phase columns."
        ),
    )
    ap.add_argument(
        "--phase-aggregate-stats",
        default="mean",
        help=(
            "Comma-separated stats for --aggregate-phase-features. "
            "Supported: mean,std,min,max. Default: mean."
        ),
    )
    ap.add_argument(
        "--mask-pattern",
        default="{pid}/{pid}_mask.nii.gz",
        help="Python format string or relative path for mask files.",
    )
    ap.add_argument(
        "--peri-radius-mm",
        type=float,
        default=0.0,
        help="If > 0, build a peritumor shell of this radius (in mm).",
    )
    ap.add_argument(
        "--peri-mode",
        choices=["3d", "2d"],
        default="3d",
        help=(
            "Peritumor dilation strategy. "
            "'3d': isotropic dilation using mean spacing (original). "
            "'2d': in-plane (x/y) dilation only — Braman-style ring, "
            "preserves z slice extent."
        ),
    )
    ap.add_argument(
        "--force-2d",
        action="store_true",
        default=False,
        help=(
            "Enable PyRadiomics force2D mode: texture matrices are computed "
            "per-slice instead of in 3D. Useful when slice thickness >> "
            "in-plane spacing."
        ),
    )
    ap.add_argument(
        "--force-2d-dimension",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help=(
            "Dimension perpendicular to the extraction plane when --force-2d "
            "is set. 0 = axial (z), 1 = coronal (y), 2 = sagittal (x). "
            "Default: 0."
        ),
    )
    ap.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for feature extraction.",
    )
    ap.add_argument(
        "--no-resume",
        action="store_true",
        help=(
            "Disable checkpoint resume. When set, cached per-patient rows are "
            "ignored and recomputed."
        ),
    )
    ap.add_argument(
        "--clear-checkpoint",
        action="store_true",
        help="Delete existing checkpoint cache under output/_checkpoint before running.",
    )
    ap.add_argument(
        "--label-override",
        type=int,
        default=None,
        help="Optional mask label value overriding any YAML label.",
    )
    ap.add_argument(
        "--non-scalar-handling",
        choices=["concat", "aggregate", "hybrid"],
        default="concat",
        help=(
            "How to handle non-scalar (vector) features from PyRadiomics. "
            "concat: each element becomes its own column (original behaviour). "
            "aggregate: summarise each vector with aggregate_stats. "
            "hybrid: concat short vectors, aggregate long ones."
        ),
    )
    ap.add_argument(
        "--aggregate-stats",
        default="mean,std,min,max",
        help=(
            "Comma-separated summary stats for aggregate/hybrid mode "
            "(default: mean,std,min,max)."
        ),
    )
    ap.add_argument(
        "--hybrid-concat-threshold",
        type=int,
        default=5,
        help=(
            "For hybrid mode: vectors with length <= this value are "
            "concatenated; longer vectors are aggregated (default: 5)."
        ),
    )

    args = ap.parse_args()

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    labels_df = load_csv(args.labels)[["patient_id", "pcr"]].copy()
    splits_df = load_csv(args.splits).copy()
    ensure_unique_patient_ids(labels_df, "labels")
    ensure_unique_patient_ids(splits_df, "splits")
    labels_df["patient_id"] = labels_df["patient_id"].astype(str)
    splits_df["patient_id"] = splits_df["patient_id"].astype(str)
    labels = labels_df.set_index("patient_id")
    splits = splits_df.set_index("patient_id")
    # Build extractor (optionally overriding mask label and force2D)
    extractor = build_extractor(
        args.params,
        label_override=args.label_override,
        force_2d=args.force_2d,
        force_2d_dimension=args.force_2d_dimension,
    )

    # Identify train/test patient IDs
    if "split" not in splits.columns:
        msg = f"{args.splits} must contain a 'split' column"
        raise ValueError(msg)
    train_ids = splits.index[splits["split"] == "train"].tolist()
    test_ids = splits.index[splits["split"] == "test"].tolist()

    # Parse image patterns (comma-separated)
    image_patterns = [
        pat.strip() for pat in args.image_pattern.split(",") if pat.strip()
    ]

    # Parse aggregate stats from comma-separated string
    aggregate_stats = [s.strip() for s in args.aggregate_stats.split(",") if s.strip()]
    phase_aggregate_stats = [
        s.strip() for s in args.phase_aggregate_stats.split(",") if s.strip()
    ]

    # Prepare per-patient checkpointing
    checkpoint_root = prepare_checkpoint_root(
        outdir,
        clear_checkpoint=args.clear_checkpoint,
    )
    fingerprint, fingerprint_payload = build_checkpoint_fingerprint(
        images=args.images,
        masks=args.masks,
        labels=args.labels,
        splits=args.splits,
        params=args.params,
        image_patterns=image_patterns,
        mask_pattern=args.mask_pattern,
        peri_radius_mm=args.peri_radius_mm,
        peri_mode=args.peri_mode,
        force_2d=args.force_2d,
        force_2d_dimension=args.force_2d_dimension,
        allow_missing_image_patterns=args.allow_missing_image_patterns,
        aggregate_phase_features=args.aggregate_phase_features,
        phase_aggregate_stats=phase_aggregate_stats,
        label_override=args.label_override,
        n_jobs=args.n_jobs,
        non_scalar_handling=args.non_scalar_handling,
        aggregate_stats=aggregate_stats,
        hybrid_concat_threshold=args.hybrid_concat_threshold,
        train_ids=[str(pid) for pid in train_ids],
        test_ids=[str(pid) for pid in test_ids],
    )
    ensure_checkpoint_manifest(checkpoint_root, fingerprint, fingerprint_payload)
    print(
        f"[CHECKPOINT] root={checkpoint_root} resume={not args.no_resume}",
        file=sys.stderr,
    )

    # Extract features
    feat_train = extract_split_features(
        train_ids,
        args.images,
        args.masks,
        image_patterns,
        args.mask_pattern,
        extractor,
        peri_radius_mm=args.peri_radius_mm,
        peri_mode=args.peri_mode,
        allow_missing_image_patterns=args.allow_missing_image_patterns,
        n_jobs=args.n_jobs,
        non_scalar_handling=args.non_scalar_handling,
        aggregate_stats=aggregate_stats,
        hybrid_concat_threshold=args.hybrid_concat_threshold,
        checkpoint_rows_dir=checkpoint_rows_dir(checkpoint_root, "train"),
        resume=not args.no_resume,
    )
    feat_test = extract_split_features(
        test_ids,
        args.images,
        args.masks,
        image_patterns,
        args.mask_pattern,
        extractor,
        peri_radius_mm=args.peri_radius_mm,
        peri_mode=args.peri_mode,
        allow_missing_image_patterns=args.allow_missing_image_patterns,
        n_jobs=args.n_jobs,
        non_scalar_handling=args.non_scalar_handling,
        aggregate_stats=aggregate_stats,
        hybrid_concat_threshold=args.hybrid_concat_threshold,
        checkpoint_rows_dir=checkpoint_rows_dir(checkpoint_root, "test"),
        resume=not args.no_resume,
    )

    if args.aggregate_phase_features:
        feat_train = aggregate_multiphase_features(
            feat_train,
            stats=phase_aggregate_stats,
            drop_original_phase_columns=True,
        )
        feat_test = aggregate_multiphase_features(
            feat_test,
            stats=phase_aggregate_stats,
            drop_original_phase_columns=True,
        )

    # Sanitize numeric-only
    feat_train_final = sanitize_numeric(feat_train, tag="train")
    # Apply the train-derived schema to test; do not perform test-only
    # zero-variance/all-NaN dropping.
    feat_test_final = align_numeric_to_reference(
        feat_test,
        feat_train_final.columns.tolist(),
        tag="test",
    )

    # Save outputs
    feat_train.to_csv(outdir / "features_train_raw.csv")
    feat_test.to_csv(outdir / "features_test_raw.csv")
    feat_train_final.to_csv(outdir / "features_train_final.csv")
    feat_test_final.to_csv(outdir / "features_test_final.csv")

    # Also save labels so training script doesn't have to reload CSVs
    labels.loc[train_ids].to_csv(outdir / "train_labels_split.csv")
    labels.loc[test_ids].to_csv(outdir / "test_labels_split.csv")


if __name__ == "__main__":
    main()
