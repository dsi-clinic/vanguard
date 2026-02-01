#!/usr/bin/env python3
"""Run PyRadiomics over all patients in a split and write feature tables to disk.

Stage 1: Run PyRadiomics over all patients in a split and
write feature tables to disk.

Inputs:
- --images : CSV from radiomics_extract.py (rows = patients, cols = features)
- --masks  : CSV from radiomics_extract.py
- --labels : CSV with at least columns: patient_id,pcr[,subtype]
- --split  : CSV with at least columns: patient_id,pcr[,subtype]
- --output : output directory to write metrics, plots, and model
- --params : PyRadiomics YAML configuration
- --image-pattern  : Comma-separated template(s) for image paths relative
- --peri-radius-mm : Optional peritumoral shell width in millimeters (0 = tumor only).
- --n-proc 8       : Number of worker processes.

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
  --split  "$SPLIT" \
  --output "$OUTROOT/extract_peri5_multiphase" \
  --params "$PARAMS" \
  --image-pattern "{pid}/{pid}_0001.nii.gz,{pid}/{pid}_0002.nii.gz" \
  --mask-pattern  "{pid}.nii.gz" \
  --peri-radius-mm 5 \
  --n-proc 8

"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import SimpleITK as sitk
from joblib import Parallel, delayed
from tqdm import tqdm

try:
    from radiomics import featureextractor
except Exception:

    def featureextractor(*args, **kwargs):  # noqa: ANN201, D103
        raise err  # noqa: F821


warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.WARNING)
logging.getLogger("radiomics").setLevel(logging.ERROR)

# Tracks which non-scalar feature keys have already been logged in this process
# so the debug message fires once per unique key rather than once per patient.
# Note: joblib Parallel spawns separate processes, so each worker logs its first
# patient independently — that is expected and acceptable.
_LOGGED_NON_SCALAR_KEYS: set[str] = set()


# small helpers
def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file and ensure it contains a ``patient_id`` column."""
    csv_path = Path(path)
    data = pd.read_csv(csv_path, comment="#")
    if "patient_id" not in data.columns:
        msg = f"{csv_path} must have patient_id"
        raise ValueError(msg)
    return data


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


# radiomics extractor builder
def build_extractor(
    params_path: str | None = None,
    label_override: int | None = None,
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
    """
    if params_path:
        extractor = featureextractor.RadiomicsFeatureExtractor(paramsFile=params_path)
    else:
        extractor = featureextractor.RadiomicsFeatureExtractor()

    if label_override is not None:
        extractor.settings["label"] = label_override

    return extractor



# peritumor mask creation (cast to UInt8)
def make_peritumor_mask(mask_path: Path, radius_mm: float) -> sitk.Image | None:
    """Dilate a binary tumor mask to create a peritumor shell of given radius.

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


# flatten radiomics output
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
        "std":  np.std,
        "min":  np.min,
        "max":  np.max,
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
                    out[f"{colkey}_{idx}"] = (
                        float(elem) if _is_number(elem) else elem
                    )

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
    non_scalar_handling: str = "concat",
    aggregate_stats: list[str] | None = None,
    hybrid_concat_threshold: int = 5,
) -> dict[str, Any]:
    """Extract radiomics features for a single patient across all phases."""
    base_mask_path = path_from_pattern(masks_dir, pid, mask_pattern)
    ensure_exists(base_mask_path, "Mask")
    try:
        mask_img = sitk.ReadImage(str(base_mask_path))
    except Exception as exc:  # pragma: no cover - I/O defensive
        msg = f"Could not read mask for {pid}: {exc}"
        raise RuntimeError(msg) from exc

    flatten_kwargs: dict[str, Any] = {
        "non_scalar_handling":    non_scalar_handling,
        "aggregate_stats":        aggregate_stats,
        "hybrid_concat_threshold": hybrid_concat_threshold,
    }

    out_row: dict[str, Any] = {"patient_id": pid}

    for pat in image_patterns:
        img_path = path_from_pattern(images_dir, pid, pat)
        ensure_exists(img_path, f"Image pattern '{pat}'")
        try:
            img = sitk.ReadImage(str(img_path))
        except Exception as exc:  # pragma: no cover - I/O defensive
            msg = f"Could not read image for {pid} ({pat}): {exc}"
            raise RuntimeError(msg) from exc

        # Run on tumor mask
        res_tumor = extractor.execute(img, mask_img)
        tumor_prefix = f"{pat}_tumor_"
        out_row.update(
            flatten_radiomics_result(res_tumor, prefix=tumor_prefix, **flatten_kwargs),
        )

        # Optional peritumor shell
        if peri_radius_mm > 0:
            peri_mask = make_peritumor_mask(base_mask_path, peri_radius_mm)
            if peri_mask is not None:
                res_peri = extractor.execute(img, peri_mask)
                peri_prefix = f"{pat}_peri{int(peri_radius_mm)}mm_"
                out_row.update(
                    flatten_radiomics_result(res_peri, prefix=peri_prefix, **flatten_kwargs),
                )

    return out_row

    return out_row



# extract for split
def extract_split_features(
    pids: list[str],
    images_dir: str,
    masks_dir: str,
    image_patterns: list[str],
    mask_pattern: str,
    extractor: featureextractor.RadiomicsFeatureExtractor,
    peri_radius_mm: float = 0.0,
    n_jobs: int = 1,
    non_scalar_handling: str = "concat",
    aggregate_stats: list[str] | None = None,
    hybrid_concat_threshold: int = 5,
) -> pd.DataFrame:
    """Extract features for a list of patients and return as a DataFrame."""
    rows: list[dict[str, Any]] = []
    if n_jobs == 1:
        for pid in tqdm(pids, desc="Extracting radiomics (serial)"):
            rows.append(
                extract_for_pid(
                    pid,
                    images_dir,
                    masks_dir,
                    image_patterns,
                    mask_pattern,
                    extractor,
                    peri_radius_mm=peri_radius_mm,
                    non_scalar_handling=non_scalar_handling,
                    aggregate_stats=aggregate_stats,
                    hybrid_concat_threshold=hybrid_concat_threshold,
                ),
            )
    else:
        # Parallel execution
        func = delayed(extract_for_pid)
        rows = Parallel(n_jobs=n_jobs)(
            func(
                pid,
                images_dir,
                masks_dir,
                image_patterns,
                mask_pattern,
                extractor,
                peri_radius_mm=peri_radius_mm,
                non_scalar_handling=non_scalar_handling,
                aggregate_stats=aggregate_stats,
                hybrid_concat_threshold=hybrid_concat_threshold,
            )
            for pid in tqdm(pids, desc=f"Extracting radiomics (n_jobs={n_jobs})")
        )

    return pd.DataFrame(rows).set_index("patient_id")


# simple numeric sanitizer for extractor (so trainer finds *_final.csv)
def sanitize_numeric(data: pd.DataFrame, tag: str) -> pd.DataFrame:
    """Keep only numeric columns and drop degenerate ones, with debug logging."""
    raw_shape = data.shape
    num = data.select_dtypes(include=[np.number]).copy()
    # drop all-NaN
    all_nan = num.columns[num.isna().all()].tolist()
    num = num.drop(columns=all_nan, errors="ignore")
    # drop zero-var
    nunique = num.nunique(dropna=True)
    zero_var = nunique[nunique <= 1].index.tolist()
    num = num.drop(columns=zero_var, errors="ignore")
    print(
        f"[DEBUG] {tag}: raw={raw_shape} -> numeric={num.shape} "
        f"(all-NaN={len(all_nan)}, zero-var={len(zero_var)})",
    )
    return num


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
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for feature extraction.",
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
            "hybrid: concat short vectors, aggregate long ones.",
        ),
    )
    ap.add_argument(
        "--aggregate-stats",
        default="mean,std,min,max",
        help=(
            "Comma-separated summary stats for aggregate/hybrid mode "
            "(default: mean,std,min,max).",
        ),
    )
    ap.add_argument(
        "--hybrid-concat-threshold",
        type=int,
        default=5,
        help=(
            "For hybrid mode: vectors with length <= this value are "
            "concatenated; longer vectors are aggregated (default: 5).",
        ),
    )

    args = ap.parse_args()

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    labels = load_csv(args.labels)[["patient_id", "pcr"]].set_index("patient_id")
    splits = load_csv(args.splits).set_index("patient_id")
    # Build extractor (optionally overriding mask label)
    extractor = build_extractor(args.params, label_override=args.label_override)

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

    # Extract features
    feat_train = extract_split_features(
        train_ids,
        args.images,
        args.masks,
        image_patterns,
        args.mask_pattern,
        extractor,
        peri_radius_mm=args.peri_radius_mm,
        n_jobs=args.n_jobs,
        non_scalar_handling=args.non_scalar_handling,
        aggregate_stats=aggregate_stats,
        hybrid_concat_threshold=args.hybrid_concat_threshold,
    )
    feat_test = extract_split_features(
        test_ids,
        args.images,
        args.masks,
        image_patterns,
        args.mask_pattern,
        extractor,
        peri_radius_mm=args.peri_radius_mm,
        n_jobs=args.n_jobs,
        non_scalar_handling=args.non_scalar_handling,
        aggregate_stats=aggregate_stats,
        hybrid_concat_threshold=args.hybrid_concat_threshold,
    )

    # Sanitize numeric-only
    feat_train_final = sanitize_numeric(feat_train, tag="train")
    feat_test_final = sanitize_numeric(feat_test, tag="test")

    # Align columns (test must have same columns as train)
    feat_test_final = feat_test_final.reindex(
        columns=feat_train_final.columns,
        fill_value=np.nan,
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
