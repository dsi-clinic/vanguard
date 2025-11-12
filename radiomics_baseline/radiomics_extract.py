#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
radiomics_extract.py

Stage 1 of the pipeline: run PyRadiomics over all patients in a split and
write feature tables to disk.

- loads labels and train/test split
- checks that each patient has (at least) the first image phase + mask
- for each patient:
    - for each requested image phase pattern (comma-separated)
        - run PyRadiomics on tumor mask
        - optionally build a peritumor shell (dilation in mm) and run again
- flattens PyRadiomics dicts into a single row per patient
- writes:
    - features_train.csv
    - features_test.csv
    - train_labels_split.csv
    - test_labels_split.csv

You can then feed these files to radiomics_train.py to build models.
"""

import argparse, os, sys, warnings, logging
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

from joblib import Parallel, delayed
from tqdm import tqdm
import yaml

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.WARNING)
logging.getLogger("radiomics").setLevel(logging.ERROR)
np.random.seed(42)


# ---------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------
def read_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data or {}

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")
    assert "patient_id" in df.columns, f"{path} must have patient_id"
    return df

def path_from_pattern(root: str, pid: str, pattern: Optional[str]) -> Path:
    if not pattern:
        return Path(root) / f"{pid}.nii.gz"
    p = pattern.format(pid=pid)
    pth = Path(p)
    return pth if pth.is_absolute() else Path(root) / p

def file_exists(p: Path) -> bool:
    return Path(p).exists()

def ensure_exists(p: Path, what: str):
    if not file_exists(p):
        raise FileNotFoundError(f"{what} not found: {p}")


# ---------------------------------------------------------------------
# radiomics extractor builder
# ---------------------------------------------------------------------
def build_extractor(params: Dict[str, Any], label_override: Optional[int] = None):
    setting = params.get("setting") or {}
    ext = featureextractor.RadiomicsFeatureExtractor(**setting)

    if label_override is not None:
        ext.settings["label"] = int(label_override)

    # image types
    if params.get("imageType") is not None:
        ext.disableAllImageTypes()
        for name, cfg in (params["imageType"] or {}).items():
            try:
                ext.enableImageTypeByName(name, **(cfg or {}))
            except TypeError:
                if name.lower() == "log" and cfg and "sigma" in cfg:
                    ext.settings["sigma"] = cfg["sigma"]
                ext.enableImageTypeByName(name)
    else:
        ext.enableImageTypeByName("Original")

    # feature classes
    if params.get("featureClass") is not None:
        ext.disableAllFeatures()
        for fcls, feats in (params["featureClass"] or {}).items():
            ok = False
            for name_try in (fcls, fcls.lower()):
                try:
                    if feats:
                        for feat in feats:
                            ext.enableFeatureByName(name_try, feat)
                    else:
                        ext.enableFeatureClassByName(name_try)
                    ok = True
                    break
                except Exception:
                    continue
            if not ok:
                print(f"[WARN] could not enable feature class {fcls}", file=sys.stderr)
    else:
        for fcls in ("firstorder","shape","glcm","glrlm","glszm","gldm","ngtdm"):
            try: ext.enableFeatureClassByName(fcls)
            except Exception: pass

    return ext


# ---------------------------------------------------------------------
# peritumor mask creation (cast to UInt8)
# ---------------------------------------------------------------------
def make_peritumor_mask(mask_path: Path, radius_mm: float) -> Optional[sitk.Image]:
    try:
        m = sitk.ReadImage(str(mask_path))
        m_bin = sitk.Cast(m > 0, sitk.sitkUInt8)
        spacing = m.GetSpacing()
        rad_vox = [max(1, int(round(radius_mm / s))) for s in spacing]
        dil = sitk.BinaryDilate(m_bin, rad_vox)
        shell = sitk.Subtract(dil, m_bin)
        shell.CopyInformation(m)
        return shell
    except Exception as e:
        print(f"[WARN] peritumor failed for {mask_path.stem}: {e}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------
# flatten radiomics output
# ---------------------------------------------------------------------
def _is_number(x):
    try:
        float(x)
        return True
    except Exception:
        return False

def flatten_radiomics_result(res: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out = {}
    for k, v in res.items():
        colkey = f"{prefix}{k}" if prefix else k

        if v is None:
            out[colkey] = None
            continue

        if isinstance(v, np.ndarray):
            if v.ndim == 0:
                val = v.item()
                out[colkey] = float(val) if _is_number(val) else val
            elif v.ndim == 1:
                for i, vv in enumerate(v.tolist()):
                    out[f"{colkey}_{i}"] = float(vv) if _is_number(vv) else vv
            else:
                out[colkey] = len(v.ravel())
            continue

        if isinstance(v, (list, tuple)):
            for i, vv in enumerate(v):
                out[f"{colkey}_{i}"] = float(vv) if _is_number(vv) else vv
            continue

        out[colkey] = float(v) if _is_number(v) else v

    return out


# ---------------------------------------------------------------------
# feature extraction for one patient (multi-phase)
# ---------------------------------------------------------------------
def extract_for_pid(
    pid: str,
    images_dir: str,
    masks_dir: str,
    image_patterns: List[str],
    mask_pattern: str,
    extractor,
    peri_radius_mm: float = 0.0,
) -> Dict[str, Any]:
    base_mask_path = path_from_pattern(masks_dir, pid, mask_pattern)
    ensure_exists(base_mask_path, "Mask")

    row = {"patient_id": pid}

    for pi, pat in enumerate(image_patterns):
        img_path = path_from_pattern(images_dir, pid, pat)
        ensure_exists(img_path, f"Image (phase {pi+1})")

        # tumor features for this phase
        tumor_res = extractor.execute(str(img_path), str(base_mask_path))
        row.update(flatten_radiomics_result(tumor_res, prefix=f"p{pi+1}_tumor_"))

        # peritumor features for this phase
        if peri_radius_mm and peri_radius_mm > 0.0:
            peri_mask = make_peritumor_mask(base_mask_path, peri_radius_mm)
            if peri_mask is not None:
                peri_res = extractor.execute(str(img_path), peri_mask)
                row.update(flatten_radiomics_result(peri_res, prefix=f"p{pi+1}_peri_"))

    return row


# ---------------------------------------------------------------------
# extract for split
# ---------------------------------------------------------------------
def extract_split_features(
    pids: List[str],
    images_dir: str,
    masks_dir: str,
    image_patterns: List[str],
    mask_pattern: str,
    extractor,
    peri_radius_mm: float = 0.0,
    n_jobs: int = 1,
    tag: str = "train",
) -> pd.DataFrame:
    if n_jobs == 1:
        rows = [
            extract_for_pid(pid, images_dir, masks_dir, image_patterns, mask_pattern,
                            extractor, peri_radius_mm)
            for pid in tqdm(pids, desc=f"{tag.upper()} radiomics", unit="case")
        ]
    else:
        rows = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(extract_for_pid)(
                pid, images_dir, masks_dir, image_patterns, mask_pattern,
                extractor, peri_radius_mm
            )
            for pid in tqdm(pids, desc=f"{tag.upper()} radiomics", unit="case")
        )
    return pd.DataFrame.from_records(rows).set_index("patient_id")


# ---------------------------------------------------------------------
# simple numeric sanitizer for extractor (so trainer finds *_final.csv)
# ---------------------------------------------------------------------
def sanitize_numeric(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    raw = df.shape
    num = df.select_dtypes(include=[np.number]).copy()
    # drop all-NaN
    all_nan = num.columns[num.isna().all()].tolist()
    num = num.drop(columns=all_nan, errors="ignore")
    # drop zero-var
    nunique = num.nunique(dropna=True)
    zero_var = nunique[nunique <= 1].index.tolist()
    num = num.drop(columns=zero_var, errors="ignore")
    print(f"[DEBUG] {tag}: raw={raw} -> numeric={num.shape} (all-NaN={len(all_nan)}, zero-var={len(zero_var)})")
    return num


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Extract radiomics features only.")
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--params", default=None)
    ap.add_argument("--image-pattern", default="{pid}/{pid}_0001.nii.gz",
                    help="comma-separated list for multiple phases")
    ap.add_argument("--mask-pattern",  default="{pid}.nii.gz")
    ap.add_argument("--label", type=int, default=None)
    ap.add_argument("--n-proc", type=int, default=1)
    ap.add_argument("--peri-radius-mm", type=float, default=0.0,
                    help="0 = no peritumor extraction")
    args = ap.parse_args()

    outdir = Path(args.output); outdir.mkdir(parents=True, exist_ok=True)

    labels = load_csv(args.labels)[["patient_id","pcr"]].set_index("patient_id")
    split  = load_csv(args.split)[["patient_id","split"]].set_index("patient_id")
    split["split"] = split["split"].str.lower().replace({"val":"test"})
    joined = split.join(labels, how="inner")

    # turn comma-separated patterns into a list
    image_patterns = [s.strip() for s in args.image_pattern.split(",") if s.strip()]

    # filter to patients that actually have the FIRST image + mask
    train_pids, test_pids = [], []
    for pid, row in joined.iterrows():
        first_img = path_from_pattern(args.images, pid, image_patterns[0])
        msk_p     = path_from_pattern(args.masks,  pid, args.mask_pattern)
        if not (file_exists(first_img) and file_exists(msk_p)):
            continue
        if row["split"] == "train":
            train_pids.append(pid)
        elif row["split"] == "test":
            test_pids.append(pid)

    if not train_pids or not test_pids:
        sys.exit("[ERROR] After filtering for image+mask, no train or test patients remain.")

    params    = read_yaml(args.params) if args.params else {}
    extractor = build_extractor(params, label_override=args.label)

    print("[INFO] Extracting radiomics...")
    Xtr = extract_split_features(train_pids, args.images, args.masks,
                                 image_patterns, args.mask_pattern,
                                 extractor, peri_radius_mm=args.peri_radius_mm,
                                 n_jobs=args.n_proc, tag="train")
    Xtr.to_csv(outdir / "features_train.csv")

    Xte = extract_split_features(test_pids, args.images, args.masks,
                                 image_patterns, args.mask_pattern,
                                 extractor, peri_radius_mm=args.peri_radius_mm,
                                 n_jobs=args.n_proc, tag="test")
    Xte.to_csv(outdir / "features_test.csv")

    # also save labels so training script doesn't have to reload CSVs
    joined.loc[Xtr.index].to_csv(outdir / "train_labels_split.csv")
    joined.loc[Xte.index].to_csv(outdir / "test_labels_split.csv")

    # create numeric-only versions (this is what trainer will read)
    Xtr_final = sanitize_numeric(Xtr, "train")
    Xte_final = sanitize_numeric(Xte, "test")
    Xtr_final.to_csv(outdir / "features_train_final.csv")
    Xte_final.to_csv(outdir / "features_test_final.csv")

    print("[INFO] Done. Wrote features_train(.csv|_final.csv), features_test(.csv|_final.csv), and label/split CSVs.")


if __name__ == "__main__":
    main()
