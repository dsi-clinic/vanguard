#!/usr/bin/env python3
"""
Extract PyRadiomics features for ONE patient to quickly sanity-check
image/mask paths, YAML params, and feature extraction.

Usage example:
python radiomics_baseline/extract_one_patient.py \
  --images /net/projects2/vanguard/MAMA-MIA/images \
  --masks  /net/projects2/vanguard/MAMA-MIA/masks \
  --patient-id PATIENT001 \
  --output radiomics_baseline/outdir \
  --params radiomics_baseline/pyradiomics_params.yaml
"""
import argparse, sys, json
import os
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
import yaml


def path_from_pattern(root: str, pid: str, pattern: Optional[str]) -> Path:
    # If user passed an absolute path, use it as-is.
    if pattern and os.path.isabs(pattern):
        return Path(pattern)
    rootp = Path(root)
    return rootp / (pattern.format(pid=pid) if pattern else f"{pid}.nii.gz")


def read_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data or {}


def build_extractor(params: Optional[Dict[str, Any]]):
    if not params:
        return featureextractor.RadiomicsFeatureExtractor(
            binWidth=25, normalize=True, normalizeScale=100,
            resampledPixelSpacing=[1, 1, 1], interpolator=sitk.sitkBSpline, label=1
        )

    setting = params.get("setting") or {}
    ext = featureextractor.RadiomicsFeatureExtractor(**setting)

    # Image types
    if "imageType" in params:
        ext.disableAllImageTypes()
        for name, cfg in (params["imageType"] or {}).items():
            try:
                # Newer PyRadiomics supports kwargs here
                ext.enableImageTypeByName(name, **(cfg or {}))
            except TypeError:
                # Older PyRadiomics: set special keys in settings, then enable without kwargs
                if name.lower() == "log" and cfg and "sigma" in cfg:
                    ext.settings["sigma"] = cfg["sigma"]
                ext.enableImageTypeByName(name)

    # Feature classes
    if "featureClass" in params:
        ext.disableAllFeatures()
        for fcls, feats in (params["featureClass"] or {}).items():
            if feats:
                for feat in feats:
                    ext.enableFeatureByName(fcls, feat)
            else:
                ext.enableFeatureClassByName(fcls)

    return ext


def main():
    ap = argparse.ArgumentParser(description="Extract PyRadiomics features for one patient")
    ap.add_argument("--images", required=True, help="Directory with images (NIfTI)")
    ap.add_argument("--masks", required=True, help="Directory with masks (NIfTI, tumor label=1)")
    ap.add_argument("--patient-id", required=True, help="Patient ID (matches filenames)")
    ap.add_argument("--output", required=True, help="Output directory")
    ap.add_argument("--params", default=None, help="PyRadiomics YAML parameter file")
    ap.add_argument("--image-pattern", default=None, help='e.g. "{pid}_image.nii.gz"')
    ap.add_argument("--mask-pattern", default=None, help='e.g. "{pid}_tumorMask.nii.gz"')
    ap.add_argument("--label", type=int, default=None, help="Mask label (overrides YAML setting.label)")
    ap.add_argument("--save-name", default=None, help='Output CSV name (default: "features_<PID>.csv")')
    args = ap.parse_args()

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    pid = args.patient_id
    img_p = path_from_pattern(args.images, pid, args.image_pattern)
    msk_p = path_from_pattern(args.masks,  pid, args.mask_pattern)

    if not img_p.exists():
        sys.exit(f"[ERROR] Image not found: {img_p}")
    if not msk_p.exists():
        sys.exit(f"[ERROR] Mask not found: {msk_p}")

    # Load params & build extractor
    params = read_yaml(args.params)
    extractor = build_extractor(params)
    # Optional override label at runtime
    if args.label is not None:
        extractor.settings["label"] = args.label

    # Quick mask sanity check
    try:
        mask_img = sitk.ReadImage(str(msk_p))
        mask_arr = sitk.GetArrayFromImage(mask_img)
        label_val = int(args.label if args.label is not None else extractor.settings.get("label", 1))
        roi_voxels = int(np.sum(mask_arr == label_val))
        if roi_voxels == 0:
            print("[WARN] Mask has 0 voxels with the tumor label; features may be empty.", file=sys.stderr)
    except Exception as e:
        print(f"[WARN] Could not read mask for sanity check: {e}", file=sys.stderr)

    print(f"[INFO] Extracting features for patient {pid}")
    result = extractor.execute(str(img_p), str(msk_p))

    # Keep actual feature entries (and optionally diagnostics)
    row = {"patient_id": pid}
    for k, v in result.items():
        if k.startswith(("original", "wavelet", "log-sigma", "diagnostics")):
            row[k] = v

    df = pd.DataFrame([row]).set_index("patient_id")

    # Save CSV
    csv_name = args.save_name or f"features_{pid}.csv"
    out_csv = outdir / csv_name
    df.to_csv(out_csv)
    print(f"[INFO] Wrote {len(df.columns)} features to {out_csv}")

    # Save a tiny JSON with run metadata
    meta = {
        "patient_id": pid,
        "image_path": str(img_p),
        "mask_path": str(msk_p),
        "n_features": int(df.shape[1]),
        "label_used": int(args.label if args.label is not None else extractor.settings.get("label", 1)),
        "params_file": str(args.params) if args.params else None,
    }
    with open(outdir / f"extract_meta_{pid}.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(meta, indent=2))
    print("[INFO] Done.")


if __name__ == "__main__":
    main()

