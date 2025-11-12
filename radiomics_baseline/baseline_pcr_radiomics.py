#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Baseline pCR Prediction via Radiomics
- extracts tumor radiomics from 1 or more DCE phases
- can optionally make a peritumoral shell (dilated mask) and extract radiomics there too
- appends clinical subtype from patient JSONs (e.g. HER2+, HR+, triple_negative) as one-hot features
- trains RF or logistic
- saves metrics / plots / predictions
- skips patients whose image/mask are missing instead of crashing

Example (single phase, with subtype, no peri):
  python radiomics_baseline/baseline_pcr_radiomics.py \
    --images /net/projects2/vanguard/MAMA-MIA-syn60868042/images \
    --masks  /net/projects2/vanguard/MAMA-MIA-syn60868042/segmentations/expert \
    --labels radiomics_baseline/labels.csv \
    --split  radiomics_baseline/splits_train_test_ready.csv \
    --output radiomics_baseline/out_logreg_full_0001 \
    --params radiomics_baseline/pyradiomics_params.yaml \
    --image-pattern "{pid}/{pid}_0001.nii.gz" \
    --mask-pattern  "{pid}.nii.gz" \
    --classifier logistic \
    --patient-info-dir /net/projects2/vanguard/MAMA-MIA-syn60868042/patient_info_files

Example (two phases: 0001 + 0002, RF, peritumor 5mm):
  python radiomics_baseline/baseline_pcr_radiomics.py \
    --images .../images \
    --masks  .../segmentations/expert \
    --labels radiomics_baseline/labels.csv \
    --split  radiomics_baseline/splits_train_test_ready.csv \
    --output radiomics_baseline/out_rf_multi \
    --params radiomics_baseline/pyradiomics_params.yaml \
    --image-pattern "{pid}/{pid}_0001.nii.gz" \
    --extra-image-patterns "{pid}/{pid}_0002.nii.gz" \
    --mask-pattern "{pid}.nii.gz" \
    --classifier rf \
    --peri-radius-mm 5 \
    --patient-info-dir /net/projects2/vanguard/MAMA-MIA-syn60868042/patient_info_files

Run (small test):

python radiomics_baseline/baseline_pcr_radiomics.py \
  --images /net/projects2/vanguard/MAMA-MIA-syn60868042/images \
  --masks  /net/projects2/vanguard/MAMA-MIA-syn60868042/segmentations/expert \
  --labels radiomics_baseline/labels.csv \
  --split  radiomics_baseline/splits_debug_0001.csv \
  --output radiomics_baseline/out_debug \
  --params radiomics_baseline/pyradiomics_params.yaml \
  --image-pattern "{pid}/{pid}_0001.nii.gz" \
  --mask-pattern  "{pid}.nii.gz" \
  --classifier logistic \
  --n-proc 4 \
  --peri-radius-mm 5


"""

import argparse, json, os, sys, warnings, logging, re
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

from joblib import Parallel, delayed
from tqdm import tqdm

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import joblib
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
# peritumor mask creation (FIXED: cast mask to UInt8)
# ---------------------------------------------------------------------
def make_peritumor_mask(mask_path: Path, radius_mm: float) -> Optional[sitk.Image]:
    """Return a SimpleITK image of the peritumoral shell, or None on failure."""
    try:
        m = sitk.ReadImage(str(mask_path))
        # cast to binary UInt8 before dilation
        m_bin = sitk.Cast(m > 0, sitk.sitkUInt8)

        spacing = m.GetSpacing()
        # radius in voxels per dim
        rad_vox = [max(1, int(round(radius_mm / s))) for s in spacing]
        dil = sitk.BinaryDilate(m_bin, rad_vox)

        # peritumor shell = dilated - original
        shell = sitk.Subtract(dil, m_bin)
        # keep same origin / direction / spacing
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
    """
    Turn a PyRadiomics result dict into a flat dict of scalars.
    Handles:
      - plain scalars
      - 0-d numpy arrays (treat as scalar)
      - 1-d lists/arrays (expand to multiple cols)
    """
    out = {}
    for k, v in res.items():
        key = f"{prefix}{k}" if prefix else k

        # 0) none
        if v is None:
            out[key] = None
            continue

        # 1) numpy array cases
        if isinstance(v, np.ndarray):
            if v.ndim == 0:
                # scalar-in-an-array
                val = v.item()
                out[key] = float(val) if _is_number(val) else val
            elif v.ndim == 1:
                # expand
                for i, vv in enumerate(v.tolist()):
                    col = f"{key}_{i}"
                    out[col] = float(vv) if _is_number(vv) else vv
            else:
                # higher-dim → just store length
                out[key] = len(v.ravel())
            continue

        # 2) list/tuple
        if isinstance(v, (list, tuple)):
            for i, vv in enumerate(v):
                col = f"{key}_{i}"
                out[col] = float(vv) if _is_number(vv) else vv
            continue

        # 3) everything else → try to make it numeric, else keep
        out[key] = float(v) if _is_number(v) else v

    return out



# ---------------------------------------------------------------------
# feature extraction for one patient
# ---------------------------------------------------------------------
def extract_for_pid(
    pid: str,
    images_dir: str,
    masks_dir: str,
    image_pattern: str,
    mask_pattern: str,
    extractor,
    peri_radius_mm: float = 0.0,
) -> Dict[str, Any]:
    img_path = path_from_pattern(images_dir, pid, image_pattern)
    msk_path = path_from_pattern(masks_dir,  pid, mask_pattern)

    ensure_exists(img_path, "Image")
    ensure_exists(msk_path, "Mask")

    # tumor features
    tumor_res = extractor.execute(str(img_path), str(msk_path))
    row = {"patient_id": pid}
    row.update(flatten_radiomics_result(tumor_res, prefix="tumor_"))

    # peritumor features (optional)
    if peri_radius_mm and peri_radius_mm > 0.0:
        peri_mask = make_peritumor_mask(msk_path, peri_radius_mm)
        if peri_mask is not None:
            tmp_path = msk_path  # image stays same
            # we have an in-memory mask, so we need to execute on image + mask image
            peri_res = extractor.execute(str(img_path), peri_mask)
            row.update(flatten_radiomics_result(peri_res, prefix="peri_"))
        else:
            # still return row, just no peri_* features
            pass

    return row


# ---------------------------------------------------------------------
# extract for split
# ---------------------------------------------------------------------
def extract_split_features(
    pids: List[str],
    images_dir: str,
    masks_dir: str,
    image_pattern: str,
    mask_pattern: str,
    extractor,
    peri_radius_mm: float = 0.0,
    n_jobs: int = 1,
    tag: str = "train",
) -> pd.DataFrame:
    if n_jobs == 1:
        rows = [
            extract_for_pid(pid, images_dir, masks_dir, image_pattern, mask_pattern,
                            extractor, peri_radius_mm)
            for pid in tqdm(pids, desc=f"{tag.upper()} radiomics", unit="case")
        ]
    else:
        rows = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(extract_for_pid)(
                pid, images_dir, masks_dir, image_pattern, mask_pattern,
                extractor, peri_radius_mm
            )
            for pid in tqdm(pids, desc=f"{tag.upper()} radiomics", unit="case")
        )
    return pd.DataFrame.from_records(rows).set_index("patient_id")


# ---------------------------------------------------------------------
# sanitization
# ---------------------------------------------------------------------
def sanitize(df: pd.DataFrame, outdir: Path, tag: str) -> pd.DataFrame:
    raw_shape = df.shape
    # drop non-numeric but KEEP columns that look like radiomics numbers
    num = df.select_dtypes(include=[np.number]).copy()

    # drop all-NaN
    all_nan = num.columns[num.isna().all()].tolist()
    num = num.drop(columns=all_nan, errors="ignore")

    # drop zero-var
    nunique = num.nunique(dropna=True)
    zero_var = nunique[nunique <= 1].index.tolist()
    num = num.drop(columns=zero_var, errors="ignore")

    # save final
    # (you said too many files were being created; we'll only save the final one)
    num.to_csv(outdir / f"features_{tag}_final.csv")

    print(f"[DEBUG] {tag}: raw={raw_shape} -> final={num.shape} "
          f"(dropped nonscalar={raw_shape[1]-num.shape[1]}, "
          f"all-NaN={len(all_nan)}, zero-var={len(zero_var)})")

    return num


# ---------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------
def plot_roc(y_true, y_prob, outpath: Path):
    if len(np.unique(y_true)) < 2:
        return
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0,1],[0,1],"--")
    plt.legend(); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.tight_layout(); plt.savefig(outpath, dpi=200); plt.close()

def plot_pr(y_true, y_prob, outpath: Path):
    if len(np.unique(y_true)) < 2:
        return
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.tight_layout(); plt.savefig(outpath, dpi=200); plt.close()

def plot_calibration(y_true, y_prob, outpath: Path, n_bins=10):
    if len(np.unique(y_true)) < 2:
        return
    prob_true, prob_pred = calibration_curve(y_true, y_prob,
                                             n_bins=min(n_bins, len(y_true)),
                                             strategy="quantile")
    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0,1],[0,1],"--")
    plt.tight_layout(); plt.savefig(outpath, dpi=200); plt.close()


# ---------------------------------------------------------------------
# training/eval helper
# ---------------------------------------------------------------------
def train_and_eval(Xtr, Xte, ytr, yte, args, outdir: Path):
    if args.classifier == "logistic":
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale",  StandardScaler()),
            ("clf",    LogisticRegression(
                penalty="l2",
                C=args.logreg_C,
                max_iter=2000,
                class_weight="balanced",
                solver="lbfgs",
                random_state=42,
            ))
        ])
        clf_type = "logistic"
    else:
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=args.rf_n_estimators,
                max_depth=args.rf_max_depth,
                min_samples_leaf=args.rf_min_samples_leaf,
                min_samples_split=args.rf_min_samples_split,
                max_features=args.rf_max_features,
                ccp_alpha=args.rf_ccp_alpha,
                n_jobs=-1,
                class_weight="balanced",
                random_state=42,
            ))
        ])
        clf_type = "random_forest"

    pipe.fit(Xtr, ytr)
    clf_step = pipe.named_steps["clf"]
    pos_idx = int(np.where(clf_step.classes_ == 1)[0][0])

    p_tr = pipe.predict_proba(Xtr)[:, pos_idx]
    p_te = pipe.predict_proba(Xte)[:, pos_idx]

    auc_train = float(roc_auc_score(ytr, p_tr)) if len(np.unique(ytr))>1 else float("nan")
    auc_test  = float(roc_auc_score(yte, p_te)) if len(np.unique(yte))>1 else float("nan")

    fpr, tpr, thr = roc_curve(ytr, p_tr)
    thr_opt = float(thr[np.argmax(tpr - fpr)]) if len(thr) else 0.5
    ypred_te = (p_te >= thr_opt).astype(int)

    cm = confusion_matrix(yte, ypred_te, labels=[0,1])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    else:
        tn = fp = fn = tp = sens = spec = None

    # save preds
    pd.DataFrame({
        "patient_id": Xte.index,
        "y_true": yte,
        "y_prob": p_te,
        "y_pred": ypred_te,
    }).to_csv(outdir / "predictions.csv", index=False)

    # plots
    plot_roc(yte, p_te, outdir / "roc_test.png")
    plot_pr(yte, p_te, outdir / "pr_curve.png")
    try:
        plot_calibration(yte, p_te, outdir / "calibration_curve.png")
        calib_status = "ok"
    except Exception:
        calib_status = "none"

    metrics = {
        "auc_train": auc_train,
        "auc_test": auc_test,
        "n_features_used": int(Xtr.shape[1]),
        "classifier_type": clf_type,
        "class_order": clf_step.classes_.tolist(),
        "threshold_train_youdenJ": thr_opt,
        "sensitivity_test": sens,
        "specificity_test": spec,
        "tn_fp_fn_tp_test": [int(x) if x is not None else None for x in [tn, fp, fn, tp]],
        "calibration": calib_status,
    }
    if not (auc_test > 0.5):
        metrics["commentary"] = (
            "AUC_test ≤ 0.5. Possible causes: small sample, masks missing, or features dropped."
        )

    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump(pipe, outdir / "model.pkl")

    print(json.dumps(metrics, indent=2))


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--params", default=None)
    ap.add_argument("--image-pattern", default="{pid}/{pid}_0001.nii.gz")
    ap.add_argument("--mask-pattern",  default="{pid}.nii.gz")
    ap.add_argument("--label", type=int, default=None)

    ap.add_argument("--classifier", choices=["logistic","rf"], default="logistic")
    ap.add_argument("--logreg-C", type=float, default=1.0)

    ap.add_argument("--rf-n-estimators", type=int, default=300)
    ap.add_argument("--rf-max-depth", type=int, default=None)
    ap.add_argument("--rf-min-samples-leaf", type=int, default=1)
    ap.add_argument("--rf-min-samples-split", type=int, default=2)
    ap.add_argument("--rf-max-features", default="sqrt")
    ap.add_argument("--rf-ccp-alpha", type=float, default=0.0)

    ap.add_argument("--n-proc", type=int, default=1)
    ap.add_argument("--peri-radius-mm", type=float, default=0.0)
    args = ap.parse_args()

    outdir = Path(args.output); outdir.mkdir(parents=True, exist_ok=True)

    labels = load_csv(args.labels)[["patient_id","pcr"]].set_index("patient_id")
    split  = load_csv(args.split)[["patient_id","split"]].set_index("patient_id")
    split["split"] = split["split"].str.lower().replace({"val":"test"})
    joined = split.join(labels, how="inner")

    # keep only patients that have primary image + mask
    train_pids = []
    test_pids  = []
    for pid, row in joined.iterrows():
        msk_p = path_from_pattern(args.masks, pid, args.mask_pattern)
        img_p = path_from_pattern(args.images, pid, args.image_pattern)
        if not (file_exists(msk_p) and file_exists(img_p)):
            continue
        if row["split"] == "train":
            train_pids.append(pid)
        elif row["split"] == "test":
            test_pids.append(pid)

    if not train_pids or not test_pids:
        sys.exit("[ERROR] After filtering for primary image + mask, no train or no test patients remain.")

    params    = read_yaml(args.params) if args.params else {}
    extractor = build_extractor(params, label_override=args.label)

    print("[INFO] Extracting radiomics...")
    Xtr = extract_split_features(train_pids, args.images, args.masks,
                                 args.image_pattern, args.mask_pattern,
                                 extractor, peri_radius_mm=args.peri_radius_mm,
                                 n_jobs=args.n_proc, tag="train")
    Xtr.to_csv(outdir / "features_train_raw.csv")

    Xte = extract_split_features(test_pids, args.images, args.masks,
                                 args.image_pattern, args.mask_pattern,
                                 extractor, peri_radius_mm=args.peri_radius_mm,
                                 n_jobs=args.n_proc, tag="test")
    Xte.to_csv(outdir / "features_test_raw.csv")

    ytr = labels.loc[Xtr.index, "pcr"].astype(int).values
    yte = labels.loc[Xte.index, "pcr"].astype(int).values

    Xtr_s = sanitize(Xtr, outdir, tag="train")
    Xte_s = sanitize(Xte, outdir, tag="test").reindex(columns=Xtr_s.columns, fill_value=np.nan)

    print(f"[DEBUG] Xtr_s shape: {Xtr_s.shape}, Xte_s shape: {Xte_s.shape}")

    train_and_eval(Xtr_s, Xte_s, ytr, yte, args, outdir)


if __name__ == "__main__":
    main()
