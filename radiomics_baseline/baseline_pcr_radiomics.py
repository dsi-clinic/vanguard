#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Baseline pCR Prediction via Radiomics (PyRadiomics + RF/Logistic)
Two-pass: FULL image and BBOX-CROPPED image (from patient JSON)
===============================================================================

Example
-------
python radiomics_baseline/baseline_pcr_radiomics.py \
  --images /net/projects2/vanguard/MAMA-MIA-syn60868042/images \
  --masks  /net/projects2/vanguard/MAMA-MIA-syn60868042/segmentations/expert \
  --labels radiomics_baseline/labels.csv \
  --split  radiomics_baseline/splits_train_test_ready.csv \
  --output radiomics_baseline/out_rf_full_0001_bbox \
  --params radiomics_baseline/pyradiomics_params.yaml \
  --image-pattern "{pid}/{pid}_0001.nii.gz" \
  --mask-pattern  "{pid}.nii.gz" \
  --patient-info-dir /net/projects2/vanguard/MAMA-MIA-syn60868042/patient_info_files \
  --classifier rf \
  --rf-n-estimators 500 --rf-max-depth 12 --rf-min-samples-leaf 5 \
  --rf-max-features sqrt --rf-ccp-alpha 0.0 \
  --calibrate none \
  --n-proc 8 --bbox-pad 2 \
  --max-features 0 --max-corr 0.0

Outputs
-------
(outdir)/full/:    features_*.csv, features_*_final.csv, model.pkl, metrics.json,
                   predictions.csv, roc_test.png, pr_curve.png, calibration_curve.png,
                   example_full.png
(outdir)/crop/:    same set + example_crop.png

Notes
-----
- Regularization:
  * Logistic: --logreg-C, --logreg-penalty {l2,l1,elasticnet}, --logreg-l1-ratio.
  * RF: --rf-max-depth, --rf-min-samples-leaf, --rf-min-samples-split,
        --rf-max-features, --rf-ccp-alpha (cost-complexity pruning).
- “GLCM is symmetrical...” messages are normal info from PyRadiomics.
"""

import argparse, json, os, sys, warnings, logging, re
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt
import joblib
import yaml
from tqdm import tqdm
from joblib import Parallel, delayed

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.WARNING)
logging.getLogger("radiomics").setLevel(logging.ERROR)
np.random.seed(42)

# ------------------------- IO helpers -------------------------

def _json_default(o):
    # handle numpy / pandas types cleanly
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (pd.Series, pd.Index)):
        return o.tolist()
    return str(o)

def read_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path: return {}
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data or {}

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")
    assert "patient_id" in df.columns, f"CSV must have 'patient_id' column: {path}"
    return df

def path_from_pattern(root: str, pid: str, pattern: Optional[str]) -> Path:
    if not pattern:
        return Path(root) / f"{pid}.nii.gz"
    pat = pattern.format(pid=pid)
    p = Path(pat)
    return p if p.is_absolute() else Path(root) / pat

def ensure_exists(path: Path, what: str):
    if not Path(path).exists():
        raise FileNotFoundError(f"{what} not found: {path}")

# ------------------------- bbox helpers -------------------------

def clamp(v, lo, hi): return max(lo, min(int(v), hi))

def load_bbox_from_json(pid: str, info_dir: Optional[str]) -> Optional[Dict[str,int]]:
    if not info_dir: return None
    f = Path(info_dir) / f"{pid}.json"
    if not f.exists(): return None
    try:
        data = json.loads(Path(f).read_text())
        bc = data.get("primary_lesion", {}).get("breast_coordinates", None)
        if not bc: return None
        keys = ("x_min","x_max","y_min","y_max","z_min","z_max")
        if not all(k in bc for k in keys): return None
        return {k:int(bc[k]) for k in keys}
    except Exception:
        return None

def bbox_from_mask(mask_img: sitk.Image) -> Optional[Dict[str,int]]:
    arr = sitk.GetArrayFromImage(mask_img)  # z,y,x
    idx = np.argwhere(arr > 0)
    if idx.size == 0: return None
    z_min, y_min, x_min = idx.min(axis=0)
    z_max, y_max, x_max = idx.max(axis=0)
    return {
        "x_min": int(x_min), "x_max": int(x_max),
        "y_min": int(y_min), "y_max": int(y_max),
        "z_min": int(z_min), "z_max": int(z_max),
    }

def pad_and_clip_bbox(bbox: Dict[str,int], size_xyz: Tuple[int,int,int], pad: int) -> Dict[str,int]:
    sx, sy, sz = size_xyz  # (X, Y, Z)
    x0 = clamp(bbox["x_min"] - pad, 0, sx-1)
    x1 = clamp(bbox["x_max"] + pad, 0, sx-1)
    y0 = clamp(bbox["y_min"] - pad, 0, sy-1)
    y1 = clamp(bbox["y_max"] + pad, 0, sy-1)
    z0 = clamp(bbox["z_min"] - pad, 0, sz-1)
    z1 = clamp(bbox["z_max"] + pad, 0, sz-1)
    return {"x_min": x0, "x_max": x1, "y_min": y0, "y_max": y1, "z_min": z0, "z_max": z1}

def crop_to_bbox(img: sitk.Image, mask: sitk.Image, bbox: Dict[str,int]) -> Tuple[sitk.Image, sitk.Image]:
    # SimpleITK index order: (x, y, z)
    start = [int(bbox["x_min"]), int(bbox["y_min"]), int(bbox["z_min"])]
    size  = [int(bbox["x_max"] - bbox["x_min"] + 1),
             int(bbox["y_max"] - bbox["y_min"] + 1),
             int(bbox["z_max"] - bbox["z_min"] + 1)]
    roi_img  = sitk.RegionOfInterest(img,  size=size, index=start)
    roi_mask = sitk.RegionOfInterest(mask, size=size, index=start)
    # Keep metadata (spacing/origin/direction already preserved)
    return roi_img, roi_mask

# ------------------------- visualization -------------------------

def _best_slice_index(mask_img: Optional[sitk.Image], img: sitk.Image) -> int:
    z = img.GetSize()[2]
    if mask_img is None:
        return z // 2
    arr = sitk.GetArrayFromImage(mask_img)  # z,y,x
    if np.any(arr > 0):
        areas = (arr > 0).sum(axis=(1,2))
        return int(np.argmax(areas))
    return z // 2

def save_example_slice(img: sitk.Image, mask: Optional[sitk.Image], out_png: Path, title: str):
    arr = sitk.GetArrayFromImage(img)  # z,y,x
    z_idx = _best_slice_index(mask, img)
    sl = np.asarray(arr[z_idx], dtype=float)
    sl = (sl - np.nanmin(sl)) / (np.nanmax(sl) - np.nanmin(sl) + 1e-8)

    plt.figure(figsize=(5,5))
    plt.imshow(sl, cmap="gray", interpolation="nearest")
    if mask is not None:
        marr = sitk.GetArrayFromImage(mask)
        msl = (marr[z_idx] > 0).astype(float)
        # overlay boundary
        from scipy import ndimage as ndi
        edges = np.logical_xor(msl, ndi.binary_erosion(msl, iterations=1))
        plt.imshow(np.ma.masked_where(edges == 0, edges), alpha=0.7)
    plt.title(title); plt.axis("off"); plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200); plt.close()

# ------------------------- PyRadiomics extractor -------------------------

def build_extractor(params: Dict[str, Any], label_override: Optional[int] = None):
    setting = params.get("setting") or {}
    ext = featureextractor.RadiomicsFeatureExtractor(**setting)
    if label_override is not None:
        ext.settings["label"] = int(label_override)

    if params.get("imageType") is not None:
        ext.disableAllImageTypes()
        for name, cfg in (params.get("imageType") or {}).items():
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
        for fcls, feats in (params.get("featureClass") or {}).items():
            ok = False
            for nm in (fcls, fcls.lower()):
                try:
                    if feats:
                        for feat in feats: ext.enableFeatureByName(nm, feat)
                    else:
                        ext.enableFeatureClassByName(nm)
                    ok = True; break
                except Exception:
                    continue
            if not ok:
                print(f"[WARN] Could not enable feature class '{fcls}'", file=sys.stderr)
    else:
        for fcls in ("firstorder","shape","glcm","glrlm","glszm","gldm","ngtdm"):
            try: ext.enableFeatureClassByName(fcls)
            except Exception: pass
    return ext

# ------------------------- extraction -------------------------

def extract_one(extractor, img_obj: sitk.Image, msk_obj: sitk.Image, pid: str) -> Dict[str, Any]:
    try:
        res = extractor.execute(img_obj, msk_obj)
        out = {"patient_id": pid}
        for k, v in res.items():
            # keep original/wavelet/log/diagnostics keys
            if k.startswith(("original", "wavelet", "log", "diagnostics")):
                # drop non-scalar arrays; keep scalars/strings
                if isinstance(v, (list, tuple, np.ndarray)) and np.asarray(v).ndim > 0:
                    continue
                out[k] = v
        return out
    except Exception as e:
        print(f"[WARN] Feature extraction failed for {pid}: {e}", file=sys.stderr)
        return {"patient_id": pid}

def extract_split_features(
    pids: List[str],
    images_dir: str,
    masks_dir: str,
    image_pat: Optional[str],
    mask_pat: Optional[str],
    extractor,
    n_jobs: int,
    tag: str,
    variant: str,
    example_png: Optional[Path] = None,
    use_bbox: bool = False,
    bbox_pad: int = 0,
    info_dir: Optional[str] = None,
) -> pd.DataFrame:
    first_saved = [False]  # mutable flag

    def _worker(pid: str) -> Dict[str, Any]:
        img_p = path_from_pattern(images_dir, pid, image_pat)
        msk_p = path_from_pattern(masks_dir,  pid, mask_pat)
        ensure_exists(img_p, "Image")
        ensure_exists(msk_p, "Mask")

        img = sitk.ReadImage(str(img_p))
        msk = sitk.ReadImage(str(msk_p))

        if use_bbox:
            bbox = load_bbox_from_json(pid, info_dir) or bbox_from_mask(msk)
            if bbox is not None:
                sx, sy, sz = img.GetSize()
                bbox = pad_and_clip_bbox(bbox, (sx,sy,sz), bbox_pad)
                img2, msk2 = crop_to_bbox(img, msk, bbox)
            else:
                img2, msk2 = img, msk
        else:
            img2, msk2 = img, msk

        # Save one example slice for this variant
        if (example_png is not None) and (not first_saved[0]):
            title = f"{variant.upper()} example ({pid})"
            try:
                save_example_slice(img2, msk2, example_png, title)
                first_saved[0] = True
            except Exception as e:
                print(f"[WARN] Could not save example slice for {pid}: {e}", file=sys.stderr)

        return extract_one(extractor, img2, msk2, pid)

    if n_jobs == 1:
        rows = [ _worker(pid) for pid in tqdm(pids, desc=f"{tag.upper()} {variant}", unit="case") ]
    else:
        rows = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_worker)(pid) for pid in tqdm(pids, desc=f"{tag.upper()} {variant}", unit="case")
        )
    return pd.DataFrame.from_records(rows).set_index("patient_id")

# ------------------------- sanitization -------------------------

def family_of(col: str) -> str:
    for f in ("shape","firstorder","glcm","glrlm","glszm","gldm","ngtdm"):
        if re.search(fr"(^|_)({f})(_|$)", col, flags=re.I):
            return f
    return "other"

def sanitize_and_prune(df: pd.DataFrame, outdir: Path, tag: str,
                       max_corr: float = 0.0, max_features: int = 0) -> pd.DataFrame:
    raw_shape = df.shape
    # keep only 'original_' scalar features; drop non-scalars
    cols = []
    for c in df.columns:
        if not str(c).startswith("original_"): continue
        v0 = df[c].iloc[0]
        if isinstance(v0, (list, tuple, np.ndarray)):
            continue
        cols.append(c)
    rad = df[cols].copy()

    # coerce objects that are numeric-ish
    for c in rad.columns:
        if rad[c].dtype == "object":
            s = pd.to_numeric(rad[c], errors="coerce")
            if s.notna().any():
                rad[c] = s

    # drop all-NaN & zero-variance
    allnan_cols = rad.columns[rad.isna().all()].tolist()
    rad = rad.drop(columns=allnan_cols, errors="ignore")
    nunique = rad.nunique(dropna=True)
    zerovar_cols = nunique[nunique <= 1].index.tolist()
    rad = rad.drop(columns=zerovar_cols, errors="ignore")

    # debug artifacts
    fam_series = pd.Series({c: family_of(c) for c in rad.columns})
    fam_series.to_csv(outdir / f"families_{tag}.csv")
    fam_counts = fam_series.value_counts().sort_index()
    fam_counts.to_csv(outdir / f"family_counts_{tag}.csv")
    (outdir / f"features_{tag}_final.csv").write_text(rad.to_csv())
    pd.DataFrame({"dropped_allnan": allnan_cols}).to_csv(outdir / f"dropped_allnan_{tag}.csv", index=False)
    pd.DataFrame({"dropped_zerovar": zerovar_cols}).to_csv(outdir / f"dropped_zerovar_{tag}.csv", index=False)

    print(f"[DEBUG] {tag}: raw={raw_shape} -> final={rad.shape} "
          f"(drop all-NaN={len(allnan_cols)}, zero-var={len(zerovar_cols)})")

    # correlation pruning
    if max_corr and max_corr > 0.0 and rad.shape[1] > 1:
        corr = rad.corr(numeric_only=True).abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if any(upper[c] >= max_corr)]
        rad = rad.drop(columns=to_drop, errors="ignore")
        (outdir / f"features_{tag}_after_corr.csv").write_text(rad.to_csv())
        print(f"[DEBUG] {tag}: corr-pruned at |r|>={max_corr} -> {rad.shape} (dropped {len(to_drop)})")
    else:
        print(f"[DEBUG] {tag}: corr-prune disabled")

    # top-K variance (TRAIN only)
    if (tag == "train") and max_features and max_features > 0 and rad.shape[1] > max_features:
        var = rad.var(axis=0, skipna=True).sort_values(ascending=False)
        keep_cols = var.index[:max_features]
        rad = rad.loc[:, keep_cols]
        print(f"[DEBUG] {tag}: top-K variance kept {max_features} of {var.size}")
    else:
        print(f"[DEBUG] {tag}: top-K disabled (keeping {rad.shape[1]} features)")

    return rad

# ------------------------- plots -------------------------

def plot_roc(y_true, y_prob, outpath: Path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(); plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (test)"); plt.legend(); plt.tight_layout()
    plt.savefig(outpath, dpi=200); plt.close()

def plot_pr(y_true, y_prob, outpath: Path):
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(); plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision–Recall Curve (test)")
    plt.tight_layout(); plt.savefig(outpath, dpi=200); plt.close()

def plot_calibration(y_true, y_prob, outpath: Path, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='quantile')
    plt.figure(); plt.plot(prob_pred, prob_true, marker='o'); plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency")
    plt.title("Calibration Curve (test)")
    plt.tight_layout(); plt.savefig(outpath, dpi=200); plt.close()

# ------------------------- train/eval routine -------------------------

def train_and_eval(Xtr_s: pd.DataFrame, Xte_s: pd.DataFrame, ytr, yte,
                   args, outdir: Path, variant_name: str) -> Dict[str, Any]:
    # Model with explicit regularization controls
    if args.classifier == "logistic":
        penalty = args.logreg_penalty
        solver = "lbfgs" if penalty == "l2" else ("liblinear" if penalty == "l1" else "saga")
        clf = LogisticRegression(
            penalty=penalty,
            C=args.logreg_C,
            l1_ratio=(args.logreg_l1_ratio if penalty == "elasticnet" else None),
            solver=solver, max_iter=4000, class_weight="balanced", random_state=42
        )
        steps = [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("clf", clf)]
        classifier_type = f"logistic_{penalty}"
    else:
        clf = RandomForestClassifier(
            n_estimators=args.rf_n_estimators,
            max_depth=args.rf_max_depth,
            min_samples_leaf=args.rf_min_samples_leaf,
            min_samples_split=args.rf_min_samples_split,
            max_features=args.rf_max_features,
            ccp_alpha=args.rf_ccp_alpha,
            n_jobs=-1, class_weight="balanced", random_state=42, oob_score=False
        )
        steps = [("impute", SimpleImputer(strategy="median")), ("clf", clf)]
        classifier_type = "random_forest"

    pipe = Pipeline(steps)

    # Optional calibration
    if args.calibrate in ("platt","isotonic"):
        method = "sigmoid" if args.calibrate == "platt" else "isotonic"
        pipe = CalibratedClassifierCV(base_estimator=pipe, method=method, cv=5)

    pipe.fit(Xtr_s, ytr)

    # get proba for class 1
    def _proba(pipe, X):
        # CalibratedClassifierCV wraps the pipeline
        pr = pipe.predict_proba(X)
        # find class index=1
        classes_ = getattr(pipe, "classes_", None)
        if classes_ is None:
            # try base estimator
            classes_ = getattr(pipe.base_estimator.named_steps["clf"], "classes_", [0,1])
        pos_idx = int(np.where(np.array(classes_) == 1)[0][0])
        return pr[:, pos_idx], classes_

    p_tr, classes_ = _proba(pipe, Xtr_s)
    p_te, _         = _proba(pipe, Xte_s)

    auc_train = float(roc_auc_score(ytr, p_tr)) if len(np.unique(ytr))>1 else float("nan")
    auc_test  = float(roc_auc_score(yte, p_te)) if len(np.unique(yte))>1 else float("nan")

    fpr, tpr, thr = roc_curve(ytr, p_tr)
    thr_opt = float(thr[np.argmax(tpr - fpr)]) if len(thr) else 0.5
    ypred_te = (p_te >= thr_opt).astype(int)

    cm = confusion_matrix(yte, ypred_te, labels=[0,1])
    tn, fp, fn, tp = (cm.ravel() if cm.size == 4 else (np.nan, np.nan, np.nan, np.nan))
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")

    # Save artifacts
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"patient_id": Xte_s.index, "y_true": yte, "y_prob": p_te, "y_pred": ypred_te}) \
      .to_csv(outdir / "predictions.csv", index=False)
    plot_roc(yte, p_te, outdir / "roc_test.png")
    plot_pr(yte, p_te, outdir / "pr_curve.png")
    calib_status = "ok"
    try:
        plot_calibration(yte, p_te, outdir / "calibration_curve.png")
    except Exception:
        calib_status = "none"

    metrics = {
        "variant": variant_name,
        "auc_train": auc_train,
        "auc_test": auc_test,
        "n_features_used": int(Xtr_s.shape[1]),
        "classifier_type": classifier_type,
        "class_order": list(classes_),
        "threshold_train_youdenJ": thr_opt,
        "sensitivity_test": sens,
        "specificity_test": spec,
        "tn_fp_fn_tp_test": [int(x) if not (isinstance(x, float) and np.isnan(x)) else None for x in [tn, fp, fn, tp]],
        "calibration": calib_status
    }
    if not (auc_test > 0.5):
        metrics["commentary"] = (
            "AUC_test ≤ 0.5. Try correlation pruning (--max-corr), top-K variance (--max-features), "
            "stronger RF regularization (--rf-max-depth, --rf-min-samples-leaf, --rf-max-features), "
            "or different DCE phases / image types."
        )

    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=_json_default)

    # Save model (note: CalibratedClassifierCV not joblib-serializable across versions sometimes; usually ok)
    joblib.dump(pipe, outdir / "model.pkl")
    print(json.dumps(metrics, indent=2, default=_json_default))
    return metrics

# ------------------------- main -------------------------

def main():
    ap = argparse.ArgumentParser(description="Baseline pCR prediction from radiomics (full + bbox-crop)")
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--params", default=None)
    ap.add_argument("--image-pattern", default=None, help='e.g. "{pid}/{pid}_0001.nii.gz"')
    ap.add_argument("--mask-pattern",  default=None, help='e.g. "{pid}.nii.gz"')
    ap.add_argument("--label", type=int, default=None)

    # model selection
    ap.add_argument("--classifier", choices=["logistic","rf"], default="rf")

    # logistic regularization
    ap.add_argument("--logreg-penalty", choices=["l2","l1","elasticnet"], default="l2")
    ap.add_argument("--logreg-C", type=float, default=1.0)
    ap.add_argument("--logreg-l1-ratio", type=float, default=0.5)

    # RF regularization
    ap.add_argument("--rf-n-estimators", type=int, default=500)
    ap.add_argument("--rf-max-depth", type=int, default=12)
    ap.add_argument("--rf-min-samples-leaf", type=int, default=5)
    ap.add_argument("--rf-min-samples-split", type=int, default=2)
    ap.add_argument("--rf-max-features", default="sqrt")  # int | float | "sqrt" | "log2"
    ap.add_argument("--rf-ccp-alpha", type=float, default=0.0)

    # calibration
    ap.add_argument("--calibrate", choices=["none","platt","isotonic"], default="none")

    # radiomics & runtime
    ap.add_argument("--n-proc", type=int, default=1)
    ap.add_argument("--max-features", type=int, default=0)
    ap.add_argument("--max-corr", type=float, default=0.0)

    # bbox / json
    ap.add_argument("--patient-info-dir", default=None, help="Dir with {pid}.json files")
    ap.add_argument("--bbox-pad", type=int, default=0)
    ap.add_argument("--skip-crop", action="store_true", help="Only run FULL, skip BBOX-CROP")

    args = ap.parse_args()

    root_out = Path(args.output); root_out.mkdir(parents=True, exist_ok=True)
    out_full = root_out / "full"
    out_crop = root_out / "crop"

    labels = load_csv(args.labels)[["patient_id","pcr"]].set_index("patient_id")
    split  = load_csv(args.split)[["patient_id","split"]].set_index("patient_id")
    split["split"] = split["split"].str.lower().replace({"val":"test"})
    joined = split.join(labels, how="inner")

    train_pids = joined.index[joined["split"]=="train"].tolist()
    test_pids  = joined.index[joined["split"]=="test"].tolist()
    if not train_pids or not test_pids:
        sys.exit("[ERROR] Need at least one train and one test patient.")

    params    = read_yaml(args.params) if args.params else {}
    extractor = build_extractor(params, label_override=args.label)

    # ---------- FULL variant ----------
    print("[INFO] Extracting radiomics (FULL)...")
    Xtr_full = extract_split_features(
        train_pids, args.images, args.masks, args.image_pattern, args.mask_pattern,
        extractor, n_jobs=args.n_proc, tag="train", variant="full",
        example_png=out_full / "example_full.png", use_bbox=False,
        bbox_pad=0, info_dir=args.patient_info_dir
    )
    Xtr_full.to_csv(out_full / "features_train.csv")
    Xte_full = extract_split_features(
        test_pids, args.images, args.masks, args.image_pattern, args.mask_pattern,
        extractor, n_jobs=args.n_proc, tag="test", variant="full",
        example_png=out_full / "example_full.png", use_bbox=False,
        bbox_pad=0, info_dir=args.patient_info_dir
    )
    Xte_full.to_csv(out_full / "features_test.csv")

    ytr = labels.loc[Xtr_full.index, "pcr"].astype(int).values
    yte = labels.loc[Xte_full.index, "pcr"].astype(int).values

    Xtr_full_s = sanitize_and_prune(Xtr_full, out_full, tag="train",
                                    max_corr=args.max_corr, max_features=args.max_features)
    Xte_full_s_tmp = sanitize_and_prune(Xte_full, out_full, tag="test",
                                        max_corr=args.max_corr, max_features=0)
    Xte_full_s = Xte_full_s_tmp.reindex(columns=Xtr_full_s.columns, fill_value=np.nan)
    print(f"[DEBUG] FULL: Xtr_s={Xtr_full_s.shape}, Xte_s={Xte_full_s.shape}")

    _ = train_and_eval(Xtr_full_s, Xte_full_s, ytr, yte, args, out_full, "full")

    # ---------- CROP variant ----------
    if not args.skip_crop:
        print("[INFO] Extracting radiomics (BBOX-CROP)...")
        Xtr_crop = extract_split_features(
            train_pids, args.images, args.masks, args.image_pattern, args.mask_pattern,
            extractor, n_jobs=args.n_proc, tag="train", variant="crop",
            example_png=out_crop / "example_crop.png", use_bbox=True,
            bbox_pad=args.bbox_pad, info_dir=args.patient_info_dir
        )
        Xtr_crop.to_csv(out_crop / "features_train.csv")
        Xte_crop = extract_split_features(
            test_pids, args.images, args.masks, args.image_pattern, args.mask_pattern,
            extractor, n_jobs=args.n_proc, tag="test", variant="crop",
            example_png=out_crop / "example_crop.png", use_bbox=True,
            bbox_pad=args.bbox_pad, info_dir=args.patient_info_dir
        )
        Xte_crop.to_csv(out_crop / "features_test.csv")

        ytr_c = labels.loc[Xtr_crop.index, "pcr"].astype(int).values
        yte_c = labels.loc[Xte_crop.index, "pcr"].astype(int).values

        Xtr_crop_s = sanitize_and_prune(Xtr_crop, out_crop, tag="train",
                                        max_corr=args.max_corr, max_features=args.max_features)
        Xte_crop_s_tmp = sanitize_and_prune(Xte_crop, out_crop, tag="test",
                                            max_corr=args.max_corr, max_features=0)
        Xte_crop_s = Xte_crop_s_tmp.reindex(columns=Xtr_crop_s.columns, fill_value=np.nan)
        print(f"[DEBUG] CROP: Xtr_s={Xtr_crop_s.shape}, Xte_s={Xte_crop_s.shape}")

        _ = train_and_eval(Xtr_crop_s, Xte_crop_s, ytr_c, yte_c, args, out_crop, "crop")

if __name__ == "__main__":
    main()