#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Baseline pCR Prediction via Radiomics (PyRadiomics + RandomForest/Logistic)
- Default image phase: 0001 (override via --image-pattern if needed)
- tqdm progress bars for TRAIN/TEST extraction
- Robust feature enabling (YAML + safe fallback) that avoids shape2D unless force2D
- Safe sanitization that drops non-scalar columns and excludes shape2D keys
===============================================================================

Example
-------
python radiomics_baseline/baseline_pcr_radiomics.py \
  --images /net/projects2/vanguard/MAMA-MIA-syn60868042/images \
  --masks  /net/projects2/vanguard/MAMA-MIA-syn60868042/segmentations/expert \
  --labels radiomics_baseline/labels.csv \
  --split  radiomics_baseline/splits_train_test_ready.csv \
  --output radiomics_baseline/out_rf_full_0001_newest \
  --params radiomics_baseline/pyradiomics_params.yaml \
  --classifier rf --rf-n-estimators 500 --n-proc 8 \
  --max-features 0 --max-corr 0.0
"""

import argparse, json, os, sys, warnings, logging, re
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import joblib
import yaml
from tqdm import tqdm
from joblib import Parallel, delayed

# Silence noisy logs
warnings.filterwarnings("ignore", category=UserWarning)
for name in ("radiomics", "radiomics.glcm", "radiomics.glcm2D", "radiomics.shape2D"):
    logging.getLogger(name).setLevel(logging.ERROR)
logging.basicConfig(level=logging.WARNING)

np.random.seed(42)

# ------------------------- Helpers -------------------------

def read_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path: return {}
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data or {}

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")
    assert "patient_id" in df.columns, f"CSV must have 'patient_id' column: {path}"
    return df

def image_path(root: str, pid: str, pattern: Optional[str]) -> Path:
    """Default image pattern: {pid}/{pid}_0001.nii.gz (hard-coded)."""
    if pattern and os.path.isabs(pattern):
        return Path(pattern)
    pat = pattern if pattern else "{pid}/{pid}_0001.nii.gz"
    return Path(root) / pat.format(pid=pid)

def mask_path(root: str, pid: str, pattern: Optional[str]) -> Path:
    """Default mask pattern: {pid}.nii.gz (no phase)."""
    if pattern and os.path.isabs(pattern):
        return Path(pattern)
    pat = pattern if pattern else "{pid}.nii.gz"
    return Path(root) / pat.format(pid=pid)

def ensure_exists(path: Path, what: str):
    if not Path(path).exists():
        raise FileNotFoundError(f"{what} not found: {path}")

# ------------------------- PyRadiomics extractor -------------------------

def _summarize_enabled(ext) -> Dict[str, Any]:
    try:
        ef = ext.enabledFeatures
    except Exception:
        ef = {}
    counts = {k: (len(v) if isinstance(v, (list, tuple)) else 0) for k, v in ef.items()}
    return {
        "enabled_image_types": getattr(ext, "enabledImagetypes", None),
        "enabled_features_by_class": counts,
        "total_enabled_features": int(sum(counts.values())) if counts else 0,
    }

_ALLOWED_CLASSES = ("firstorder", "shape", "glcm", "glrlm", "glszm", "gldm", "ngtdm")

def build_extractor(params: Dict[str, Any], label_override: Optional[int] = None, outdir: Optional[Path] = None):
    setting = params.get("setting") or {}
    ext = featureextractor.RadiomicsFeatureExtractor(**setting)
    if label_override is not None:
        ext.settings["label"] = int(label_override)

    # image types
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

    # feature classes (avoid shape2D unless force2D)
    requested = (params.get("featureClass") or {}) if (params.get("featureClass") is not None) else None
    force2D = bool(setting.get("force2D", False))

    if requested is not None:
        # If shape2D requested but not force2D, drop it
        if "shape2D" in requested and not force2D:
            print("[WARN] YAML requests shape2D but setting.force2D is not True; skipping shape2D.", file=sys.stderr)
            requested = {k: v for k, v in requested.items() if k.lower() != "shape2d"}

        ext.disableAllFeatures()
        for fcls, feats in requested.items():
            enabled = False
            for alias in (fcls, fcls.lower()):
                try:
                    if feats:
                        for feat in feats:
                            ext.enableFeatureByName(alias, feat)
                    else:
                        ext.enableFeatureClassByName(alias)
                    enabled = True
                    break
                except Exception:
                    continue
            if not enabled:
                print(f"[WARN] Could not enable feature class '{fcls}'", file=sys.stderr)

        info = _summarize_enabled(ext)
        if outdir:
            (outdir / "enabled_features.json").write_text(json.dumps(info, indent=2))
        total_after = info.get("total_enabled_features", 0)

        # SAFE fallback: enable only allowed classes (no shape2D)
        if total_after < 20:
            print(f"[WARN] Only {total_after} features enabled from YAML; enabling standard classes as fallback.")
            ext.disableAllFeatures()
            for fcls in _ALLOWED_CLASSES:
                try: ext.enableFeatureClassByName(fcls)
                except Exception: pass
            if outdir:
                (outdir / "enabled_features_fallback.json").write_text(
                    json.dumps(_summarize_enabled(ext), indent=2)
                )

    else:
        # No featureClass in YAML: enable standard classes (no shape2D)
        ext.disableAllFeatures()
        for fcls in _ALLOWED_CLASSES:
            try: ext.enableFeatureClassByName(fcls)
            except Exception: pass
        if outdir:
            (outdir / "enabled_features.json").write_text(json.dumps(_summarize_enabled(ext), indent=2))

    return ext

# ------------------------- extraction -------------------------

def extract_features_for_case(extractor, image_path: Path, mask_path: Path, pid: str) -> Dict[str, Any]:
    """Run PyRadiomics and coerce all 'original_*' features to scalars.
       Any array-like values are summarized with np.nanmean so we don't lose them later.
    """
    def to_scalar(v):
        if v is None:
            return np.nan
        # Already numeric scalar
        if isinstance(v, (float, int, np.floating, np.integer)):
            return float(v)
        # Array-like -> summarize
        if isinstance(v, (list, tuple, np.ndarray)):
            try:
                arr = np.array(v, dtype=float)
                if arr.size == 0:
                    return np.nan
                return float(np.nanmean(arr))
            except Exception:
                return np.nan
        # String -> try numeric
        if isinstance(v, str):
            try:
                return float(v)
            except Exception:
                return np.nan
        # Anything else
        return np.nan

    try:
        res = extractor.execute(str(image_path), str(mask_path))
        out = {"patient_id": pid}

        # Keep only radiomics from 'original_' family; explicitly skip any shape2D keys
        for k, v in res.items():
            if not k.startswith("original_"):
                continue
            if "shape2D_" in k or k.startswith("original_shape2D_"):
                # Avoid array/2D outputs unless you explicitly enable force2D in YAML
                continue
            out[k] = to_scalar(v)

        # (Optional) You can keep diagnostics in a separate file if you want:
        # diag = {k: res[k] for k in res if k.startswith("diagnostics_")}
        # ...write diag somewhere if helpful
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
    n_jobs: int = 1,
    tag: str = "train"
) -> pd.DataFrame:
    def _worker(pid: str) -> Dict[str, Any]:
        img_p = image_path(images_dir, pid, image_pat)
        msk_p = mask_path(masks_dir,  pid, mask_pat)
        ensure_exists(img_p, "Image")
        ensure_exists(msk_p, "Mask")
        return extract_features_for_case(extractor, img_p, msk_p, pid)

    if n_jobs == 1:
        rows = [ _worker(pid) for pid in tqdm(pids, desc=f"{tag.upper()} radiomics", unit="case") ]
    else:
        rows = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_worker)(pid) for pid in tqdm(pids, desc=f"{tag.upper()} radiomics", unit="case")
        )
    return pd.DataFrame.from_records(rows).set_index("patient_id")

# ------------------------- sanitization (safe + exclude shape2D) -------------------------

def family_of(col: str) -> str:
    for f in ("shape","firstorder","glcm","glrlm","glszm","gldm","ngtdm"):
        if re.search(fr"(^|_)({f})(_|$)", col, flags=re.I):
            return f
    return "other"

def _to_scalar_or_nan(x):
    if x is None:
        return np.nan
    if isinstance(x, (np.generic, float, int, np.floating, np.integer)):
        return float(x)
    if np.isscalar(x):
        try:
            return float(x)
        except Exception:
            return np.nan
    if isinstance(x, str):
        try:
            return float(x)
        except Exception:
            return np.nan
    return np.nan  # arrays/lists/tuples/dicts -> NaN

def sanitize_and_prune(df: pd.DataFrame, outdir: Path, tag: str,
                       max_corr: float = 0.0, max_features: int = 0) -> pd.DataFrame:
    """
    Keep only 'original_*' (excluding shape2D), ensure numeric, drop all-NaN and zero-variance.
    Optional: correlation pruning and/or top-K variance (train only).
    """
    raw_shape = df.shape

    cols = [c for c in df.columns
            if str(c).startswith("original_") and "shape2D_" not in c]
    rad = df[cols].copy()

    # Ensure numeric
    rad = rad.apply(pd.to_numeric, errors="coerce")

    # Drop all-NaN and zero-variance
    allnan_cols = rad.columns[rad.isna().all()].tolist()
    rad = rad.drop(columns=allnan_cols, errors="ignore")
    nunique = rad.nunique(dropna=True)
    zerovar_cols = nunique[nunique <= 1].index.tolist()
    rad = rad.drop(columns=zerovar_cols, errors="ignore")

    # Debug artifacts
    fam_map = pd.Series({c: family_of(c) for c in rad.columns})
    fam_map.value_counts().to_csv(outdir / f"family_counts_{tag}.csv")
    fam_map.to_csv(outdir / f"families_{tag}.csv")
    rad.to_csv(outdir / f"features_{tag}_final.csv", index=True)
    pd.DataFrame({"dropped_allnan": allnan_cols}).to_csv(outdir / f"dropped_allnan_{tag}.csv", index=False)
    pd.DataFrame({"dropped_zerovar": zerovar_cols}).to_csv(outdir / f"dropped_zerovar_{tag}.csv", index=False)

    print(f"[DEBUG] {tag}: raw={raw_shape} -> final={rad.shape} "
          f"(drop all-NaN={len(allnan_cols)}, zero-var={len(zerovar_cols)})")

    # Optional correlation pruning
    if max_corr and max_corr > 0.0 and rad.shape[1] > 1:
        corr = rad.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if any(upper[c] >= max_corr)]
        rad = rad.drop(columns=to_drop, errors="ignore")
        rad.to_csv(outdir / f"features_{tag}_after_corr.csv", index=True)
        print(f"[DEBUG] {tag}: corr-pruned at |r|>={max_corr} -> {rad.shape} (dropped {len(to_drop)})")
    else:
        print(f"[DEBUG] {tag}: corr-prune disabled")

    # Optional top-K variance (TRAIN only)
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
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (test)"); plt.legend(); plt.tight_layout()
    plt.savefig(outpath, dpi=200); plt.close()

def plot_pr(y_true, y_prob, outpath: Path):
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision–Recall Curve (test)")
    plt.tight_layout(); plt.savefig(outpath, dpi=200); plt.close()

def plot_calibration(y_true, y_prob, outpath: Path, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='quantile')
    plt.figure()
    plt.plot(prob_pred, prob_true, marker='o')
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency")
    plt.title("Calibration Curve (test)")
    plt.tight_layout(); plt.savefig(outpath, dpi=200); plt.close()

# ------------------------- main -------------------------

def main():
    ap = argparse.ArgumentParser(description="Baseline pCR prediction from radiomics")
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--params", default=None)
    ap.add_argument("--image-pattern", default=None, help='default "{pid}/{pid}_0001.nii.gz"')
    ap.add_argument("--mask-pattern",  default=None, help='default "{pid}.nii.gz"')
    ap.add_argument("--label", type=int, default=None)
    ap.add_argument("--classifier", choices=["logistic","rf"], default="rf")
    ap.add_argument("--rf-n-estimators", type=int, default=500)
    ap.add_argument("--rf-max-depth", type=int, default=None)
    ap.add_argument("--n-proc", type=int, default=1)
    ap.add_argument("--max-features", type=int, default=0)
    ap.add_argument("--max-corr", type=float, default=0.0)
    args = ap.parse_args()

    outdir = Path(args.output); outdir.mkdir(parents=True, exist_ok=True)

    labels = load_csv(args.labels)[["patient_id","pcr"]].set_index("patient_id")
    split  = load_csv(args.split)[["patient_id","split"]].set_index("patient_id")
    split["split"] = split["split"].str.lower().replace({"val":"test"})
    joined = split.join(labels, how="inner")

    train_pids = joined.index[joined["split"]=="train"].tolist()
    test_pids  = joined.index[joined["split"]=="test"].tolist()
    if not train_pids or not test_pids:
        sys.exit("[ERROR] Need at least one train and one test patient.")

    # Build extractor and snapshot enabled features
    params    = read_yaml(args.params) if args.params else {}
    extractor = build_extractor(params, label_override=args.label, outdir=outdir)

    # Extract
    print("[INFO] Extracting radiomics for TRAIN...")
    Xtr = extract_split_features(train_pids, args.images, args.masks,
                                 args.image_pattern, args.mask_pattern,
                                 extractor, n_jobs=args.n_proc, tag="train")
    Xtr.to_csv(outdir / "features_train.csv")

    print("[INFO] Extracting radiomics for TEST...")
    Xte = extract_split_features(test_pids, args.images, args.masks,
                                 args.image_pattern, args.mask_pattern,
                                 extractor, n_jobs=args.n_proc, tag="test")
    Xte.to_csv(outdir / "features_test.csv")

    ytr = labels.loc[Xtr.index, "pcr"].astype(int).values
    yte = labels.loc[Xte.index, "pcr"].astype(int).values

    # Sanitize + align
    Xtr_s = sanitize_and_prune(Xtr, outdir, tag="train",
                               max_corr=args.max_corr, max_features=args.max_features)
    Xte_s_tmp = sanitize_and_prune(Xte, outdir, tag="test",
                                   max_corr=args.max_corr, max_features=0)
    Xte_s = Xte_s_tmp.reindex(columns=Xtr_s.columns, fill_value=np.nan)

    print(f"[DEBUG] Xtr_s shape: {Xtr_s.shape}, Xte_s shape: {Xte_s.shape}")

    # Model
    if args.classifier == "logistic":
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale",  StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(penalty="l2", solver="lbfgs",
                                       max_iter=2000, class_weight="balanced",
                                       random_state=42))
        ])
        classifier_type = "logistic_l2"
    else:
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=args.rf_n_estimators, max_depth=args.rf_max_depth,
                n_jobs=-1, class_weight="balanced", random_state=42
            ))
        ])
        classifier_type = "random_forest"

    pipe.fit(Xtr_s, ytr)

    clf_step = pipe.named_steps["clf"]
    try:
        pos_idx = int(np.where(clf_step.classes_ == 1)[0][0])
    except Exception:
        pos_idx = 1

    p_tr = pipe.predict_proba(Xtr_s)[:, pos_idx]
    p_te = pipe.predict_proba(Xte_s)[:, pos_idx]

    auc_train = float(roc_auc_score(ytr, p_tr)) if len(np.unique(ytr))>1 else float("nan")
    auc_test  = float(roc_auc_score(yte, p_te)) if len(np.unique(yte))>1 else float("nan")

    # Threshold by Youden J on TRAIN
    fpr, tpr, thr = roc_curve(ytr, p_tr)
    thr_opt = float(thr[np.argmax(tpr - fpr)]) if len(thr) else 0.5
    ypred_te = (p_te >= thr_opt).astype(int)

    cm = confusion_matrix(yte, ypred_te, labels=[0,1])
    tn, fp, fn, tp = (cm.ravel() if cm.size == 4 else (np.nan, np.nan, np.nan, np.nan))
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")

    # Save artifacts
    pd.DataFrame({"patient_id": Xte_s.index, "y_true": yte,
                  "y_prob": p_te, "y_pred": ypred_te}).to_csv(outdir / "predictions.csv", index=False)
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
        "n_features_used": int(Xtr_s.shape[1]),
        "classifier_type": classifier_type,
        "class_order": getattr(clf_step, "classes_", []).tolist(),
        "threshold_train_youdenJ": thr_opt,
        "sensitivity_test": sens,
        "specificity_test": spec,
        "tn_fp_fn_tp_test": [int(x) if not (isinstance(x, float) and np.isnan(x)) else None for x in [tn, fp, fn, tp]],
        "calibration": calib_status
    }
    if not (auc_test > 0.5):
        metrics["commentary"] = (
            "AUC_test ≤ 0.5. Consider trying other DCE phases, enabling LoG/Wavelet, "
            "and modest correlation pruning or top-K variance selection."
        )

    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    joblib.dump(pipe, outdir / "model.pkl")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
