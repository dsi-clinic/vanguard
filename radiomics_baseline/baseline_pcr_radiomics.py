
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===============================================================================
Baseline pCR Prediction via Radiomics (PyRadiomics + RandomForest)
===============================================================================


Example Usage
-------------
python radiomics_baseline/baseline_pcr_radiomics.py \
  --images /net/projects2/vanguard/MAMA-MIA-syn60868042/images \
  --masks  /net/projects2/vanguard/MAMA-MIA-syn60868042/segmentations/expert \
  --labels radiomics_baseline/labels.csv \
  --split  radiomics_baseline/splits_small.csv \
  --output radiomics_baseline/outdir_test_small \
  --params radiomics_baseline/pyradiomics_params.yaml \
  --image-pattern "{pid}/{pid}_0000.nii.gz" \
  --mask-pattern  "{pid}.nii.gz" \
  --classifier rf \
  --rf-n-estimators 500 \
  --n-proc 8

  # Logistic baseline (if you want to compare):
... --classifier logistic


Outputs (written to --output directory)
----------------------------------------
- features_train.csv         radiomic features for TRAIN patients
- features_test.csv          radiomic features for TEST patients
- predictions.csv            [patient_id, y_true, y_prob, y_pred] for TEST
- metrics.json               includes: auc_train, auc_test, n_features_used,
                             classifier_type, threshold (Youden J), confusion
                             matrix counts, sensitivity, specificity, class_order.
                             If auc_test ≤ 0.5, adds a brief commentary.
- model.pkl                  fitted sklearn Pipeline
- roc_test.png               ROC curve on TEST
- pr_curve.png               Precision–Recall curve (optional)
- calibration_curve.png      Calibration plot (optional)


"""

import argparse, json, os, sys, warnings, logging
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

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

# Quiet noisy logs
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.WARNING)
logging.getLogger("radiomics").setLevel(logging.ERROR)
np.random.seed(42)



def read_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data or {}

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert "patient_id" in df.columns, f"CSV must have 'patient_id' column: {path}"
    return df

def path_from_pattern(root: str, pid: str, pattern: Optional[str]) -> Path:
    """If pattern is an absolute path, return it as-is; else join root/pattern."""
    if pattern and os.path.isabs(pattern):
        return Path(pattern)
    rootp = Path(root)
    return rootp / (pattern.format(pid=pid) if pattern else f"{pid}.nii.gz")

def ensure_exists(path: Path, what: str):
    if not Path(path).exists():
        raise FileNotFoundError(f"{what} not found: {path}")

def build_extractor(params: Dict[str, Any], label_override: Optional[int] = None):
    """
    Build a PyRadiomics extractor compatible with older builds:
    - sets LoG sigma via settings if kwargs unsupported
    - supports lower/upper-cased feature class names
    """
    setting = params.get("setting") or {}
    ext = featureextractor.RadiomicsFeatureExtractor(**setting)

    # label override if provided
    if label_override is not None:
        ext.settings["label"] = int(label_override)

    # image types
    if "imageType" in params and params["imageType"] is not None:
        ext.disableAllImageTypes()
        for name, cfg in (params["imageType"] or {}).items():
            try:
                ext.enableImageTypeByName(name, **(cfg or {}))
            except TypeError:
                # older pyradiomics can't take kwargs here (e.g., LoG sigma); push to settings
                if name.lower() == "log" and cfg and "sigma" in cfg:
                    ext.settings["sigma"] = cfg["sigma"]
                ext.enableImageTypeByName(name)
    else:
        # Minimal default if no YAML provided
        ext.enableImageTypeByName("Original")

    # feature classes
    if "featureClass" in params and params["featureClass"] is not None:
        ext.disableAllFeatures()
        for fcls, feats in (params["featureClass"] or {}).items():
            # Try as-given; on failure, try lowercase
            try_names = [fcls, fcls.lower()]
            enabled = False
            for name_try in try_names:
                try:
                    if feats:
                        for feat in feats:
                            ext.enableFeatureByName(name_try, feat)
                    else:
                        ext.enableFeatureClassByName(name_try)
                    enabled = True
                    break
                except Exception:
                    continue
            if not enabled:
                print(f"[WARN] Could not enable feature class '{fcls}'", file=sys.stderr)
    else:
        # Sensible minimal defaults
        for fcls in ("firstorder", "shape", "glcm"):
            try:
                ext.enableFeatureClassByName(fcls)
            except Exception:
                pass

    return ext

def extract_features_for_case(extractor, image_path: Path, mask_path: Path, pid: str) -> Dict[str, Any]:
    try:
        res = extractor.execute(str(image_path), str(mask_path))
        # Keep both features and diagnostics in the CSV; we’ll drop diagnostics for modeling later
        out = {"patient_id": pid}
        for k, v in res.items():
            if k.startswith(("original", "wavelet", "log", "diagnostics")):
                out[k] = v
        return out
    except Exception as e:
        print(f"[WARN] Feature extraction failed for {pid}: {e}", file=sys.stderr)
        return {"patient_id": pid}

def extract_split_features(pids, images_dir, masks_dir, image_pat, mask_pat, extractor, n_jobs=1):
    from joblib import Parallel, delayed

    def _work(pid):
        img_p = path_from_pattern(images_dir, pid, image_pat)
        msk_p = path_from_pattern(masks_dir,  pid, mask_pat)
        ensure_exists(img_p, "Image")
        ensure_exists(msk_p, "Mask")
        return extract_features_for_case(extractor, img_p, msk_p, pid)

    if n_jobs == 1:
        rows = [_work(pid) for pid in pids]
    else:
        rows = Parallel(n_jobs=n_jobs, prefer="processes")(delayed(_work)(pid) for pid in pids)

    return pd.DataFrame.from_records(rows).set_index("patient_id")

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




def main():
    ap = argparse.ArgumentParser(description="Baseline pCR prediction from radiomics")
    ap.add_argument("--images", required=True, help="Dir with images (NIfTI)")
    ap.add_argument("--masks", required=True, help="Dir with masks (NIfTI)")
    ap.add_argument("--labels", required=True, help="CSV with patient_id,pcr")
    ap.add_argument("--split", required=True, help="CSV with patient_id,split (train/test)")
    ap.add_argument("--output", required=True, help="Output directory")
    ap.add_argument("--params", default=None, help="PyRadiomics YAML parameter file")
    ap.add_argument("--image-pattern", default=None, help='e.g. "{pid}/{pid}_0000.nii.gz"')
    ap.add_argument("--mask-pattern", default=None, help='e.g. "{pid}.nii.gz"')
    ap.add_argument("--label", type=int, default=None, help="Mask label override (e.g., 1 or 2)")
    ap.add_argument("--classifier", choices=["logistic","rf"], default="rf")  # default RF
    ap.add_argument("--rf-n-estimators", type=int, default=500)
    ap.add_argument("--rf-max-depth", type=int, default=None)
    ap.add_argument("--n-proc", type=int, default=1, help="Parallel processes for feature extraction")
    args = ap.parse_args()

    outdir = Path(args.output); outdir.mkdir(parents=True, exist_ok=True)

    labels = load_csv(args.labels)[["patient_id","pcr"]].set_index("patient_id")
    split  = load_csv(args.split)[["patient_id","split"]].set_index("patient_id")
    joined = split.join(labels, how="inner")

    train_pids = joined.index[joined["split"].str.lower()=="train"].tolist()
    test_pids  = joined.index[joined["split"].str.lower()=="test"].tolist()
    if len(train_pids) == 0 or len(test_pids) == 0:
        sys.exit("[ERROR] Need at least one train and one test patient.")

    # Build PyRadiomics extractor
    params = read_yaml(args.params) if args.params else {}
    extractor = build_extractor(params, label_override=args.label)

    # Extract features
    print("[INFO] Extracting radiomics for TRAIN...")
    Xtr = extract_split_features(train_pids, args.images, args.masks,
                                 args.image_pattern, args.mask_pattern, extractor,
                                 n_jobs=args.n_proc)
    Xtr.to_csv(outdir / "features_train.csv")

    print("[INFO] Extracting radiomics for TEST...")
    Xte = extract_split_features(test_pids, args.images, args.masks,
                                 args.image_pattern, args.mask_pattern, extractor,
                                 n_jobs=args.n_proc)
    Xte.to_csv(outdir / "features_test.csv")

    ytr = labels.loc[Xtr.index, "pcr"].astype(int).values
    yte = labels.loc[Xte.index, "pcr"].astype(int).values

    # Sanitize features: drop diagnostics*, keep numeric only, drop zero-variance, align columns
    def sanitize(df: pd.DataFrame) -> pd.DataFrame:
        df2 = df.drop(columns=[c for c in df.columns if str(c).startswith("diagnostics")], errors="ignore")
        num = df2.select_dtypes(include=[np.number]).copy()
        if num.shape[1] == 0:
            return num
        num = num.dropna(axis=1, how="all")
        nunique = num.nunique(dropna=True)
        num = num.loc[:, nunique > 1]
        return num

    Xtr_s = sanitize(Xtr)
    Xte_s = sanitize(Xte)
    Xte_s = Xte_s.reindex(columns=Xtr_s.columns, fill_value=np.nan)

    # Build model
    if args.classifier == "logistic":
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(
                penalty="l2", solver="lbfgs", max_iter=2000,
                class_weight="balanced", random_state=42
            ))
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

    # Fit on TRAIN
    pipe.fit(Xtr_s, ytr)

    # Always use the correct probability column for class=1
    clf_step = pipe.named_steps.get("clf", pipe)
    try:
        pos_idx = int(np.where(clf_step.classes_ == 1)[0][0])
    except Exception:
        pos_idx = 1  # fallback; most estimators keep [0, 1] order

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

    # Save predictions
    pd.DataFrame({
        "patient_id": Xte_s.index,
        "y_true": yte,
        "y_prob": p_te,
        "y_pred": ypred_te
    }).to_csv(outdir / "predictions.csv", index=False)

    # Plots
    plot_roc(yte, p_te, outpath=outdir / "roc_test.png")
    plot_pr(yte, p_te, outpath=outdir / "pr_curve.png")
    plot_calibration(yte, p_te, outpath=outdir / "calibration_curve.png")

    # Metrics JSON
    metrics = {
        "auc_train": auc_train,
        "auc_test": auc_test,
        "n_features_used": int(Xtr_s.shape[1]),
        "classifier_type": classifier_type,
        "class_order": getattr(clf_step, "classes_", []).tolist() if hasattr(clf_step, "classes_") else None,
        "threshold_train_youdenJ": thr_opt,
        "sensitivity_test": sens,
        "specificity_test": spec,
        "tn_fp_fn_tp_test": [int(x) if not (isinstance(x, float) and np.isnan(x)) else None for x in [tn, fp, fn, tp]],
    }
    if not (auc_test > 0.5):
        metrics["commentary"] = (
            "AUC_test ≤ 0.5. Possible causes: small sample size, noisy/unaligned masks, "
            "insufficient features/normalization, domain shift, or overfitting."
        )

    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save model
    joblib.dump(pipe, outdir / "model.pkl")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()

