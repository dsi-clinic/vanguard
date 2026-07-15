"""Phase 0 go/no-go gate for the harmonized single-breast dataset plan.

Checks two things using only what already exists on disk (no new pipeline
code, no retraining):

1. Whether ``pcr`` is associated with the case-level ``bilateral`` flag at
   all, on the full labeled cohort (clinical metadata, ``bilateral`` from
   patient_info JSONs via ``clinical_features.load_clinical_from_patient_info``).
   ``analysis/graph_laterality_feature_analysis.ipynb`` only ever tested
   laterality against graph *feature distributions*, never against the
   ``pcr`` label -- this fills that gap.
2. Whether the existing matched voxel-mode, frozen-fold run
   (``experiments/graph_repr_compare_v1/voxel/.../predictions.csv``) already
   shows worse held-out AUC on bilateral cases than on unilateral cases,
   paired by fold. This is the cheap pre-registration-style diagnostic before
   paying for a breast-mask splitting pipeline: if bilateral cases are *not*
   consistently worse under the current pipeline, that's evidence against the
   dilution-via-pooling hypothesis being the dominant driver.

Usage::

    python -m analysis.laterality_pcr_check \
        --predictions-csv experiments/graph_repr_compare_v1/voxel/gnn_voxel_compare_20260710_121055/gnn_voxel_compare/predictions.csv \
        --patient-info-dir /gpfs/data/karczmar-lab/MAMA-MIA-syn60868042/patient_info_files
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import roc_auc_score

from clinical_features import load_clinical_from_patient_info

#: roc_auc_score requires both classes present in the subset being scored.
_MIN_CLASSES_FOR_AUC = 2

_DEFAULT_PREDICTIONS = Path(
    "experiments/graph_repr_compare_v1/voxel/gnn_voxel_compare_20260710_121055"
    "/gnn_voxel_compare/predictions.csv"
)
_DEFAULT_PATIENT_INFO_DIR = Path(
    "/gpfs/data/karczmar-lab/MAMA-MIA-syn60868042/patient_info_files"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-csv", type=Path, default=_DEFAULT_PREDICTIONS)
    parser.add_argument(
        "--patient-info-dir", type=Path, default=_DEFAULT_PATIENT_INFO_DIR
    )
    parser.add_argument(
        "--out-csv", type=Path, default=Path("analysis/laterality_pcr_check.csv")
    )
    return parser.parse_args()


def load_bilateral_flag(patient_info_dir: Path) -> pd.Series:
    """Return a ``case_id``-indexed boolean Series from ``imaging_data.bilateral``.

    Raises if any row's ``bilateral`` value is missing or not already a bool
    -- a silently-coerced missing/unbilateral flag would corrupt both checks
    below, so this fails loudly rather than dropping or imputing.
    """
    clinical_df = load_clinical_from_patient_info(patient_info_dir)
    if clinical_df["bilateral"].isna().any():
        missing = clinical_df.loc[clinical_df["bilateral"].isna(), "case_id"].tolist()
        raise ValueError(
            f"{len(missing)} cases have no 'bilateral' value: {missing[:10]}..."
        )
    non_bool = ~clinical_df["bilateral"].apply(lambda v: isinstance(v, bool | np.bool_))
    if non_bool.any():
        bad = clinical_df.loc[non_bool, ["case_id", "bilateral"]]
        raise ValueError(f"'bilateral' is not boolean for some cases:\n{bad}")
    return clinical_df.set_index("case_id")["bilateral"].astype(bool)


def pcr_vs_bilateral_association(cases_df: pd.DataFrame) -> dict[str, object]:
    """Chi-square test of independence between ``pcr`` and ``bilateral``."""
    table = pd.crosstab(cases_df["bilateral"], cases_df["pcr"])
    chi2, p_value, dof, _ = chi2_contingency(table)
    pcr_rate_bilateral = cases_df.loc[cases_df["bilateral"], "pcr"].mean()
    pcr_rate_unilateral = cases_df.loc[~cases_df["bilateral"], "pcr"].mean()
    return {
        "contingency_table": table,
        "chi2": chi2,
        "p_value": p_value,
        "dof": dof,
        "pcr_rate_bilateral": pcr_rate_bilateral,
        "pcr_rate_unilateral": pcr_rate_unilateral,
        "n_bilateral": int(cases_df["bilateral"].sum()),
        "n_unilateral": int((~cases_df["bilateral"]).sum()),
    }


def held_out_auc_by_laterality(cases_df: pd.DataFrame) -> pd.DataFrame:
    """Per-fold held-out AUC on the bilateral subset vs. the unilateral subset."""
    rows = []
    for fold, fold_df in cases_df.groupby("fold"):
        row = {"fold": fold, "n_bilateral": int(fold_df["bilateral"].sum())}
        row["n_unilateral"] = int((~fold_df["bilateral"]).sum())
        for label, subset in (
            ("bilateral", fold_df[fold_df["bilateral"]]),
            ("unilateral", fold_df[~fold_df["bilateral"]]),
        ):
            if subset["pcr"].nunique() < _MIN_CLASSES_FOR_AUC:
                row[f"auc_{label}"] = float("nan")
                continue
            row[f"auc_{label}"] = roc_auc_score(subset["pcr"], subset["y_prob"])
        rows.append(row)
    result = pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)
    result["auc_delta_bilateral_minus_unilateral"] = (
        result["auc_bilateral"] - result["auc_unilateral"]
    )
    return result


def main() -> None:
    """Run the pCR-vs-laterality association check and the bilateral-vs-unilateral AUC gate."""
    args = _parse_args()

    predictions = pd.read_csv(args.predictions_csv)
    bilateral = load_bilateral_flag(args.patient_info_dir)

    missing_clinical = set(predictions["case_id"]) - set(bilateral.index)
    if missing_clinical:
        raise ValueError(
            f"{len(missing_clinical)} cases in predictions.csv have no clinical "
            f"row in patient_info_dir: {sorted(missing_clinical)[:10]}..."
        )

    cases_df = predictions.copy()
    cases_df["bilateral"] = cases_df["case_id"].map(bilateral)

    print(f"Loaded {len(cases_df)} cases from {args.predictions_csv}")
    print(
        f"  bilateral={cases_df['bilateral'].sum()}, unilateral={(~cases_df['bilateral']).sum()}"
    )
    print()

    print("=== 1. pCR vs. bilateral association (full cohort) ===")
    assoc = pcr_vs_bilateral_association(cases_df)
    print(assoc["contingency_table"])
    print(f"chi2={assoc['chi2']:.3f}, dof={assoc['dof']}, p={assoc['p_value']:.4f}")
    print(
        f"pCR rate | bilateral:  {assoc['pcr_rate_bilateral']:.3f} (n={assoc['n_bilateral']})"
    )
    print(
        f"pCR rate | unilateral: {assoc['pcr_rate_unilateral']:.3f} (n={assoc['n_unilateral']})"
    )
    print()

    print("=== 2. Held-out AUC by laterality, paired by fold (current pipeline) ===")
    per_fold = held_out_auc_by_laterality(cases_df)
    print(per_fold.to_string(index=False))
    print()
    n_folds_worse = (per_fold["auc_delta_bilateral_minus_unilateral"] < 0).sum()
    n_folds_total = per_fold["auc_delta_bilateral_minus_unilateral"].notna().sum()
    print(
        f"Bilateral AUC < unilateral AUC in {n_folds_worse}/{n_folds_total} folds "
        f"(mean delta = {per_fold['auc_delta_bilateral_minus_unilateral'].mean():.4f})"
    )

    per_fold.to_csv(args.out_csv, index=False)
    print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
