"""Is the UChicago ~0.61 vessel-kinetics AUC physiology, or acquisition protocol?

The published Phase 1 bar (``experiments/uchicago_tabular_bar_floored``) is a
logistic regression on 7 per-case vessel-kinetic means, pooled OOF AUC 0.614.
Three of those features carry the acquisition timing directly:

- ``peak_time`` and ``time_to_enhancement`` are divided by that case's own
  acquisition duration (``gnn.data_loader._attach_node_features``), which ranges
  95.6-480.3 s across the cohort;
- ``auc_positive`` is a trapezoid over physical seconds, so it grows with how
  long the scan ran;
- ``washout_slope`` is measured to the last acquired frame, so it depends on
  when acquisition stopped.

Acquisition protocol is not randomly distributed here: the three UChicago
sub-families differ systematically in frame count and sampling interval, and one
of them (``her2_naclike``, 35 cases) is 100% non-pCR. That opens a
protocol -> sub-family -> label path a vessel model can ride without measuring
any vasculature. This script measures how much of the 0.61 travels that path.

Three independent reads, none of which requires trusting the others:

1. **Protocol alone.** Fit the same OOF logistic regression on nothing but
   ``(n_frames, duration, dt_med)``. Whatever AUC that reaches is available with
   zero imaging content.
2. **Protocol regressed out.** Per fold, residualize each of the 7 means on the
   protocol descriptors (fit on train, apply to test, so the correction leaks
   nothing), then refit. What survives is the physiology-only bar.
3. **Protocol held fixed.** Refit within the ``n_frames == 13`` stratum and
   within each sub-family, where protocol varies far less.

Bootstrap CIs use case resampling, and the drop from residualizing is a
**paired** bootstrap (shared resamples) since the two models score the same
cases.

Reads only small CSVs and the per-case ``ufast_times_seconds.npy`` sidecars (no
graph-cache load) -- head-node safe.

Usage:
    python -m analysis.protocol_confound_check \
        --tabular-features experiments/uchicago_tabular_bar_floored/tabular_baseline_features.csv \
        --dce-root /ess/scratch/scratch1/t-9svena/uchicago/preprocessing_out_v5/dce \
        --out-dir experiments/uchicago_protocol_confound
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_N_BOOTSTRAP = 2000
_CI_PCT = (2.5, 97.5)
_TIMES_SIDECAR = "ufast_times_seconds.npy"
# A bootstrap resample or stratum with only one class present has no defined AUC.
_MIN_CLASSES = 2
# The frame count 112 of the 179 cases share -- the largest protocol-matched
# stratum available, used as the non-parametric counterpart to residualizing.
_MATCHED_FRAME_COUNT = 13

# The 7 per-case node-feature means that constitute the published tabular bar.
_MEANS7 = (
    "peak_time_mean",
    "peak_enhancement_mean",
    "time_to_enhancement_mean",
    "washin_slope_mean",
    "washout_slope_mean",
    "auc_positive_mean",
    "radius_mean",
)
# Acquisition descriptors, all derived from the physical-time sidecar alone --
# no voxel, no vessel, no patient anatomy.
_PROTOCOL = ("n_frames", "duration", "dt_med")


def load_protocol(dce_root: Path, case_ids: list[str]) -> pd.DataFrame:
    """Read each case's physical acquisition times into protocol descriptors.

    ``dt_med`` is the median inter-frame interval -- the temporal resolution the
    scanner ran at. A missing sidecar raises: every UChicago v5 case ships one,
    and silently dropping cases would change the cohort this analysis reports on.
    """
    rows = []
    for case_id in case_ids:
        path = dce_root / case_id / _TIMES_SIDECAR
        if not path.is_file():
            raise FileNotFoundError(f"case={case_id}: missing {path}")
        times = np.load(path, allow_pickle=False)
        rows.append(
            {
                "case_id": case_id,
                "n_frames": int(times.size),
                "duration": float(times[-1] - times[0]),
                "dt_med": float(np.median(np.diff(times))),
            }
        )
    return pd.DataFrame(rows)


def oof_predictions(
    features: np.ndarray, y: np.ndarray, folds: np.ndarray
) -> np.ndarray:
    """Out-of-fold probabilities from a standardized logistic regression.

    Matches the estimator the published tabular bar uses, so AUCs here are
    directly comparable to ``experiments/uchicago_tabular_bar_floored``.
    """
    probs = np.zeros(len(y), dtype=float)
    for fold in np.unique(folds):
        train, test = folds != fold, folds == fold
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        model.fit(features[train], y[train])
        probs[test] = model.predict_proba(features[test])[:, 1]
    return probs


def residualize_on_protocol(
    frame: pd.DataFrame, feature_names: tuple[str, ...], folds: np.ndarray
) -> np.ndarray:
    """Remove the protocol-predictable part of each feature, fold-honestly.

    The linear map from protocol to feature is fit on the training split only and
    applied to the held-out split, so the correction carries no test information.
    What remains is the part of each vessel summary that acquisition timing does
    *not* explain.
    """
    residuals = np.zeros((len(frame), len(feature_names)), dtype=float)
    for fold in np.unique(folds):
        train, test = folds != fold, folds == fold
        for column, name in enumerate(feature_names):
            fit = LinearRegression().fit(
                frame.loc[train, list(_PROTOCOL)], frame.loc[train, name]
            )
            residuals[test, column] = frame.loc[test, name] - fit.predict(
                frame.loc[test, list(_PROTOCOL)]
            )
    return residuals


def bootstrap_auc_ci(
    y: np.ndarray, probs: np.ndarray, rng: np.random.Generator
) -> tuple[float, float]:
    """Case-resampling bootstrap CI for a single model's pooled OOF AUC."""
    draws = []
    for _ in range(_N_BOOTSTRAP):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < _MIN_CLASSES:
            continue
        draws.append(roc_auc_score(y[idx], probs[idx]))
    return tuple(np.percentile(draws, _CI_PCT))


def bootstrap_paired_delta(
    y: np.ndarray, probs_a: np.ndarray, probs_b: np.ndarray, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Paired bootstrap of ``AUC(a) - AUC(b)``: mean delta, CI low, CI high.

    Both models score the same cases, so resampling them jointly cancels the
    shared sampling noise that would make two marginal CIs look inconclusive.
    """
    draws = []
    for _ in range(_N_BOOTSTRAP):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < _MIN_CLASSES:
            continue
        draws.append(
            roc_auc_score(y[idx], probs_a[idx]) - roc_auc_score(y[idx], probs_b[idx])
        )
    return float(np.mean(draws)), *np.percentile(draws, _CI_PCT)


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface for the protocol-confound audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tabular-features", type=Path, required=True)
    parser.add_argument("--dce-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    """Run the three reads, print them, and write the CSVs to ``--out-dir``."""
    args = build_parser().parse_args()
    rng = np.random.default_rng(args.seed)

    data = pd.read_csv(args.tabular_features)
    data = data.merge(
        load_protocol(args.dce_root, data["case_id"].tolist()), on="case_id"
    )
    data["sub_family"] = data["case_id"].str.split("_1_3_").str[0]
    y = data["pcr"].to_numpy()
    folds = data["fold"].to_numpy()

    # 1. How much acquisition timing does each modeled feature carry?
    leakage = pd.DataFrame(
        [
            {
                "feature": name,
                **{
                    f"abs_corr_{p}": abs(np.corrcoef(data[name], data[p])[0, 1])
                    for p in _PROTOCOL
                },
                "univariate_auc": max(
                    roc_auc_score(y, data[name]), 1.0 - roc_auc_score(y, data[name])
                ),
            }
            for name in _MEANS7
        ]
    )

    # 2. The three reads.
    probs_full = oof_predictions(data[list(_MEANS7)].to_numpy(), y, folds)
    probs_protocol = oof_predictions(data[list(_PROTOCOL)].to_numpy(), y, folds)
    probs_resid = oof_predictions(
        residualize_on_protocol(data, _MEANS7, folds), y, folds
    )

    arms = {
        "tabular_7means (published bar)": probs_full,
        "protocol_only (n_frames,duration,dt_med)": probs_protocol,
        "tabular_7means_protocol_residualized": probs_resid,
    }
    auc_rows = []
    for name, probs in arms.items():
        low, high = bootstrap_auc_ci(y, probs, rng)
        auc_rows.append(
            {
                "model": name,
                "n": len(y),
                "auc": roc_auc_score(y, probs),
                "ci_low": low,
                "ci_high": high,
            }
        )

    # 3. Protocol held fixed: the shared-frame-count stratum and each sub-family.
    strata = [
        (
            f"n_frames=={_MATCHED_FRAME_COUNT}",
            data[data["n_frames"] == _MATCHED_FRAME_COUNT],
        )
    ]
    strata += [(f"sub_family={fam}", part) for fam, part in data.groupby("sub_family")]
    strata_rows = []
    for label, part in strata:
        if part["pcr"].nunique() < _MIN_CLASSES:
            strata_rows.append(
                {
                    "stratum": label,
                    "n": len(part),
                    "pcr_rate": part["pcr"].mean(),
                    "auc_7means": np.nan,
                    "auc_dt_med_alone": np.nan,
                }
            )
            continue
        part = part.reset_index(drop=True)
        probs = oof_predictions(
            part[list(_MEANS7)].to_numpy(),
            part["pcr"].to_numpy(),
            part["fold"].to_numpy(),
        )
        auc_dt = roc_auc_score(part["pcr"], part["dt_med"])
        strata_rows.append(
            {
                "stratum": label,
                "n": len(part),
                "pcr_rate": part["pcr"].mean(),
                "auc_7means": roc_auc_score(part["pcr"], probs),
                "auc_dt_med_alone": max(auc_dt, 1.0 - auc_dt),
            }
        )

    # 4. Paired: how much AUC does removing protocol cost?
    delta, delta_low, delta_high = bootstrap_paired_delta(
        y, probs_full, probs_resid, rng
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    leakage.to_csv(args.out_dir / "feature_protocol_coupling.csv", index=False)
    pd.DataFrame(auc_rows).to_csv(args.out_dir / "auc_ci.csv", index=False)
    pd.DataFrame(strata_rows).to_csv(args.out_dir / "stratified_auc.csv", index=False)
    pd.DataFrame(
        [
            {
                "comparison": "tabular_7means - protocol_residualized",
                "mean_delta": delta,
                "ci_low": delta_low,
                "ci_high": delta_high,
            }
        ]
    ).to_csv(args.out_dir / "paired_delta.csv", index=False)

    print(leakage.round(3).to_string(index=False))
    print()
    print(pd.DataFrame(auc_rows).round(3).to_string(index=False))
    print()
    print(pd.DataFrame(strata_rows).round(3).to_string(index=False))
    print()
    print(
        f"paired dAUC (published bar - protocol removed): {delta:+.3f} "
        f"[{delta_low:+.3f}, {delta_high:+.3f}]"
    )


if __name__ == "__main__":
    main()
