"""Visual QC for the bilateral -> single-breast splitter (``gnn.breast_split``).

Phase 3 of the harmonized single-breast dataset plan: before any training
run consumes the split cohort, this generates one overlay panel per sampled
bilateral case (breast mask, tumor mask, both-breast skeleton, retained
single-breast skeleton) plus a summary CSV and an HTML index, so a human can
verify the tumor-bearing breast is always selected and there's no
contralateral/midline leakage before Phase 4 wires this into training.

Reuses the real-data loading and coordinate-axis-detection helpers from
``tabular/duke_final_symmetry_lines.py`` (mask loading/thresholding,
per-case coordinate-axis detection from morphometry segment coordinates --
assuming a fixed axis order is wrong for at least some real cases, see
``LAB_NOTEBOOK.md`` 2026-07-14) and the panel layout from that script's
``_make_case_figure``, adapted to show the before/after split rather than
just the inferred side.

Usage::

    python -m analysis.breast_split_qc --n-duke 15 --n-ispy2 10 --n-ispy1 4 \
        --outdir qc/breast_split_v1 --seed 0
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import numpy as np
import pandas as pd

from clinical_features import load_clinical_from_patient_info
from gnn.breast_split import (
    detect_case_coord_axes,
    resolve_breast_mask_path,
    resolve_study_dir,
    resolve_tumor_mask_orientation,
    split_case,
)
from tabular.duke_final_symmetry_lines import (
    _load_array,
    _load_mask_matching_shape,
    _project_to_yx,
    _rotate_180_about_x_axis,
)

PATIENT_INFO_DIR = Path(
    "/gpfs/data/karczmar-lab/MAMA-MIA-syn60868042/patient_info_files"
)
TUMOR_MASK_DIR = Path(
    "/gpfs/data/karczmar-lab/MAMA-MIA-syn60868042/segmentations/expert"
)

_DATASET_SAMPLE_ARGS = ("duke", "ispy1", "ispy2", "nact")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in _DATASET_SAMPLE_ARGS:
        parser.add_argument(f"--n-{name}", type=int, default=0)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _study_dir(dataset: str, case_id: str) -> Path:
    return resolve_study_dir(dataset, case_id)


def _case_has_required_files(dataset: str, case_id: str) -> bool:
    study_dir = _study_dir(dataset, case_id)
    skeleton_path = study_dir / f"{case_id}_skeleton_4d_exam_mask.npy"
    tumor_path = TUMOR_MASK_DIR / f"{case_id}.nii.gz"
    return (
        skeleton_path.exists()
        and tumor_path.exists()
        and resolve_breast_mask_path(dataset, case_id) is not None
    )


def select_sample(sample_sizes: dict[str, int], *, seed: int) -> pd.DataFrame:
    """Sample bilateral cases per dataset that have skeleton/breast/tumor masks.

    Raises if a dataset's request exceeds the number of eligible cases --
    silently returning fewer than asked would make the QC sample size
    reviewers see not match what was requested, without any signal.
    """
    clinical = load_clinical_from_patient_info(PATIENT_INFO_DIR)
    bilateral = clinical[clinical["bilateral"] == True]  # noqa: E712

    rows = []
    rng = np.random.default_rng(seed)
    for dataset, n in sample_sizes.items():
        if n <= 0:
            continue
        dataset_upper = dataset.upper()
        candidates = bilateral.loc[
            bilateral["dataset"] == dataset_upper, "case_id"
        ].tolist()
        eligible = [c for c in candidates if _case_has_required_files(dataset_upper, c)]
        if len(eligible) < n:
            raise ValueError(
                f"{dataset_upper}: requested {n} cases but only {len(eligible)} "
                f"bilateral cases have skeleton+breast_mask+tumor_mask available "
                f"(of {len(candidates)} bilateral cases total)."
            )
        chosen = rng.choice(sorted(eligible), size=n, replace=False)
        rows.extend({"case_id": c, "dataset": dataset_upper} for c in chosen)
    return pd.DataFrame(rows)


def _detect_axes(
    case_id: str, dataset: str, array_shape: tuple[int, int, int]
) -> tuple[int, int, int]:
    return detect_case_coord_axes(case_id, _study_dir(dataset, case_id), array_shape)


def run_one_case(case_id: str, dataset: str) -> dict[str, object]:
    """Load real data for one case, run the splitter, return a summary row."""
    study_dir = _study_dir(dataset, case_id)
    skeleton = np.asarray(
        _load_array(study_dir / f"{case_id}_skeleton_4d_exam_mask.npy"), dtype=bool
    )
    z_axis, y_axis, x_axis = _detect_axes(case_id, dataset, skeleton.shape)

    breast_mask_path = resolve_breast_mask_path(dataset, case_id)
    breast_mask, _ = _load_mask_matching_shape(
        breast_mask_path, expected_shape=skeleton.shape, threshold=0.5
    )
    tumor_mask, _ = _load_mask_matching_shape(
        TUMOR_MASK_DIR / f"{case_id}.nii.gz",
        expected_shape=skeleton.shape,
        threshold=0.5,
    )
    tumor_mask, tumor_was_flipped = resolve_tumor_mask_orientation(
        tumor_mask, breast_mask=breast_mask
    )

    result = split_case(
        case_id,
        skeleton=skeleton,
        breast_mask=breast_mask,
        tumor_mask=tumor_mask,
        x_axis=x_axis,
        y_axis=y_axis,
        z_axis=z_axis,
    )
    row = {
        "case_id": case_id,
        "dataset": dataset,
        "included": result.included,
        "exclude_reason": result.exclude_reason,
        "tumor_mask_was_flipped": tumor_was_flipped,
        "retained_side": result.retained_side,
        "midline_x": result.midline_x,
        "tumor_side_voxel_fraction": result.tumor_side_voxel_fraction,
        "left_breast_voxels": result.left_breast_voxels,
        "right_breast_voxels": result.right_breast_voxels,
        "skeleton_voxels_before": int(skeleton.sum()),
        "skeleton_voxels_after": (
            int(result.skeleton_single_breast.sum()) if result.included else None
        ),
    }
    return row, {
        "coord_to_axis": (z_axis, y_axis, x_axis),
        "skeleton": skeleton,
        "breast_mask": breast_mask,
        "tumor_mask": tumor_mask,
        "result": result,
    }


def make_qc_figure(
    row: dict[str, object], arrays: dict[str, object], outdir: Path
) -> Path:
    """Render one case's before/after overlay panel and save it as a PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    case_id = row["case_id"]
    z_axis, y_axis, x_axis = arrays["coord_to_axis"]
    coord_to_axis = (z_axis, y_axis, x_axis)
    skeleton = arrays["skeleton"]
    breast_mask = arrays["breast_mask"]
    tumor_mask = arrays["tumor_mask"]
    result = arrays["result"]

    skeleton_disp = _rotate_180_about_x_axis(skeleton, x_axis=x_axis)
    breast_disp = _rotate_180_about_x_axis(breast_mask, x_axis=x_axis)
    tumor_disp = _rotate_180_about_x_axis(tumor_mask, x_axis=x_axis)
    skeleton_yx = _project_to_yx(skeleton_disp, coord_to_axis=coord_to_axis)
    breast_yx = _project_to_yx(breast_disp, coord_to_axis=coord_to_axis)
    tumor_yx = _project_to_yx(tumor_disp, coord_to_axis=coord_to_axis)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    left_ax, right_ax = axes

    left_ax.imshow(breast_yx, cmap="Greys", alpha=0.25, interpolation="nearest")
    left_ax.imshow(skeleton_yx, cmap="Blues", alpha=0.8, interpolation="nearest")
    left_ax.imshow(np.ma.masked_where(~tumor_yx, tumor_yx), cmap="Reds", alpha=0.5)
    left_ax.contour(
        tumor_yx.astype(float), levels=[0.5], colors="yellow", linewidths=1.2
    )
    if result.midline_x is not None:
        left_ax.axvline(result.midline_x, color="cyan", linewidth=1.5)
    left_ax.set_title("Before: both breasts + tumor + midline")
    left_ax.set_aspect("equal")

    if result.included:
        retained_disp = _rotate_180_about_x_axis(
            result.skeleton_single_breast, x_axis=x_axis
        )
        retained_yx = _project_to_yx(retained_disp, coord_to_axis=coord_to_axis)
        right_ax.imshow(breast_yx, cmap="Greys", alpha=0.25, interpolation="nearest")
        right_ax.imshow(retained_yx, cmap="Greens", alpha=0.85, interpolation="nearest")
        right_ax.imshow(np.ma.masked_where(~tumor_yx, tumor_yx), cmap="Reds", alpha=0.5)
        right_ax.set_title(f"After: retained_side={result.retained_side}")
    else:
        right_ax.imshow(np.zeros_like(skeleton_yx, dtype=float), cmap="Greys")
        right_ax.text(
            0.5,
            0.5,
            f"EXCLUDED:\n{result.exclude_reason}",
            ha="center",
            va="center",
            fontsize=13,
            color="red",
            transform=right_ax.transAxes,
        )
    right_ax.set_aspect("equal")

    fig.suptitle(
        f"{case_id} ({row['dataset']}) | included={result.included} | "
        f"side={result.retained_side} | tumor_frac={result.tumor_side_voxel_fraction}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = outdir / f"{case_id}_breast_split_qc.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def write_index(
    outdir: Path, rows: list[dict[str, object]], figure_paths: dict[str, Path]
) -> None:
    """Write an HTML index with an embedded thumbnail per case for quick review."""
    lines = [
        "<html><body>",
        "<h1>Breast-split QC</h1>",
        "<table border='1' cellspacing='0' cellpadding='4'>",
        "<tr><th>case</th><th>dataset</th><th>included</th><th>side</th>"
        "<th>exclude_reason</th><th>tumor_mask_flipped</th><th>tumor_frac</th>"
        "<th>voxels before/after</th><th>figure</th></tr>",
    ]
    for row in rows:
        case_id = str(row["case_id"])
        fig_path = figure_paths.get(case_id)
        fig_link = ""
        if fig_path is not None:
            fig_link = (
                f"<a href='{html.escape(fig_path.name)}'>png</a><br>"
                f"<img src='{html.escape(fig_path.name)}' width='420'>"
            )
        lines.append(
            "<tr>"
            f"<td>{html.escape(case_id)}</td>"
            f"<td>{html.escape(str(row['dataset']))}</td>"
            f"<td>{row['included']}</td>"
            f"<td>{html.escape(str(row['retained_side']))}</td>"
            f"<td>{html.escape(str(row['exclude_reason']))}</td>"
            f"<td>{row['tumor_mask_was_flipped']}</td>"
            f"<td>{row['tumor_side_voxel_fraction']}</td>"
            f"<td>{row['skeleton_voxels_before']} / {row['skeleton_voxels_after']}</td>"
            f"<td>{fig_link}</td>"
            "</tr>"
        )
    lines.extend(["</table>", "</body></html>"])
    (outdir / "index.html").write_text("\n".join(lines))


def main() -> None:
    """Select the sample, run the splitter on each case, write panels + index."""
    args = _parse_args()
    if args.outdir.exists() and not args.overwrite:
        raise FileExistsError(f"Output dir exists: {args.outdir}. Use --overwrite.")
    args.outdir.mkdir(parents=True, exist_ok=True)

    sample_sizes = {name: getattr(args, f"n_{name}") for name in _DATASET_SAMPLE_ARGS}
    sample = select_sample(sample_sizes, seed=args.seed)
    print(f"Selected {len(sample)} cases:\n{sample.to_string(index=False)}")

    rows = []
    figure_paths = {}
    for _, sample_row in sample.iterrows():
        case_id, dataset = sample_row["case_id"], sample_row["dataset"]
        print(f"Processing {case_id} ({dataset})...")
        row, arrays = run_one_case(case_id, dataset)
        rows.append(row)
        figure_paths[case_id] = make_qc_figure(row, arrays, args.outdir)

    summary = pd.DataFrame(rows)
    summary.to_csv(args.outdir / "breast_split_qc_summary.csv", index=False)
    write_index(args.outdir, rows, figure_paths)

    print(f"\n{summary['included'].sum()}/{len(summary)} included")
    print(summary.groupby("dataset")["included"].agg(["sum", "count"]))
    print(f"\nWrote {args.outdir / 'index.html'}")


if __name__ == "__main__":
    main()
