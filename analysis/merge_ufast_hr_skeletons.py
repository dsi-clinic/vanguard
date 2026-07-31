r"""Exploratory: merge the HR-mapped skeleton with the UFAST-direct skeleton.

**Not the supported route** -- same caveat as ``analysis.skeletonize_ufast_vessels``
and ``analysis.segment_ufast_vessels``: the HR-mapped skeleton
(``preprocessing.pipeline.map_case``'s output) is the one trusted/supported
centerline for an exam. This script combines it with the UFAST-direct
skeleton (``analysis.skeletonize_ufast_vessels``'s output) to see whether the
UFAST route recovers real structure the HR route missed, without discarding
HR's authority over voxels both routes could plausibly describe differently.

Both skeletons are binary voxel masks on the *same* physical UFAST grid (no
resampling needed here -- see ``analysis.skeletonize_ufast_vessels``'s
``render_skeleton_comparison_mp4``, which already asserts shape equality
between them). But because the HR-mapped skeleton is produced by a
nearest-physical-voxel *rasterization* (not interpolation) of the HR route's
native-resolution skeleton, a real shared vessel will rarely land on the
exact same voxel in both sources -- so "agreement" is computed via a small
dilation-radius proximity match, not exact voxel equality.

Merge policy (HR is authoritative; UFAST only adds, never overrides):
  - Every HR-mapped skeleton voxel is always kept.
  - A UFAST-direct voxel within ``--dilation-radius-voxels`` of an HR voxel
    is "confirmed" (both routes agree on that vessel) but doesn't add any new
    coverage -- it's HR's voxel that gets kept either way.
  - A UFAST-direct voxel with no nearby HR voxel is a merge candidate only if
    it belongs to a connected component at least ``VESSEL_LIKE_MIN_VOXELS``
    large (reusing ``analysis.segment_ufast_vessels``'s existing noise-vs-real
    threshold) -- this drops UFAST-only noise specks before they can pollute
    the merge.
  - A UFAST-only component that fails the size filter gets a second, additive
    chance (``--enable-kinetics-filter``, on by default): it's kept anyway if
    its mean per-voxel contrast-enhancement peak-SNR (computed directly from
    the raw motion-corrected UFAST signal, same ``peak_snr`` concept
    ``analysis.bolus_arrival_time`` already validates and gates bolus-arrival
    detection on) clears ``--kinetics-min-peak-snr``, AND it's spatially
    continuous with already-trusted vasculature (HR skeleton or a
    size-qualifying UFAST component) within ``--kinetics-adjacency-radius-voxels``.
    Both conditions are required: prototyping found that amplitude alone
    (peak-SNR) is a weak discriminator in this data -- typical background
    tissue commonly shows peak-SNR of 3-4+ under this per-voxel metric, so a
    bare amplitude threshold lets in components with no real relationship to
    known vasculature. Spatial continuity is the condition doing the real
    work of keeping this trustworthy. This path is strictly additive to the
    size filter (never removes anything the size filter already accepted).
  - The candidate mask (HR | qualifying UFAST-only) is re-cleaned with TC4D's
    own internal topology finalization (``graph_extraction.tc4d._finalize_exam_mask_topology``)
    so any UFAST-only/repair additions are a proper single-voxel-wide,
    support-filtered skeleton, not a blob of unioned voxels -- but that
    cleanup has no mechanism to preserve trusted HR voxels (it can drop whole
    HR-only components via its own automatic support-score threshold), so the
    raw HR skeleton is unioned back in *after* cleanup, unconditionally, to
    make "every HR-mapped skeleton voxel is always kept" actually true rather
    than just true of cleanup's input. That stage's own repair step can
    bridge two components with a handful of new voxels that belong to neither
    original source -- those get their own provenance label rather than being folded
    into "UFAST-only-added".

Reads both source skeletons as read-only inputs; never writes into
``centerlines/...`` or ``ufast_tc4d/``.

Usage::

    micromamba run -n vanguard python -m analysis.merge_ufast_hr_skeletons \\
        --preprocessing-root /gpfs/data/karczmar-lab/workspaces/saritbose/preprocessing_out \\
        --output-root /gpfs/data/karczmar-lab/workspaces/saritbose/ufast-seg \\
        --exam-id uch_nac_1_3_1004294011319227272751368633606051346978841464528
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.segment_ufast_vessels import connectivity_stats  # noqa: E402
from analysis.skeletonize_ufast_vessels import (  # noqa: E402
    find_hr_mapped_skeleton,
    is_tc4d_complete,
)
from preprocessing.dicom import DicomGeometry  # noqa: E402
from preprocessing.merge import (  # noqa: E402
    DEFAULT_DILATION_RADIUS_VOXELS,
    DEFAULT_KINETICS_ADJACENCY_RADIUS_VOXELS,
    DEFAULT_KINETICS_MIN_PEAK_SNR,
    LABEL_CONFIRMED_BOTH,
    LABEL_HR_ONLY,
    LABEL_KINETICS_ADJACENCY_ADDED,
    LABEL_REPAIR_BRIDGE_ADDED,
    LABEL_UFAST_ONLY_ADDED,
    VESSEL_LIKE_MIN_VOXELS,
    merge_skeletons,
)
from preprocessing.spatial import physical_points_from_zyx  # noqa: E402


def find_hr_mapped_support(preprocessing_root: Path, exam_id: str) -> Path:
    """Sibling support-mask file next to ``find_hr_mapped_skeleton``'s result."""
    skeleton_path = find_hr_mapped_skeleton(preprocessing_root, exam_id)
    support_path = skeleton_path.parent / f"{exam_id}_skeleton_4d_exam_support_mask.npy"
    if not support_path.exists():
        raise FileNotFoundError(f"no HR-mapped support mask at {support_path}")
    return support_path


def load_skeleton_pair(
    preprocessing_root: Path, output_root: Path, exam_id: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load (hr_skeleton, hr_support, ufast_skeleton, ufast_support), all (z,y,x) bool."""
    hr_skeleton = np.load(find_hr_mapped_skeleton(preprocessing_root, exam_id)).astype(
        bool
    )
    hr_support = np.load(find_hr_mapped_support(preprocessing_root, exam_id)).astype(
        bool
    )
    ufast_dir = output_root / exam_id / "ufast_tc4d"
    ufast_skeleton = np.load(ufast_dir / "exam_skeleton_zyx.npy").astype(bool)
    ufast_support = np.load(ufast_dir / "exam_support_zyx.npy").astype(bool)
    shapes = {
        "hr_skeleton": hr_skeleton.shape,
        "hr_support": hr_support.shape,
        "ufast_skeleton": ufast_skeleton.shape,
        "ufast_support": ufast_support.shape,
    }
    if len(set(shapes.values())) != 1:
        raise ValueError(f"case={exam_id}: shape mismatch across inputs: {shapes}")
    return hr_skeleton, hr_support, ufast_skeleton, ufast_support


def merge_exam(
    preprocessing_root: Path,
    output_root: Path,
    exam_id: str,
    *,
    dilation_radius_voxels: int = DEFAULT_DILATION_RADIUS_VOXELS,
    vessel_like_min_voxels: int = VESSEL_LIKE_MIN_VOXELS,
    enable_kinetics_filter: bool = True,
    kinetics_min_peak_snr: float = DEFAULT_KINETICS_MIN_PEAK_SNR,
    kinetics_adjacency_radius_voxels: int = DEFAULT_KINETICS_ADJACENCY_RADIUS_VOXELS,
) -> Path:
    """Merge one exam's HR-mapped and UFAST-direct skeletons. Returns the merge output dir."""
    merge_dir = output_root / exam_id / "merged_tc4d"
    if merge_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing merge output: {merge_dir}"
        )

    t_start = time.perf_counter()
    hr_skeleton, hr_support, ufast_skeleton, ufast_support = load_skeleton_pair(
        preprocessing_root, output_root, exam_id
    )

    merged_skeleton, merged_support, provenance, provenance_label = merge_skeletons(
        preprocessing_root=preprocessing_root,
        exam_id=exam_id,
        hr_skeleton=hr_skeleton,
        hr_support=hr_support,
        ufast_skeleton=ufast_skeleton,
        ufast_support=ufast_support,
        dilation_radius_voxels=dilation_radius_voxels,
        vessel_like_min_voxels=vessel_like_min_voxels,
        enable_kinetics_filter=enable_kinetics_filter,
        kinetics_min_peak_snr=kinetics_min_peak_snr,
        kinetics_adjacency_radius_voxels=kinetics_adjacency_radius_voxels,
    )
    provenance["route"] = (
        "hr_ufast_skeleton_merge (exploratory, HR-authoritative + filtered UFAST additions)"
    )
    provenance["wall_seconds"] = float(time.perf_counter() - t_start)

    merge_dir.mkdir(parents=True)
    try:
        np.save(merge_dir / "merged_skeleton_zyx.npy", merged_skeleton.astype(np.uint8))
        np.save(merge_dir / "merged_support_zyx.npy", merged_support.astype(np.uint8))
        np.save(merge_dir / "provenance_label_zyx.npy", provenance_label)
        (merge_dir / "merge_provenance.json").write_text(
            json.dumps(provenance, indent=2)
        )
    except Exception:
        shutil.rmtree(merge_dir, ignore_errors=True)
        raise
    return merge_dir


def is_merge_complete(output_root: Path, exam_id: str) -> bool:
    """Whether an exam's merge output has all expected artifacts.

    Same rationale as ``is_tc4d_complete``/``is_exam_complete``: a preemption
    kill on this cluster doesn't give the except-block a chance to clean up,
    so a plain directory-exists check can silently treat a partial run as
    done forever.
    """
    merge_dir = output_root / exam_id / "merged_tc4d"
    expected = (
        "merged_skeleton_zyx.npy",
        "merged_support_zyx.npy",
        "provenance_label_zyx.npy",
        "merge_provenance.json",
    )
    return all((merge_dir / name).exists() for name in expected)


def discover_mergeable_exam_ids(
    preprocessing_root: Path, output_root: Path
) -> list[str]:
    """Every exam with both a completed UFAST TC4D skeleton and an HR-mapped one."""
    exam_ids = []
    for exam_id in sorted(
        d.name
        for d in output_root.iterdir()
        if d.is_dir() and (d / "ufast_tc4d").exists()
    ):
        if not is_tc4d_complete(output_root, exam_id):
            continue
        try:
            find_hr_mapped_skeleton(preprocessing_root, exam_id)
        except FileNotFoundError:
            continue
        exam_ids.append(exam_id)
    return exam_ids


def _mip_label_rgb(provenance_label: np.ndarray) -> np.ndarray:
    """Project the (z,y,x) label volume to a 2D RGB MIP, one solid color per class.

    Same green/magenta/white convention already used elsewhere in this
    exploration (HR=green, UFAST=magenta, overlap=white), plus cyan for the
    cleanup stage's own repair-bridge voxels (belong to neither source) and
    orange for the kinetics-adjacency additive filter's own voxels (belong to
    UFAST but were rejected by the plain size filter).
    """
    colors = {
        LABEL_HR_ONLY: (0.18, 0.80, 0.25),  # green
        LABEL_CONFIRMED_BOTH: (1.0, 1.0, 1.0),  # white
        LABEL_UFAST_ONLY_ADDED: (0.77, 0.31, 0.81),  # magenta
        LABEL_KINETICS_ADJACENCY_ADDED: (1.0, 0.55, 0.10),  # orange
        LABEL_REPAIR_BRIDGE_ADDED: (0.20, 0.85, 0.85),  # cyan
    }
    projected_label = np.zeros(provenance_label.shape[1:], dtype=np.uint8)
    for label_value in (
        LABEL_HR_ONLY,
        LABEL_CONFIRMED_BOTH,
        LABEL_UFAST_ONLY_ADDED,
        LABEL_KINETICS_ADJACENCY_ADDED,
        LABEL_REPAIR_BRIDGE_ADDED,
    ):
        present = np.any(provenance_label == label_value, axis=0)
        projected_label[present] = label_value

    rgb = np.zeros((*projected_label.shape, 3), dtype=np.float32)
    for label_value, color in colors.items():
        rgb[projected_label == label_value] = color
    return rgb


def save_merge_overlay_mip(output_root: Path, exam_id: str) -> Path:
    """Save an RGB MIP of the merged skeleton, color-coded by provenance label."""
    merge_dir = output_root / exam_id / "merged_tc4d"
    provenance_label = np.load(merge_dir / "provenance_label_zyx.npy")
    rgb = _mip_label_rgb(provenance_label)

    fig, ax = plt.subplots(figsize=(6, 6.5))
    ax.imshow(np.transpose(rgb, (1, 0, 2)), origin="lower")
    ax.set_title(
        f"{exam_id}\nMerged skeleton -- green=HR only, white=confirmed by both, "
        f"magenta=UFAST added (size filter), orange=kinetics-adjacency added, cyan=repair bridge",
        fontsize=8,
        wrap=True,
    )
    ax.set_axis_off()
    fig.tight_layout()
    out_path = merge_dir / f"{exam_id}_merged_skeleton_mip.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _pick_labeled_points(
    provenance_label: np.ndarray, label_value: int, cap: int, seed: int
) -> np.ndarray:
    idx = np.argwhere(provenance_label == label_value)
    if len(idx) > cap:
        idx = idx[np.random.default_rng(seed).choice(len(idx), cap, replace=False)]
    return idx


#: Point cap per trace, matching analysis.segment_ufast_vessels's interactive viewer.
_INTERACTIVE_POINT_CAP = 80000


def write_merge_interactive_html(
    preprocessing_root: Path, output_root: Path, exam_id: str, out_path: Path
) -> None:
    """Self-contained Plotly 3D HTML: the four merge provenance classes.

    Extends ``analysis.segment_ufast_vessels.write_interactive_comparison_html``'s
    pattern (same color convention, same dark theme, same point-cap/hover
    style) from a 3-way probability-mask comparison to this merge's 4-way
    skeleton-voxel provenance.
    """
    import plotly.graph_objects as go

    provenance = json.loads(
        (
            preprocessing_root / "work" / exam_id / "preprocessing_provenance.json"
        ).read_text()
    )
    ufast_geometry = DicomGeometry.from_dict(provenance["ufast_output_geometry"])

    merge_dir = output_root / exam_id / "merged_tc4d"
    provenance_label = np.load(merge_dir / "provenance_label_zyx.npy")

    def _trace(
        label_value: int, seed: int, color: str, name: str, size: float
    ) -> go.Scatter3d | None:
        idx = _pick_labeled_points(
            provenance_label, label_value, _INTERACTIVE_POINT_CAP, seed
        )
        if len(idx) == 0:
            return None
        pts = physical_points_from_zyx(idx, ufast_geometry)
        n_total = int((provenance_label == label_value).sum())
        return go.Scatter3d(
            x=pts[:, 0],
            y=pts[:, 1],
            z=pts[:, 2],
            mode="markers",
            marker={"size": size, "color": color, "opacity": 0.8},
            name=f"{name} (n={n_total})",
        )

    traces = []
    for label_value, seed, color, name, size in (
        (LABEL_HR_ONLY, 0, "#2ecc40", "HR only", 1.6),
        (LABEL_UFAST_ONLY_ADDED, 1, "#c44ecf", "UFAST added (size filter)", 1.8),
        (LABEL_KINETICS_ADJACENCY_ADDED, 4, "#ff8c1a", "kinetics-adjacency added", 1.9),
        (LABEL_CONFIRMED_BOTH, 2, "#ffffff", "confirmed (both)", 2.0),
        (LABEL_REPAIR_BRIDGE_ADDED, 3, "#33d9d9", "repair bridge", 2.2),
    ):
        trace = _trace(label_value, seed, color, name, size)
        if trace is not None:
            traces.append(trace)

    def _scene_axis_no_grid(title: str) -> dict:
        return {
            "title": title,
            "showgrid": False,
            "showbackground": False,
            "zeroline": False,
            "showline": True,
            "ticks": "outside",
            "gridcolor": "rgba(0,0,0,0)",
            "backgroundcolor": "rgba(0,0,0,0)",
        }

    fig = go.Figure(data=traces)
    fig.update_layout(
        title={
            "text": (
                f"<b>{exam_id}</b> -- merged HR/UFAST-direct skeleton<br>"
                f"<span style='font-size:13px'>"
                f"green = HR only · white = confirmed by both · magenta = UFAST added (size filter) · "
                f"orange = kinetics-adjacency added · cyan = repair bridge · "
                f"drag to rotate, click legend to toggle"
                f"</span>"
            ),
            "x": 0.02,
        },
        scene={
            "aspectmode": "data",
            "xaxis": _scene_axis_no_grid("x (mm, LPS)"),
            "yaxis": _scene_axis_no_grid("y (mm, LPS)"),
            "zaxis": _scene_axis_no_grid("z (mm, LPS)"),
            "bgcolor": "#111111",
        },
        legend={"itemsizing": "constant", "x": 0.0, "y": -0.02, "orientation": "h"},
        margin={"l": 0, "r": 0, "t": 90, "b": 0},
        paper_bgcolor="#1a1a1a",
        font={"color": "#dddddd"},
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs=True, full_html=True)


def compute_merge_stats(output_root: Path, exam_id: str) -> dict[str, float | str]:
    """Per-exam merge summary for a mentor-facing table."""
    merge_dir = output_root / exam_id / "merged_tc4d"
    merge_provenance = json.loads((merge_dir / "merge_provenance.json").read_text())
    merged_skeleton = np.load(merge_dir / "merged_skeleton_zyx.npy").astype(bool)

    hr_voxels = merge_provenance["hr_skeleton_voxels"]
    merged_voxels_final = merge_provenance["merged_voxels_final"]
    pct_added_over_hr = (
        (merged_voxels_final - hr_voxels) / hr_voxels if hr_voxels > 0 else float("nan")
    )

    row: dict[str, float | str] = {
        "exam_id": exam_id,
        "hr_skeleton_voxels": hr_voxels,
        "ufast_skeleton_voxels": merge_provenance["ufast_skeleton_voxels"],
        "confirmed_ufast_voxels": merge_provenance["confirmed_ufast_voxels"],
        "ufast_only_candidate_voxels": merge_provenance["ufast_only_candidate_voxels"],
        "ufast_added_voxels_pre_cleanup": merge_provenance[
            "ufast_added_voxels_pre_cleanup"
        ],
        "voxels_dropped_as_noise": merge_provenance["filter_stats"][
            "voxels_dropped_as_noise"
        ],
        "n_components_ufast_only_total": merge_provenance["filter_stats"][
            "n_components_total"
        ],
        "n_components_ufast_only_kept": merge_provenance["filter_stats"][
            "n_components_kept"
        ],
        "merged_voxels_final": merged_voxels_final,
        "pct_added_over_hr": pct_added_over_hr,
        "dilation_radius_voxels": merge_provenance["dilation_radius_voxels"],
        "vessel_like_min_voxels": merge_provenance["vessel_like_min_voxels"],
        **{f"label_{k}": v for k, v in merge_provenance["label_voxel_counts"].items()},
    }
    for key, value in connectivity_stats(merged_skeleton).items():
        row[f"merged_{key}"] = value
    return row


def main() -> None:
    """CLI entrypoint: merge HR-mapped and UFAST-direct skeletons for one or more exams."""
    p = argparse.ArgumentParser(
        description=(
            "Exploratory: merge the HR-mapped skeleton with the UFAST-direct "
            "skeleton, HR-authoritative with connectivity-filtered UFAST "
            "additions (see module docstring for the merge policy)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--preprocessing-root", required=True, type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--exam-id", action="append", dest="exam_ids")
    p.add_argument(
        "--full-cohort",
        action="store_true",
        help="Merge every exam with both a completed UFAST TC4D skeleton and an HR-mapped one.",
    )
    p.add_argument(
        "--dilation-radius-voxels", type=int, default=DEFAULT_DILATION_RADIUS_VOXELS
    )
    p.add_argument("--vessel-like-min-voxels", type=int, default=VESSEL_LIKE_MIN_VOXELS)
    p.add_argument(
        "--disable-kinetics-filter",
        action="store_true",
        help=(
            "Skip the additive kinetics-adjacency filter (peak-SNR + spatial "
            "continuity to trusted vasculature) and use only the plain size "
            "filter, matching pre-kinetics-filter behavior."
        ),
    )
    p.add_argument(
        "--kinetics-min-peak-snr", type=float, default=DEFAULT_KINETICS_MIN_PEAK_SNR
    )
    p.add_argument(
        "--kinetics-adjacency-radius-voxels",
        type=int,
        default=DEFAULT_KINETICS_ADJACENCY_RADIUS_VOXELS,
    )
    p.add_argument(
        "--mip-only",
        action="store_true",
        help="Skip merging; just (re)generate the merge overlay MIP for exams already merged.",
    )
    p.add_argument(
        "--interactive-html",
        action="store_true",
        help="Skip merging; write the interactive 3D merge-provenance HTML for exams already merged.",
    )
    p.add_argument(
        "--stats-csv",
        type=Path,
        default=None,
        help="Skip merging; write per-exam merge summary statistics to this CSV.",
    )
    args = p.parse_args()
    if not args.full_cohort and not args.exam_ids:
        p.error("--exam-id is required unless --full-cohort is set")

    exam_ids = (
        discover_mergeable_exam_ids(args.preprocessing_root, args.output_root)
        if args.full_cohort
        else args.exam_ids
    )

    if args.stats_csv is not None:
        import csv

        rows: list[dict[str, float | str]] = []
        for exam_id in exam_ids:
            if not is_merge_complete(args.output_root, exam_id):
                print(f"[{exam_id}] not merged yet (or incomplete), skipping")
                continue
            try:
                rows.append(compute_merge_stats(args.output_root, exam_id))
            except Exception as exc:  # noqa: BLE001 - one bad exam shouldn't kill a full-cohort report
                print(f"[{exam_id}] could not compute merge stats: {exc}")
                continue
            print(f"[{exam_id}] computed merge stats")
        if rows:
            args.stats_csv.parent.mkdir(parents=True, exist_ok=True)
            with args.stats_csv.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"Wrote {args.stats_csv} ({len(rows)} exams)")
        return

    if args.interactive_html:
        for exam_id in exam_ids:
            if not is_merge_complete(args.output_root, exam_id):
                print(f"[{exam_id}] not merged yet (or incomplete), skipping")
                continue
            out_path = (
                args.output_root
                / exam_id
                / "merged_tc4d"
                / f"{exam_id}_merged_skeleton_interactive.html"
            )
            try:
                write_merge_interactive_html(
                    args.preprocessing_root, args.output_root, exam_id, out_path
                )
                print(f"[{exam_id}] wrote {out_path}")
            except Exception as exc:  # noqa: BLE001 - one bad exam shouldn't kill a full-cohort batch
                print(f"[{exam_id}] could not write interactive HTML: {exc}")
        return

    if args.mip_only:
        for exam_id in exam_ids:
            if not is_merge_complete(args.output_root, exam_id):
                print(f"[{exam_id}] not merged yet (or incomplete), skipping")
                continue
            try:
                mip_path = save_merge_overlay_mip(args.output_root, exam_id)
                print(f"[{exam_id}] wrote {mip_path}")
            except Exception as exc:  # noqa: BLE001 - one bad exam shouldn't kill a full-cohort batch
                print(f"[{exam_id}] could not write merge MIP: {exc}")
        return

    skipped: list[dict[str, str]] = []
    for exam_id in exam_ids:
        merge_dir_existing = args.output_root / exam_id / "merged_tc4d"
        if merge_dir_existing.exists():
            if is_merge_complete(args.output_root, exam_id):
                print(f"[{exam_id}] already merged, skipping")
                continue
            print(
                f"[{exam_id}] found incomplete merge output (likely killed mid-run), re-running"
            )
            shutil.rmtree(merge_dir_existing)
        print(f"[{exam_id}] merging HR-mapped + UFAST-direct skeletons...")
        try:
            merge_dir = merge_exam(
                args.preprocessing_root,
                args.output_root,
                exam_id,
                dilation_radius_voxels=args.dilation_radius_voxels,
                vessel_like_min_voxels=args.vessel_like_min_voxels,
                enable_kinetics_filter=not args.disable_kinetics_filter,
                kinetics_min_peak_snr=args.kinetics_min_peak_snr,
                kinetics_adjacency_radius_voxels=args.kinetics_adjacency_radius_voxels,
            )
            save_merge_overlay_mip(args.output_root, exam_id)
        except Exception as exc:  # noqa: BLE001 - log and continue a long unattended batch job
            skipped.append({"exam_id": exam_id, "reason": str(exc)})
            print(f"[{exam_id}] FAILED: {exc}")
            continue
        print(f"[{exam_id}] done -> {merge_dir}")

    if skipped:
        import csv

        skipped_path = args.output_root / "merge_skipped_exams.csv"
        with skipped_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["exam_id", "reason"])
            writer.writeheader()
            writer.writerows(skipped)
        print(f"Wrote {skipped_path} ({len(skipped)} exams skipped)")


if __name__ == "__main__":
    main()
