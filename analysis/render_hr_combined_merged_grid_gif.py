r"""Exploratory: one rotating-GIF grid comparing HR-only, combined, and merged skeletons.

Renders a ``len(exam_ids) x 3`` grid animated GIF: each row is one exam, and
the three columns are the exam's HR-only skeleton (``preprocessing.pipeline.map_case``'s
output, the currently-supported route), the joint HR+UFAST TC4D skeleton
(``analysis.skeletonize_combined_hr_ufast``), and the post-hoc merged skeleton
(``analysis.merge_ufast_hr_skeletons``). All three are already on the same
physical UFAST grid (no resampling needed here -- same precondition
``analysis.skeletonize_ufast_vessels.render_skeleton_comparison_mp4`` relies
on), so within a row they're directly comparable at the same physical scale.

Every cell in the grid rotates through the same camera azimuth in lock-step
(one shared frame sequence for the whole grid), so scrubbing through the GIF
compares all three routes -- across all exams -- from the same viewing angle
at every instant.

Column color convention (dark theme, matches
``analysis.merge_ufast_hr_skeletons``'s green/magenta/white/cyan scheme):
  - HR-only: green (#2ecc40) -- the currently-supported route.
  - Combined (joint TC4D): orange (#ff8c1a) -- new exploratory route.
  - Merged (post-hoc): white (#ffffff) -- existing exploratory route.

Usage::

    micromamba run -n vanguard python -m analysis.render_hr_combined_merged_grid_gif \\
        --preprocessing-root /gpfs/data/karczmar-lab/workspaces/saritbose/preprocessing_out \\
        --output-root /gpfs/data/karczmar-lab/workspaces/saritbose/ufast-seg \\
        --exam-id exam1 --exam-id exam2 --exam-id exam3 --exam-id exam4 \\
        --out-path /gpfs/data/karczmar-lab/workspaces/saritbose/ufast-seg/hr_combined_merged_grid.gif
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.skeletonize_ufast_vessels import find_hr_mapped_skeleton  # noqa: E402
from preprocessing.dicom import DicomGeometry  # noqa: E402
from preprocessing.spatial import physical_points_from_zyx  # noqa: E402

#: (label, color) per column, in grid order.
COLUMNS: tuple[tuple[str, str], ...] = (
    ("HR-only (supported)", "#2ecc40"),
    ("Combined (joint TC4D)", "#ff8c1a"),
    ("Merged (post-hoc)", "#ffffff"),
)

#: Point cap per cell -- a grid has 3x more subplots per frame than a single
#: rotating render, so this stays well under
#: ``analysis.skeletonize_ufast_vessels.render_skeleton_comparison_mp4``'s
#: 40000-point default to keep per-frame render time reasonable.
DEFAULT_POINT_CAP = 15000


def _mpl_axis_no_grid(ax: object) -> None:
    """Turn off matplotlib 3D major/minor gridlines; keep tick labels."""
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"].update({"linewidth": 0.0, "color": (0, 0, 0, 0)})
        try:
            axis.pane.fill = False
            axis.pane.set_edgecolor((0.25, 0.25, 0.25, 0.4))
        except AttributeError:
            pass


def load_exam_skeletons(
    preprocessing_root: Path, output_root: Path, exam_id: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, DicomGeometry]:
    """Load (hr_skeleton, combined_skeleton, merged_skeleton) as (z,y,x) bool, plus UFAST geometry."""
    hr_skeleton = np.load(find_hr_mapped_skeleton(preprocessing_root, exam_id)).astype(
        bool
    )
    combined_skeleton = np.load(
        output_root / exam_id / "combined_tc4d" / "exam_skeleton_zyx.npy"
    ).astype(bool)
    merged_skeleton = np.load(
        output_root / exam_id / "merged_tc4d" / "merged_skeleton_zyx.npy"
    ).astype(bool)
    shapes = {
        "hr": hr_skeleton.shape,
        "combined": combined_skeleton.shape,
        "merged": merged_skeleton.shape,
    }
    if len(set(shapes.values())) != 1:
        raise ValueError(f"case={exam_id}: shape mismatch across skeletons: {shapes}")

    provenance = json.loads(
        (
            preprocessing_root / "work" / exam_id / "preprocessing_provenance.json"
        ).read_text()
    )
    ufast_geometry = DicomGeometry.from_dict(provenance["ufast_output_geometry"])
    return hr_skeleton, combined_skeleton, merged_skeleton, ufast_geometry


def _pick_points(
    mask_zyx: np.ndarray, geometry: DicomGeometry, cap: int, seed: int
) -> np.ndarray:
    idx = np.argwhere(mask_zyx)
    if len(idx) > cap:
        idx = idx[np.random.default_rng(seed).choice(len(idx), cap, replace=False)]
    return physical_points_from_zyx(idx, geometry) if len(idx) else np.empty((0, 3))


#: Exam ids longer than this get truncated with an ellipsis in figure labels.
_SHORT_ID_MAX_LEN = 34


def _short_id(exam_id: str) -> str:
    if len(exam_id) <= _SHORT_ID_MAX_LEN:
        return exam_id
    return exam_id[:16] + "…" + exam_id[-14:]


def render_grid_gif(
    preprocessing_root: Path,
    output_root: Path,
    exam_ids: list[str],
    out_path: Path,
    *,
    n_frames: int = 36,
    fps: int = 10,
    point_cap: int = DEFAULT_POINT_CAP,
) -> Path:
    """Render the exam_ids x 3-column rotating comparison grid to an animated GIF."""
    import imageio.v2 as imageio

    n_rows = len(exam_ids)
    rows_data: list[dict[str, object]] = []
    for row_index, exam_id in enumerate(exam_ids):
        hr_sk, combined_sk, merged_sk, geometry = load_exam_skeletons(
            preprocessing_root, output_root, exam_id
        )
        points_by_column = [
            _pick_points(hr_sk, geometry, point_cap, seed=row_index * 3 + 0),
            _pick_points(combined_sk, geometry, point_cap, seed=row_index * 3 + 1),
            _pick_points(merged_sk, geometry, point_cap, seed=row_index * 3 + 2),
        ]
        non_empty = [p for p in points_by_column if len(p)]
        if not non_empty:
            raise ValueError(
                f"case={exam_id}: all three skeletons are empty, nothing to render"
            )
        all_pts = np.vstack(non_empty)
        lo = all_pts.min(axis=0)
        hi = all_pts.max(axis=0)
        pad = 0.05 * (hi - lo + 1e-6)
        rows_data.append(
            {
                "exam_id": exam_id,
                "points_by_column": points_by_column,
                "lo": lo - pad,
                "hi": hi + pad,
            }
        )

    azimuths = np.linspace(0.0, 360.0, int(max(n_frames, 2)), endpoint=False)
    frames_rgb: list[np.ndarray] = []
    for frame_index, azim in enumerate(azimuths):
        fig, axes = plt.subplots(
            n_rows,
            3,
            figsize=(4.2 * 3, 4.2 * n_rows),
            facecolor="#0d0d0d",
            subplot_kw={"projection": "3d"},
        )
        if n_rows == 1:
            axes = axes[np.newaxis, :]
        for row_index, row in enumerate(rows_data):
            lo, hi = row["lo"], row["hi"]
            points_by_column = row["points_by_column"]
            for col_index, (col_label, color) in enumerate(COLUMNS):
                ax = axes[row_index, col_index]
                ax.set_facecolor("#0d0d0d")
                ax.set_xlim(lo[0], hi[0])
                ax.set_ylim(lo[1], hi[1])
                ax.set_zlim(lo[2], hi[2])
                ax.view_init(elev=15, azim=float(azim))
                ax.tick_params(colors="#777777", labelsize=6)
                _mpl_axis_no_grid(ax)
                pts = points_by_column[col_index]
                if len(pts):
                    ax.scatter(
                        pts[:, 0],
                        pts[:, 1],
                        pts[:, 2],
                        c=color,
                        s=1.6,
                        alpha=0.85,
                        linewidths=0,
                    )
                if row_index == 0:
                    ax.set_title(col_label, color="#eeeeee", fontsize=10, pad=6)
                if col_index == 0:
                    ax.text2D(
                        -0.12,
                        0.5,
                        _short_id(row["exam_id"]),
                        transform=ax.transAxes,
                        color="#eeeeee",
                        fontsize=8,
                        rotation=90,
                        va="center",
                        ha="center",
                    )
        fig.suptitle(
            "HR-only (green) vs Combined joint TC4D (orange) vs Merged post-hoc (white)",
            color="#cccccc",
            fontsize=11,
            y=0.995,
        )
        fig.tight_layout(rect=(0.02, 0.0, 1.0, 0.97))

        buf = io.BytesIO()
        # No bbox_inches="tight": crops content-dependent, which shifts frame
        # pixel dimensions as azimuth/labels change -- imageio's GIF writer
        # rejects frames of different sizes.
        fig.savefig(buf, format="png", dpi=90, facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        frames_rgb.append(imageio.imread(buf))
        print(f"rendered frame {frame_index + 1}/{len(azimuths)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out_path), frames_rgb, fps=int(fps))
    return out_path


def main() -> None:
    """CLI entrypoint: render the HR-only/combined/merged rotating comparison grid GIF."""
    p = argparse.ArgumentParser(
        description=(
            "Exploratory: render a rotating GIF grid (rows=exams, "
            "columns=HR-only/combined/merged skeleton) for a small set of exams."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--preprocessing-root", required=True, type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--exam-id", action="append", dest="exam_ids", required=True)
    p.add_argument("--out-path", required=True, type=Path)
    p.add_argument("--n-frames", type=int, default=36)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--point-cap", type=int, default=DEFAULT_POINT_CAP)
    args = p.parse_args()

    out_path = render_grid_gif(
        args.preprocessing_root,
        args.output_root,
        args.exam_ids,
        args.out_path,
        n_frames=args.n_frames,
        fps=args.fps,
        point_cap=args.point_cap,
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
