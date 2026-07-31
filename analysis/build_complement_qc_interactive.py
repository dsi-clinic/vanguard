r"""Self-contained Plotly 3-D HTML for inspecting complement additions in context.

A maximum-intensity projection flattens one axis away, which is exactly the axis you
need to judge whether an addition lies along the vessel tree or floats beside it. This
builds a rotatable 3-D view of the same three populations the QC figures summarise:

  grey  -- the production merged skeleton (what was already known)
  red   -- voxels the MATLAB complement stage added (the population under judgement)
  black -- voxels that passed the SAME bar on the no-contrast negative control, i.e.
           structures the threshold admits that cannot possibly be vessels. An empty
           black trace is the result you want; anything visible there is the shape of
           this stage's false positives, and worth looking at directly.

Reads the ``*_added_zyx.npy`` / ``*_merged_zyx.npy`` / ``*_null_passing_zyx.npy`` masks
written by ``analysis.qc_complement_vessels``, so it costs no extra SegVessel passes.

Usage::

    micromamba run -n vanguard python -m analysis.build_complement_qc_interactive \
        --qc-dir /gpfs/data/.../qc_complement \
        --exam-id <exam>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

#: Matches the point cap the other interactive viewers in this repo use.
_POINT_CAP = 80000

_TRACES = (
    ("merged skeleton (known)", "_merged_zyx.npy", "#b0b0b0", 1.4, 0.45),
    ("complement added", "_added_zyx.npy", "#d1495b", 3.2, 1.0),
    ("passed on no-contrast control", "_null_passing_zyx.npy", "#111111", 3.2, 1.0),
)


def _points(mask: np.ndarray, seed: int) -> np.ndarray:
    idx = np.argwhere(mask)
    if len(idx) > _POINT_CAP:
        idx = idx[
            np.random.default_rng(seed).choice(len(idx), _POINT_CAP, replace=False)
        ]
    return idx


def _axis(title: str) -> dict:
    return {
        "title": title,
        "showgrid": False,
        "zeroline": False,
        "showbackground": False,
    }


def build(qc_dir: Path, exam_id: str) -> Path:
    """Write one exam's interactive QC page and return its path."""
    figure = go.Figure()
    counts = []
    for index, (label, suffix, colour, size, opacity) in enumerate(_TRACES):
        path = qc_dir / f"{exam_id}{suffix}"
        if not path.exists():
            raise FileNotFoundError(
                f"missing {path}; run analysis.qc_complement_vessels first"
            )
        mask = np.load(path).astype(bool)
        counts.append((label, int(mask.sum())))
        points = _points(mask, seed=index)
        if not len(points):
            continue
        figure.add_trace(
            go.Scatter3d(
                x=points[:, 2],
                y=points[:, 1],
                z=points[:, 0],
                mode="markers",
                marker={"size": size, "color": colour, "opacity": opacity},
                name=f"{label} ({int(mask.sum())})",
            )
        )

    subtitle = "  |  ".join(f"{label}: {count}" for label, count in counts)
    figure.update_layout(
        title={"text": f"Complement QC — {exam_id[:48]}<br><sub>{subtitle}</sub>"},
        scene={
            "xaxis": _axis("x"),
            "yaxis": _axis("y"),
            "zaxis": _axis("slice"),
            "aspectmode": "data",
        },
        template="plotly_white",
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "t": 70, "b": 0},
    )

    out = qc_dir / f"{exam_id}_complement_qc_interactive.html"
    figure.write_html(str(out), include_plotlyjs=True, full_html=True)
    return out


def main() -> None:
    """Build one interactive QC page per requested exam."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qc-dir", type=Path, required=True)
    parser.add_argument("--exam-id", action="append", required=True)
    args = parser.parse_args()

    for exam_id in args.exam_id:
        try:
            print(f"wrote {build(args.qc_dir, exam_id)}")
        except FileNotFoundError as exc:
            print(f"skipped {exam_id[:40]}: {exc}")


if __name__ == "__main__":
    main()
