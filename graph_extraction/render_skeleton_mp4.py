"""
Render a rotating 3D mp4 of a skeleton_4d_exam_mask.npy file.
Output is written next to the input file.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def render(skeleton_path: Path, n_frames: int = 90, dpi: int = 120) -> Path:
    sk = np.load(skeleton_path)
    # Axes match preprocessed image: d0=left-right, d1=ant-post (breast depth), d2=sup-inf
    d0, d1, d2 = np.nonzero(sk)
    if len(d0) == 0:
        raise ValueError(f"Skeleton is empty: {skeleton_path}")

    def norm(arr):
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / max(hi - lo, 1)

    # matplotlib x=breast-depth, y=left-right, z=sup-inf (vertical)
    mx, my, mz = norm(d1), norm(d0), norm(d2)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(mx, my, mz, s=1, c=mz, cmap="Blues", alpha=0.6)
    ax.set_axis_off()
    case_id = skeleton_path.parent.name
    ax.set_title(case_id, fontsize=10, pad=2)

    def update(frame):
        ax.view_init(elev=20, azim=frame * 360 / n_frames)
        return (fig,)

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=50, blit=False
    )

    out_path = skeleton_path.parent / f"{case_id}_skeleton_rotating.mp4"
    writer = animation.FFMpegWriter(fps=20, bitrate=800)
    ani.save(str(out_path), writer=writer, dpi=dpi)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("skeleton_path", type=Path)
    parser.add_argument("--n-frames", type=int, default=60)
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()

    if not args.skeleton_path.exists():
        print(f"ERROR: file not found: {args.skeleton_path}", file=sys.stderr)
        sys.exit(1)

    out = render(args.skeleton_path, n_frames=args.n_frames, dpi=args.dpi)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
