#!/usr/bin/env python3
"""Render shared-window QC panels for a prepared UChicago tumor run.

Aakrithi Ram implemented this renderer for the original 30-case validation
pilot. It remains compatible with that run while accepting any prepared tumor
cohort root explicitly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd

from preprocessing.dicom import load_dicom_series


DEFAULT_OUT_DIR = Path("qc/uchic_tumor_masks_contract_30")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _load(path: Path) -> np.ndarray:
    return np.asanyarray(nib.load(path).dataobj)


def _mask(path: Path) -> np.ndarray:
    return _load(path) > 0


def _display(arr: np.ndarray) -> np.ndarray:
    return np.rot90(arr)


def _shared_window(paths: list[Path]) -> tuple[float, float]:
    samples = []
    for path in paths:
        arr = np.ravel(_load(path).astype(np.float32))
        arr = arr[np.isfinite(arr)]
        if arr.size:
            samples.append(arr[:: max(1, arr.size // 100_000)])
    vals = np.concatenate(samples)
    vals = vals[vals > 0]
    lo, hi = np.percentile(vals, [0.5, 99.5])
    if not hi > lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def _best_slice(*masks: np.ndarray) -> int:
    counts = None
    for mask in masks:
        z_counts = np.asarray(mask, dtype=bool).reshape(-1, mask.shape[2]).sum(axis=0)
        counts = z_counts if counts is None else counts + z_counts
    if counts is not None and counts.max() > 0:
        return int(np.argmax(counts))
    return int(masks[0].shape[2] // 2)


def _overlay(ax: plt.Axes, image: np.ndarray, mask: np.ndarray, *, vmin: float, vmax: float, color: str) -> None:
    ax.imshow(_display(image), cmap="gray", vmin=vmin, vmax=vmax)
    shown = np.ma.masked_where(~mask.astype(bool), mask.astype(bool))
    ax.imshow(_display(shown), cmap=color, alpha=0.45, vmin=0, vmax=1)
    ax.axis("off")


def _sub_limit(first: np.ndarray, images: list[np.ndarray]) -> float:
    limits = []
    for image in images:
        sub = image.astype(np.float32) - first.astype(np.float32)
        limits.extend([abs(float(np.percentile(sub, 1))), abs(float(np.percentile(sub, 99)))])
    value = max(limits) if limits else 1.0
    return value if value > 0 else 1.0


def _ufast_baseline(provenance: dict[str, object]) -> np.ndarray:
    case = provenance["case"]
    ufast = load_dicom_series(
        provenance["inventory_path"],
        study_uid=case["study_instance_uid"],
        series_uid=case["ufast_series_instance_uid"],
    )
    n_baseline = int(provenance["ufast_source"]["baseline_frame_count"])
    return np.asarray(ufast.signal_tzyx[:n_baseline].mean(axis=0), dtype=np.float32)


def render_case(run_root: Path, out_dir: Path, row: pd.Series) -> dict[str, object]:
    exam_id = str(row["exam_id"])
    case_root = run_root / "work" / exam_id
    provenance = json.loads((case_root / "tumor_preprocessing_provenance.json").read_text())
    phases = pd.read_csv(case_root / "hr_phase_manifest.csv")
    image_paths = [Path(path) for path in phases["hr_image"]]
    images = [_load(path).astype(np.float32) for path in image_paths]
    vmin, vmax = _shared_window(image_paths)
    pre = images[0]
    first_post = images[1]
    sub_lim = _sub_limit(pre, images)

    all_components = _mask(case_root / "tumor_masks" / "tumor_all_components_hr.nii.gz")
    primary = _mask(case_root / "tumor_masks" / "tumor_primary_hr.nii.gz")
    later_rows = phases.loc[phases["phase_index"].gt(1)].copy()
    later_masks = []
    for _, phase in later_rows.iterrows():
        pred = Path(str(phase["prediction"]))
        if pred.exists():
            later_masks.append((int(phase["phase_index"]), float(phase["time_seconds"]), _mask(pred)))

    z = _best_slice(primary, all_components)
    n_cols = 6 + len(later_masks)
    fig, axes = plt.subplots(2, n_cols, figsize=(2.4 * n_cols, 5.2), constrained_layout=True)
    if n_cols == 1:
        axes = np.asarray(axes).reshape(2, 1)
    fig.suptitle(f"{exam_id} tumor QC, HR axial z={z}", fontsize=11)

    axes[0, 0].imshow(_display(pre[:, :, z]), cmap="gray", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("HR tp0 pre", fontsize=8)
    axes[0, 0].axis("off")

    _overlay(axes[0, 1], first_post[:, :, z], primary[:, :, z], vmin=vmin, vmax=vmax, color="autumn")
    axes[0, 1].set_title(f"HR tp1 final\n{float(phases.iloc[1]['time_seconds']):.1f}s", fontsize=8)

    sub = first_post - pre
    axes[0, 2].imshow(_display(sub[:, :, z]), cmap="seismic", vmin=-sub_lim, vmax=sub_lim)
    axes[0, 2].imshow(_display(np.ma.masked_where(~primary[:, :, z], primary[:, :, z])), cmap="autumn", alpha=0.45)
    axes[0, 2].set_title("tp1 - tp0", fontsize=8)
    axes[0, 2].axis("off")

    _overlay(axes[0, 3], first_post[:, :, z], all_components[:, :, z], vmin=vmin, vmax=vmax, color="spring")
    axes[0, 3].set_title("all tp1 comps", fontsize=8)

    _overlay(axes[0, 4], first_post[:, :, z], primary[:, :, z], vmin=vmin, vmax=vmax, color="autumn")
    axes[0, 4].set_title("primary comp", fontsize=8)

    ufast_base = _ufast_baseline(provenance)
    primary_ufast = _mask(case_root / "tumor_masks" / "tumor_primary_ufast.nii.gz")
    z_ufast = _best_slice(primary_ufast)
    uf_vals = ufast_base[ufast_base > 0]
    uf_vmin, uf_vmax = np.percentile(uf_vals, [0.5, 99.5]) if uf_vals.size else (0.0, 1.0)
    _overlay(
        axes[0, 5],
        np.transpose(ufast_base, (2, 1, 0))[:, :, z_ufast],
        primary_ufast[:, :, z_ufast],
        vmin=float(uf_vmin),
        vmax=float(uf_vmax),
        color="autumn",
    )
    axes[0, 5].set_title(f"UFAST baseline\nz={z_ufast}", fontsize=8)

    for col, (phase_index, time_seconds, mask) in enumerate(later_masks, start=6):
        image = images[phase_index]
        _overlay(axes[0, col], image[:, :, z], mask[:, :, z], vmin=vmin, vmax=vmax, color="autumn")
        axes[0, col].set_title(f"later tp{phase_index}\n{time_seconds:.1f}s", fontsize=8)

    for col in range(n_cols):
        axes[1, col].axis("off")
    axes[1, 0].text(
        0,
        0.95,
        "\n".join(
            [
                f"primary ml: {float(row['primary_component_volume_ml']):.3f}",
                f"all comps ml: {float(row['all_components_volume_ml']):.3f}",
                f"median later Dice: {row['median_first_to_later_dice']}",
                f"empty: {row['tumor_mask_empty']}",
                f"tiny: {row['review_tiny']}",
                f"temporal review: {row['review_temporal_instability']}",
                f"alignment: {row['alignment_qc_status']}",
                f"mapping: {row['mapping_status']}",
            ]
        ),
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
    )
    for col, image in enumerate(images[: min(n_cols, len(images))]):
        axes[1, col].imshow(_display(image[:, :, z]), cmap="gray", vmin=vmin, vmax=vmax)
        axes[1, col].set_title(f"natural tp{col}", fontsize=8)
        axes[1, col].axis("off")

    out_path = out_dir / f"{exam_id}_tumor_contract_qc.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return {"exam_id": exam_id, "dataset": row.get("dataset", ""), "qc_panel": str(out_path)}


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.run_root / "tumor_mask_manifest.csv")
    rows = [render_case(args.run_root, args.out_dir, row) for _, row in manifest.iterrows()]
    index = pd.DataFrame(rows)
    index.to_csv(args.out_dir / "index.csv", index=False)
    with (args.out_dir / "index.html").open("w") as stream:
        stream.write("<html><body><h1>UChicago tumor segmentation QC</h1>\n")
        for row in rows:
            name = Path(row["qc_panel"]).name
            stream.write(f"<h2>{row['exam_id']}</h2>\n")
            stream.write(f"<img src='{name}' style='max-width:100%;'>\n")
        stream.write("</body></html>\n")
    print(f"wrote {args.out_dir / 'index.html'}")
    print(f"n_panels={len(rows)}")


if __name__ == "__main__":
    main()
