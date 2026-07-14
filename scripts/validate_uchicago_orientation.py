#!/usr/bin/env python3
"""Decide the UChicago preprocessing transform by scoring it against ground truth.

Three candidate ``UChicagoDataset.preprocess`` transforms have been proposed, and
**the tiling-coverage crash cannot distinguish two of them** -- ``order_only`` and
``mirrored`` produce identically-shaped arrays, so both make inference run. They
differ only in *anatomy*: ``mirrored`` is ``order_only`` with all three axes
flipped, a reflection (determinant -1), so it swaps the patient's left and right.
Near-symmetric breast anatomy means a MIP can't tell them apart either.

    passthrough  identity                      (the original assumption; crashes
                                               the vessel stage -- thin slice axis
                                               lands in a model in-plane dim)
    mirrored     base MAMA-MIA transform on
                 volume[::-1, :, ::-1]         (commit e341d0e; det -1, MIRRORED)
    order_only   np.transpose(v, (1, 2, 0))    (commit a8f8bcb; det +1, preserves
                                               anatomical directions)

What settles it: the breast U-Net was trained on non-mirrored anatomy, so it is
not mirror-equivariant. Feed it a mirrored volume and its predicted breast mask
degrades. We have independent ground truth to score against -- the BreastDivider
whole-breast masks Anna supplied for this same manifest -- so for each transform
we run the pinned breast model on a real phase and compute Dice against the
ground-truth mask carried through the *same* geometric transform.

The transform whose Dice is highest is the one that presents the model with the
anatomy it was trained on. Prediction under Anna's argument (HFDP already
canonicalized the anatomical directions, so only array *order* needs fixing):
``order_only`` wins, and ``mirrored`` is materially worse.

Also reported, as a direct read on the mirror: the BreastDivider masks label
``1=left / 2=right``, so after each transform we report which side of the array
the patient's left breast lands on. ``mirrored`` and ``order_only`` must disagree
here -- that is the whole question, made concrete.

Run (GPU needed for the breast model):
    PROJECT_ROOT="$(pwd)" sbatch segmentation/slurm/submit_uchicago_orientation_check.slurm
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "segmentation"))
sys.path.insert(0, str(_PROJECT_ROOT))

import batch_segmentation as bs  # noqa: E402
import external_breast_masks as ebm  # noqa: E402

from cohorts.base import DatasetAdapter  # noqa: E402
from cohorts.uchicago import UChicagoDataset  # noqa: E402

DEFAULT_ROOT = "/gpfs/data/karczmar-lab/vanguard/dce2d_internal_ultrafast_manifest"


class _PassthroughUChicago(UChicagoDataset):
    """The original assumption: HFDP output is already in the model's layout."""

    def preprocess(self, volume: np.ndarray) -> np.ndarray:
        return volume


class _MirroredUChicago(UChicagoDataset):
    """Commit e341d0e: header-derived RAS->LAI flip, then the base transform."""

    def preprocess(self, volume: np.ndarray) -> np.ndarray:
        return DatasetAdapter.preprocess(self, volume[::-1, :, ::-1])


VARIANTS: dict[str, type[UChicagoDataset]] = {
    "passthrough": _PassthroughUChicago,
    "mirrored": _MirroredUChicago,
    "order_only": UChicagoDataset,  # current HEAD (a8f8bcb)
}

# Laterality is read off array axis 1, which is the left-right axis only once the
# volume is in the model's (rows, cols, slices) layout. Passthrough leaves it in
# SimpleITK's (slices, rows, cols), so axis 1 is rows there -- reading a left/right
# centroid off it would be meaningless, not merely wrong.
LATERALITY_READABLE = {"mirrored", "order_only"}


MASK_THRESHOLD = 0.5
LABEL_LEFT, LABEL_RIGHT = 1, 2  # BreastDivider mask labels (0 = background)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice coefficient between two binary masks."""
    a_bin, b_bin = a > MASK_THRESHOLD, b > MASK_THRESHOLD
    denom = a_bin.sum() + b_bin.sum()
    if denom == 0:
        return float("nan")
    return float(2.0 * np.logical_and(a_bin, b_bin).sum() / denom)


def left_breast_side(labelled: np.ndarray) -> str:
    """Report which half of the array's column axis the patient's LEFT breast occupies.

    ``labelled`` is a BreastDivider mask (1=left, 2=right) already carried through
    a candidate transform. Preprocessed volumes are (rows, cols, slices), so the
    column axis (1) is the left-right image axis. A transform that mirrors anatomy
    puts label 1 on the opposite side from one that doesn't -- which is exactly the
    disagreement we are trying to resolve.
    """
    ncols = labelled.shape[1]
    left_cols = np.where(labelled == LABEL_LEFT)[1]
    right_cols = np.where(labelled == LABEL_RIGHT)[1]
    if left_cols.size == 0 or right_cols.size == 0:
        return "n/a (mask has no L/R split)"
    left_c, right_c = left_cols.mean(), right_cols.mean()
    side = "LOW column index" if left_c < right_c else "HIGH column index"
    return f"{side} (left centroid col {left_c:.0f}, right {right_c:.0f}, of {ncols})"


def pick_cases(root: Path, explicit: list[str] | None) -> list[str]:
    """One exam per manifest sub-source, unless explicit ids are given."""
    import pandas as pd

    manifest = pd.read_csv(root / "dce2d_internal_ultrafast_manifest.csv")
    if explicit:
        missing = set(explicit) - set(manifest["exam_id"])
        if missing:
            raise SystemExit(f"unknown exam ids: {sorted(missing)}")
        return explicit
    return [g.iloc[0]["exam_id"] for _, g in manifest.groupby("dataset", sort=True)]


def phase0_path(root: Path, adapter: UChicagoDataset, case_id: str) -> Path:
    """Absolute path to a case's phase_0000 file, via the adapter's own timepoints."""
    timepoints = adapter.load_timepoints(case_id)
    return Path(timepoints[0])


def main() -> None:
    """Score each candidate transform against the BreastDivider ground truth."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", default=DEFAULT_ROOT)
    p.add_argument("--breast-model-path", required=True)
    p.add_argument("--mask-manifest", default=ebm.DEFAULT_MANIFEST_CSV)
    p.add_argument("--work-dir", required=True, help="scratch dir for step1/step2")
    p.add_argument("--out-csv", required=True)
    p.add_argument("--case-ids", nargs="*", default=None)
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()

    root = Path(args.dataset_root)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ref_adapter = UChicagoDataset(root=root)
    cases = pick_cases(root, args.case_ids)
    print(f"cases ({len(cases)}), one per sub-source unless overridden:")
    for c in cases:
        print(f"  {ref_adapter.case_dataset_name(c):14s} {c}")

    mask_manifest = ebm.load_manifest(args.mask_manifest)
    ebm.validate_masks_available(cases, mask_manifest)  # fail-closed, before GPU work

    rows: list[dict[str, object]] = []
    for name, cls in VARIANTS.items():
        print(f"\n=== variant: {name} ===")
        adapter = cls(root=root)
        step1, step2, extract = (work / name / d for d in ("step1", "step2", "masks"))
        for d in (step1, step2, extract):
            d.mkdir(parents=True, exist_ok=True)

        for case_id in cases:
            src = phase0_path(root, ref_adapter, case_id)
            ok = bs.preprocess_image(
                str(src), str(step1 / f"{case_id}_phase_0000.npy"), adapter
            )
            if not ok:
                raise SystemExit(f"{name}: preprocessing failed for {case_id}")

        bs._run_breast_inference(
            step1, step2, args.breast_model_path, args.batch_size, 1, True, device
        )

        for case_id in cases:
            pred = np.load(step2 / f"{case_id}_phase_0000.npy").astype(np.float32)
            gt = ebm.build_whole_breast_mask(case_id, mask_manifest, extract, adapter)
            gt = np.asarray(gt, dtype=np.float32)
            if gt.shape != pred.shape:
                raise SystemExit(
                    f"{name}/{case_id}: gt {gt.shape} != pred {pred.shape}; "
                    "the mask no longer shares the image grid -- investigate before trusting."
                )

            # laterality, straight off the labelled (un-binarized) ground truth
            archive, member = ebm._resolve_mask_member(case_id, mask_manifest)
            import tarfile

            with tarfile.open(archive) as tf:
                tf.extract(member, path=extract)
            labelled = sitk.GetArrayFromImage(sitk.ReadImage(str(extract / member)))
            labelled_t = adapter.preprocess(labelled.astype(np.float32))

            d = dice(pred, gt)
            side = (
                left_breast_side(labelled_t)
                if name in LATERALITY_READABLE
                else "n/a (not in (rows, cols, slices) layout; axis 1 is not left-right)"
            )
            print(f"  {case_id[:28]:30s} dice={d:.3f}  left breast -> {side}")
            rows.append(
                {
                    "variant": name,
                    "case_id": case_id,
                    "sub_source": ref_adapter.case_dataset_name(case_id),
                    "breast_dice_vs_groundtruth": round(d, 4),
                    "pred_voxels": int((pred > MASK_THRESHOLD).sum()),
                    "gt_voxels": int((gt > MASK_THRESHOLD).sum()),
                    "left_breast_side": side,
                }
            )

    with Path(args.out_csv).open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(
        "\n=== mean breast Dice vs BreastDivider ground truth (higher = correct anatomy) ==="
    )
    summary = {}
    for name in VARIANTS:
        vals = [r["breast_dice_vs_groundtruth"] for r in rows if r["variant"] == name]
        summary[name] = float(np.nanmean(vals))
        print(f"  {name:14s} {summary[name]:.3f}")
    best = max(summary, key=summary.get)
    print(f"\nbest: {best}")
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
