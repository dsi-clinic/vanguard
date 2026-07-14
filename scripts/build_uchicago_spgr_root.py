#!/usr/bin/env python3
"""Materialize a UChicago dataset root from the RECOMMENDED motion-corrected data.

We have been segmenting the **legacy** image contract
(``dce2d_internal_ultrafast_manifest.csv``), which the dataset's own README says to
use "only when reproducing the legacy clipped, robust-z-score image baseline". Two
consequences, both bad for vessel segmentation:

1. **Double normalization.** The legacy phase files are already clipped and robust
   z-scored (verified: min -3.14, mean 2.80 -- raw MRI signal is nonnegative), and
   then ``segmentation/batch_segmentation.py:preprocess_image`` applies
   ``zscore_image(normalize_image(...))`` on top. Normalizing twice can only
   compress vessel-to-background contrast, which is the signal the model keys on.
2. **No motion correction.** Our runs have none; a QC-gated motion-corrected export
   exists, and it also makes the single phase-0 whole-breast mask valid for every
   phase (all frames are registered to phase 0), retiring the "one mask reused
   across phases" approximation.

This script writes a new root in the layout ``UChicagoDataset`` already expects --
``images/<dataset>/<exam_id>/phase_XXXX.nii.gz`` plus a manifest CSV with the legacy
schema -- so the adapter can be pointed at it via ``dataset_root`` with **zero code
change**. That is exactly what the run-config-injected root is for (cohorts/README.md).

Sources (per the dataset README, "Which Image Contract To Use"):
    manifest : spgr_safe_motion_corrected/manifest_recommended_180.csv
    images   : spgr_safe_motion_corrected/shards/<NNNN>_<exam_id>/phase_images.tar
               (raw signal, float32, nonnegative, 1 mm isotropic, every acquired frame)

Run:
    python scripts/build_uchicago_spgr_root.py --out-root <dir> --case-ids <file>
"""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from pathlib import Path

import pandas as pd

SRC = Path("/gpfs/data/karczmar-lab/vanguard/dce2d_internal_ultrafast_manifest")
MC = SRC / "spgr_safe_motion_corrected"
RECOMMENDED = MC / "manifest_recommended_180.csv"
LEGACY = SRC / "dce2d_internal_ultrafast_manifest.csv"
MANIFEST_NAME = (
    "dce2d_internal_ultrafast_manifest.csv"  # cohorts.uchicago.DEFAULT_MANIFEST_NAME
)


def parse_case_ids(value: str | None) -> list[str] | None:
    """Resolve a comma-separated list or a file of ids (one per line, # = comment)."""
    if value is None:
        return None
    path = Path(value)
    raw = path.read_text().splitlines() if path.is_file() else value.split(",")
    ids = [s.strip() for s in raw]
    ids = [s for s in ids if s and not s.startswith("#")]
    if not ids:
        raise SystemExit(f"--case-ids resolved to an empty selection: {value!r}")
    return ids


def shard_for(exam_id: str) -> Path:
    """Locate the per-exam shard directory (prefixed with its row index)."""
    hits = sorted(MC.glob(f"shards/*_{exam_id}"))
    if len(hits) != 1:
        raise SystemExit(f"expected exactly 1 shard for {exam_id}, found {len(hits)}")
    return hits[0]


def main() -> None:
    """Extract the recommended motion-corrected frames into an adapter-ready root."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root", required=True)
    p.add_argument(
        "--case-ids", default=None, help="file or comma list; default = all 180"
    )
    args = p.parse_args()

    out = Path(args.out_root)
    (out / "images").mkdir(parents=True, exist_ok=True)

    rec = pd.read_csv(RECOMMENDED)
    legacy = pd.read_csv(LEGACY)

    # Labels/folds/patient keys are identical across contracts; take them from the
    # legacy manifest so the adapter's schema (and our folds) are unchanged.
    keep = ["exam_id", "dataset", "patient_key", "fold", "pcr"]
    meta = legacy[keep].copy()

    wanted = parse_case_ids(args.case_ids) or rec["exam_id"].tolist()
    missing = set(wanted) - set(rec["exam_id"])
    if missing:
        raise SystemExit(
            f"not in the recommended manifest (180 rows): {sorted(missing)}"
        )

    rows = []
    for exam_id in wanted:
        r = rec[rec.exam_id == exam_id].iloc[0]
        dataset = r["dataset"]
        dest = out / "images" / dataset / exam_id
        dest.mkdir(parents=True, exist_ok=True)

        shard = shard_for(exam_id)
        with tarfile.open(shard / "phase_images.tar") as tf:
            members = [m for m in tf.getmembers() if m.name.endswith(".nii.gz")]
            members.sort(key=lambda m: m.name)
            for m in members:
                target = dest / Path(m.name).name  # phase_XXXX.nii.gz
                if target.exists():
                    continue
                src = tf.extractfile(m)
                if src is None:
                    raise SystemExit(f"{exam_id}: could not read {m.name}")
                with target.open("wb") as fh:
                    shutil.copyfileobj(src, fh)

        phases = sorted(dest.glob("phase_*.nii.gz"))
        if not phases:
            raise SystemExit(f"{exam_id}: no phases extracted")

        mrow = meta[meta.exam_id == exam_id]
        if mrow.empty:
            raise SystemExit(
                f"{exam_id}: absent from the legacy manifest (no label/fold)"
            )
        row = mrow.iloc[0].to_dict()
        row["phase_files"] = json.dumps([str(p) for p in phases])
        row["n_phases"] = len(phases)
        row["baseline_frame_count"] = int(r.get("baseline_frame_count", 5))
        row["times_seconds_json"] = r.get("times_seconds_json")
        row["motion_correction_applied"] = True
        row["preprocessing_policy"] = r.get("preprocessing_policy")
        rows.append(row)
        print(f"  {dataset:14s} {exam_id[:26]}  {len(phases):3d} phases")

    manifest_out = pd.DataFrame(rows)
    manifest_out.to_csv(out / MANIFEST_NAME, index=False)
    print(
        f"\nwrote {out / MANIFEST_NAME}  ({len(manifest_out)} exams, {manifest_out.n_phases.sum()} frames)"
    )
    print(f"point the adapter at:  dataset_root = {out}")


if __name__ == "__main__":
    main()
