"""Export the UChicago manifest's labels + provided folds to a GNN labels CSV.

The GNN loader reads pCR labels and the ``fold`` column from a CSV
(``data_paths.gnn_labels_path``), not from the manifest directly. UChicago ships
its own patient-grouped, pcr-stratified folds in the manifest, so this exporter
does **not** recompute folds (unlike ``gnn.make_fold_assignments``, which freezes
a fresh StratifiedKFold split for MAMA-MIA). It simply materializes the adapter's
``load_labels()`` + ``load_folds()`` to a ``case_id,pcr,fold`` CSV so
``split_mode: predefined`` consumes the shipped assignment.

The output carries the pcr label, so (like other labels files) it is written to
a workspace/scratch path and is not committed; this script is committed so the
file can be regenerated. See ``configs/uchicago_gnn.yaml``.

Example::

    python -m gnn.export_uchicago_labels \
        --dataset-root /gpfs/data/karczmar-lab/vanguard/dce2d_internal_ultrafast_manifest \
        --out /gpfs/data/karczmar-lab/workspaces/spencervenancio/uchicago/uchicago_labels_folded.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from cohorts.uchicago import UChicagoDataset


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="UChicago manifest directory (the adapter's root).",
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=None,
        help="Manifest CSV; defaults to the adapter's default under --dataset-root.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output CSV path (case_id,pcr,fold). Written to a workspace/scratch "
        "path since it carries the pcr label.",
    )
    return parser


def main() -> None:
    """Merge the adapter's labels and provided folds, and write the CSV."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = build_parser().parse_args()

    adapter = UChicagoDataset(root=args.dataset_root, manifest_csv=args.manifest_csv)
    labels = adapter.load_labels()  # case_id, pcr
    folds = adapter.load_folds()  # case_id, fold

    merged = labels.merge(folds, on="case_id", how="inner", validate="one_to_one")
    if len(merged) != len(labels):
        dropped = set(labels["case_id"]) - set(merged["case_id"])
        raise ValueError(
            f"{len(dropped)} labeled case(s) have no fold assignment: "
            f"{sorted(dropped)[:10]}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    merged = merged.sort_values("case_id").reset_index(drop=True)
    merged.to_csv(args.out, index=False)
    logging.info(
        "Wrote %d labeled case(s) with folds to %s (pcr: %s)",
        len(merged),
        args.out,
        dict(merged["pcr"].value_counts().sort_index()),
    )


if __name__ == "__main__":
    main()
