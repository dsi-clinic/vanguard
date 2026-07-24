"""Freeze a shared k-fold assignment to a labels-with-fold CSV.

Writes ``<labels>`` augmented with a ``fold`` column so that every graph
representation (``node_mode`` = voxel / segment / junction) trained with
``split_mode: predefined`` consumes the *identical* per-case fold assignment.
This is what makes the voxel-vs-segment-vs-junction comparison matched: with the
default random ``StratifiedKFold`` each arm re-derives folds by row position
after a seeded shuffle, so folds only agree if the three caches have identical
membership *and* order -- which silently breaks if one representation drops a
case (e.g. a degenerate component). Freezing the split to disk and consuming it
via ``create_predefined_splits`` assigns each case its fold by ``case_id``, so
every case keeps the same fold in every arm regardless of cache membership.

The fold assignment is produced by the shared evaluation framework
(``evaluation.kfold.create_kfold_splits``, i.e. ``StratifiedKFold(shuffle=True,
random_state=...)`` stratified on the label) rather than a hand-rolled splitter,
so the frozen CSV is exactly the framework's own seeded split materialized to
disk. Cases are sorted by ``case_id`` first so the assignment is reproducible
from the CSV alone, independent of the source file's row order.

The output CSV carries the pcr label, so (like the source labels file) it is
written to scratch and is not committed; this script is committed so the file
can be regenerated. See ``gnn/DESIGN_segment_graph.md`` and the campaign README.

Example::

    python -m gnn.make_fold_assignments \
        --labels-path /gpfs/.../pcr_labels.csv \
        --out /ess/scratch/scratch1/$USER/gnn_caches/pcr_labels_folded.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.kfold import create_kfold_splits
from tabular.cohort import load_labels

_DEFAULT_LABELS_PATH = Path(
    "/gpfs/data/karczmar-lab/workspaces/saritbose/metadata/pcr_labels.csv"
)
_DEFAULT_OUT = Path("/ess/scratch/scratch1/t-9svena/gnn_caches/pcr_labels_folded.csv")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-path", type=Path, default=_DEFAULT_LABELS_PATH)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--id-column", type=str, default="case_id")
    parser.add_argument("--label-column", type=str, default="pcr")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def assign_folds(
    labels_path: Path,
    id_column: str,
    label_column: str,
    n_splits: int,
    random_state: int,
) -> pd.DataFrame:
    """Return the label frame with an added integer ``fold`` column.

    Cases are sorted by ``id_column`` before splitting so the assignment is a
    deterministic function of (case set, label, ``n_splits``, ``random_state``).
    """
    frame = load_labels(labels_path, id_column, label_column)
    frame = frame[[id_column, label_column]].sort_values(id_column)
    frame = frame.reset_index(drop=True)

    case_ids = frame[id_column].to_numpy()
    y = frame[label_column].to_numpy()
    x_dummy = np.zeros((len(frame), 1), dtype=float)

    splits = create_kfold_splits(
        X=x_dummy,
        y=y,
        case_ids=case_ids,
        n_splits=n_splits,
        stratify=True,
        shuffle=True,
        random_state=random_state,
    )

    fold_of: dict[str, int] = {}
    for split in splits:
        for case_id in split["val_case_ids"]:
            fold_of[str(case_id)] = int(split["fold_idx"])

    # Every case must land in exactly one validation fold; anything else means
    # the split framework changed under us -- surface it rather than write a
    # partial fold column (fail-fast).
    missing = [c for c in case_ids if str(c) not in fold_of]
    if missing:
        raise ValueError(
            f"{len(missing)} case(s) received no fold assignment: {missing[:5]}"
        )

    frame["fold"] = [fold_of[str(c)] for c in case_ids]
    return frame


def main() -> None:
    """Generate the folded labels CSV and log the per-fold class balance."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    args = build_parser().parse_args()
    frame = assign_folds(
        args.labels_path,
        args.id_column,
        args.label_column,
        args.n_splits,
        args.random_state,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    logging.info(
        "Wrote %d cases with %d folds to %s",
        len(frame),
        frame["fold"].nunique(),
        args.out,
    )
    balance = (
        frame.groupby("fold")[args.label_column]
        .agg(["size", "sum", "mean"])
        .rename(columns={"size": "n", "sum": "n_pos", "mean": "pos_frac"})
    )
    logging.info("Per-fold class balance:\n%s", balance.to_string())


if __name__ == "__main__":
    main()
