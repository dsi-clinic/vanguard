"""Assert two built GNN caches contain exactly the same case-ID set.

Phase 5 of the harmonized single-breast dataset plan: arms 1 (mixed
baseline) and 2 (single breast) are designed to share an identical cohort
by construction -- both consume the same matched-cohort labels file
(``pcr_labels_folded_matched_cohort.csv``) -- but this checks it directly
against each cache's own build-time record (``graph_qc.csv``) rather than
assuming the construction was correct. A silent case-set mismatch between
the two arms would confound "does single-breast help" with "are these even
the same cases," so this is a hard gate before any training run: raises
loudly on any mismatch instead of proceeding.

Usage::

    python -m analysis.assert_arm_parity \
        --cache-a /ess/scratch/.../gnn_cache_voxel_mixed_baseline \
        --cache-b /ess/scratch/.../gnn_cache_voxel_single_breast
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-a", type=Path, required=True)
    parser.add_argument("--cache-b", type=Path, required=True)
    return parser.parse_args()


def load_case_ids(cache_dir: Path) -> set[str]:
    """Return the set of case_ids actually built into a cache, from its graph_qc.csv."""
    qc_path = cache_dir / "processed" / "graph_qc.csv"
    if not qc_path.exists():
        raise FileNotFoundError(f"No graph_qc.csv under {cache_dir} -- was it built?")
    return set(pd.read_csv(qc_path)["case_id"].astype(str))


def main() -> None:
    """Load both caches' case-ID sets and raise on any mismatch."""
    args = _parse_args()
    cases_a = load_case_ids(args.cache_a)
    cases_b = load_case_ids(args.cache_b)

    only_a = cases_a - cases_b
    only_b = cases_b - cases_a

    print(f"cache_a ({args.cache_a}): {len(cases_a)} cases")
    print(f"cache_b ({args.cache_b}): {len(cases_b)} cases")

    if only_a or only_b:
        raise RuntimeError(
            f"Arm case-ID mismatch: {len(only_a)} case(s) only in cache_a "
            f"({sorted(only_a)[:10]}...), {len(only_b)} case(s) only in "
            f"cache_b ({sorted(only_b)[:10]}...). The two arms must share "
            "an identical cohort for the comparison to be meaningful."
        )
    print(f"MATCH: both caches contain the identical {len(cases_a)} case(s).")


if __name__ == "__main__":
    main()
