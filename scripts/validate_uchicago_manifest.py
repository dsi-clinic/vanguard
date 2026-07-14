r"""Step 3 validation: exercise the UChicago adapter against the real manifest.

Builds a ``UChicagoDataset`` through the run-config factory (the seam a stage
would use) and checks the manifest-driven path — case discovery, labels,
provided CV folds, sub-source identity, timepoints, and preprocess pass-through —
against the known counts in the manifest data dictionary. No vessel pipeline and
no heavy compute; this is the merge check for Step 3.

The manifest root defaults to the canonical location but can be overridden:

    UCHICAGO_ROOT=/path/to/dce2d_internal_ultrafast_manifest \
    python scripts/validate_uchicago_manifest.py

Exit code 0 = all checks pass, 1 = a check failed.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from cohorts import build_adapter_from_config, resolve_folds
from config import ConfigNode

# Expected counts from the manifest data dictionary (181 labeled exams).
EXPECTED_TOTAL = 181
EXPECTED_LABELS = {0: 117, 1: 64}
EXPECTED_FOLDS = {0: 36, 1: 36, 2: 37, 3: 36, 4: 36}
EXPECTED_SUBSOURCES = {"simbiosys": 86, "uch_nac": 60, "her2_naclike": 35}

DEFAULT_ROOT = "/gpfs/data/karczmar-lab/vanguard/dce2d_internal_ultrafast_manifest"


def parse_args() -> argparse.Namespace:
    """Parse CLI args, defaulting the manifest root to env/canonical path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=os.environ.get("UCHICAGO_ROOT", DEFAULT_ROOT),
        help="UChicago manifest root dir (or set UCHICAGO_ROOT).",
    )
    return parser.parse_args()


def _check(label: str, got: object, want: object, failures: list[str]) -> None:
    """Record and print one check result."""
    ok = got == want
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {got}")
    if not ok:
        failures.append(f"{label}: got {got}, expected {want}")


def main() -> int:
    """Run the manifest-path checks and return a process exit code."""
    args = parse_args()
    config = ConfigNode._wrap(
        {
            "dataset": {
                "name": "uchicago",
                "cohort": None,
                "root": args.root,
                "split_policy": "auto",
            }
        }
    )
    adapter = build_adapter_from_config(config)
    print(f"Adapter: {type(adapter).__name__}  root={args.root}")

    failures: list[str] = []

    cases = adapter.discover_cases()
    _check("discover_cases (total exams)", len(cases), EXPECTED_TOTAL, failures)

    labels = adapter.load_labels()
    label_counts = {int(k): int(v) for k, v in labels["pcr"].value_counts().items()}
    _check("load_labels (pcr counts)", label_counts, EXPECTED_LABELS, failures)

    folds = resolve_folds(config, adapter)  # auto -> provided
    fold_counts = {int(k): int(v) for k, v in folds["fold"].value_counts().items()}
    _check(
        "resolve_folds (provided fold counts)", fold_counts, EXPECTED_FOLDS, failures
    )

    subsource_counts: dict[str, int] = {}
    for case_id in cases:
        name = adapter.case_dataset_name(case_id)
        subsource_counts[name] = subsource_counts.get(name, 0) + 1
    _check(
        "case_dataset_name (sub-source counts)",
        subsource_counts,
        EXPECTED_SUBSOURCES,
        failures,
    )

    # Timepoints resolve to real, existing phase files for the first case.
    tps = adapter.load_timepoints(cases[0])
    _check(
        "load_timepoints[0] all exist",
        bool(tps) and all(p.exists() for p in tps),
        True,
        failures,
    )

    # preprocess is a pass-through (data already preprocessed).
    vol = np.arange(24).reshape(2, 3, 4)
    _check(
        "preprocess is pass-through",
        np.array_equal(adapter.preprocess(vol), vol),
        True,
        failures,
    )

    print()
    if failures:
        print(f"FAIL: {len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASS: UChicago adapter matches the manifest dictionary on all checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
