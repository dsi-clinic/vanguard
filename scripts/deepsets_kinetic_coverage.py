"""Report kinetic_signal_ok coverage for a dynamic Deep Sets experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

KINETIC_OK_THRESHOLD = 0.5
LOW_COVERAGE_FRACTION = 0.01


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for kinetic coverage reporting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Experiment root containing sets/*.pt",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=50,
        help="Maximum number of .pt sets to scan (0 = all, first N when capped)",
    )
    return parser.parse_args()


def main() -> None:
    """Sample Deep Sets .pt sets and print kinetic_signal_ok point fractions."""
    args = parse_args()
    sets_dir = args.out_root / "sets"
    paths = sorted(sets_dir.glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No .pt sets in {sets_dir}")
    if args.max_cases > 0 and len(paths) > args.max_cases:
        paths = paths[: args.max_cases]
    case_rows = []
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        feats = payload.get("features")
        names = payload.get("feature_names") or []
        if feats is None or "kinetic_signal_ok" not in names:
            continue
        idx = list(names).index("kinetic_signal_ok")
        col = feats[:, idx].numpy()
        frac_ok = float((col > KINETIC_OK_THRESHOLD).mean())
        case_rows.append((path.stem, frac_ok, int(col.shape[0])))
    if not case_rows:
        print(f"No kinetic_signal_ok column in sampled sets under {sets_dir}")
        return
    fracs = [r[1] for r in case_rows]
    print(f"Sampled {len(case_rows)} sets from {sets_dir}")
    print(
        f"  mean fraction points with kinetic_signal_ok==1: {sum(fracs) / len(fracs):.3f}"
    )
    print(f"  min / max: {min(fracs):.3f} / {max(fracs):.3f}")
    low = [r for r in case_rows if r[1] < LOW_COVERAGE_FRACTION]
    if low:
        print(f"  cases with <1% kinetic ok (first 5): {[r[0] for r in low[:5]]}")


if __name__ == "__main__":
    main()
