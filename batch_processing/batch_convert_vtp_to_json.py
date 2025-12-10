#!/usr/bin/env python3
"""Batch convert VTP centerline files to JSON format."""

import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

SCRIPT_DIR = Path(__file__).parent.parent
CENTERLINES_DIR = Path("/net/projects2/vanguard/centerlines")
JSON_DIR = CENTERLINES_DIR / "json_files"
SCRIPT_PATH = SCRIPT_DIR / "centerline_to_json.py"


def convert_vtp_to_json(vtp_file: Path, json_file: Path) -> bool:
    """Convert a single VTP file to JSON.

    Args:
        vtp_file: Path to input VTP file
        json_file: Path to output JSON file

    Returns:
        True if successful, False otherwise
    """
    try:
        result = subprocess.run(  # noqa: S603
            [
                sys.executable,
                str(SCRIPT_PATH),
                str(vtp_file),
                str(json_file),
                "--spacing",
                "1.0",
                "1.0",
                "1.0",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            print(f"Error converting {vtp_file.name}: {result.stderr}", file=sys.stderr)
            return False

        return True
    except subprocess.TimeoutExpired:
        print(f"Timeout converting {vtp_file.name}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Exception converting {vtp_file.name}: {e}", file=sys.stderr)
        return False


def main() -> None:
    """Convert all VTP files in centerlines directory to JSON."""
    JSON_DIR.mkdir(parents=True, exist_ok=True)

    vtp_files = sorted(CENTERLINES_DIR.glob("*.vtp"))
    total = len(vtp_files)

    if total == 0:
        print(f"No VTP files found in {CENTERLINES_DIR}")
        return

    print(f"Found {total} VTP files to convert")
    print(f"Output directory: {JSON_DIR}")

    successful = 0
    failed = 0

    for vtp_file in tqdm(vtp_files, desc="Converting"):
        json_file = JSON_DIR / f"{vtp_file.stem}.json"

        # Skip if already converted
        if json_file.exists():
            successful += 1
            continue

        if convert_vtp_to_json(vtp_file, json_file):
            successful += 1
        else:
            failed += 1

    print("\nConversion complete:")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total: {total}")


if __name__ == "__main__":
    main()
