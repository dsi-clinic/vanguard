#!/usr/bin/env bash
# Run the 4d morphometry merge job individually (no SLURM dependency).
# Use this when the array jobs have already completed but the merge submission
# failed (e.g. "Job dependency problem" after jobs finished).
#
# Usage:
#   ./run_4d_merge.sh [OUTPUT_DIR]
#
#   OUTPUT_DIR  Output directory containing manifest_task_*.json (default: /net/projects2/vanguard/report/4d_morphometry)
#
# Submit via sbatch: sbatch --export=OUTPUT_DIR=... submit_4d_merge.slurm
# Or run directly:  ./run_4d_merge.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${1:-/net/projects2/vanguard/report/4d_morphometry}"

echo "Submitting merge job (no dependency) for OUTPUT_DIR=${OUTPUT_DIR}"
sbatch \
  --export=OUTPUT_DIR="${OUTPUT_DIR}" \
  "${SCRIPT_DIR}/submit_4d_merge.slurm"

echo "Done. Check logs/ for output."
