#!/usr/bin/env bash
#SBATCH --job-name=vanguard_motion_merge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --partition=tier1q

set -euo pipefail

: "${VANGUARD_ROOT:?set VANGUARD_ROOT}"
: "${VANGUARD_PYTHON:?set VANGUARD_PYTHON}"
: "${MANIFEST:?set MANIFEST}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT}"

cd "$VANGUARD_ROOT"
test -x "$VANGUARD_PYTHON"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=2

"$VANGUARD_PYTHON" -m preprocessing.merge_motion_shards \
  --manifest "$MANIFEST" \
  --output-root "$OUTPUT_ROOT" \
  --maximum-qc-panels "${MAXIMUM_QC_PANELS:-24}"
