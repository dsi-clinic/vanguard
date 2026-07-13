#!/usr/bin/env bash
#SBATCH --job-name=vanguard_motion
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --partition=tier1q

set -euo pipefail

: "${VANGUARD_ROOT:?set VANGUARD_ROOT}"
: "${VANGUARD_PYTHON:?set VANGUARD_PYTHON}"
: "${MANIFEST:?set MANIFEST}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT}"
: "${SLURM_ARRAY_TASK_ID:?submit this script as an array}"

cd "$VANGUARD_ROOT"
test -x "$VANGUARD_PYTHON"
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1

"$VANGUARD_PYTHON" -m preprocessing.run_motion_shard \
  --manifest "$MANIFEST" \
  --output-root "$OUTPUT_ROOT" \
  --row-index "$SLURM_ARRAY_TASK_ID"
