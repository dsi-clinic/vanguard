#!/bin/bash
#SBATCH --job-name=vanguard-prep
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/vanguard-prep-%j.out

set -euo pipefail

: "${INVENTORY:?set INVENTORY}"
: "${CASE_MANIFEST:?set CASE_MANIFEST}"
: "${EXAM_ID:?set EXAM_ID}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT}"
: "${VANGUARD_ENV_PREFIX:=/ess/scratch/scratch1/aakrithiram/micromamba/envs/vanguard}"

micromamba run -p "$VANGUARD_ENV_PREFIX" \
  python -m preprocessing.pipeline prepare \
  --inventory "$INVENTORY" \
  --case-manifest "$CASE_MANIFEST" \
  --exam-id "$EXAM_ID" \
  --output-root "$OUTPUT_ROOT"
