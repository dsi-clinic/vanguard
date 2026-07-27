#!/bin/bash
#SBATCH --job-name=vanguard-vessel
#SBATCH --partition=gpuq
#SBATCH --qos=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/vanguard-vessel-%j.out
# infer now runs breast+vessel model inference twice per case (HR route,
# then the UFAST-direct route -- UFAST typically has several times more
# temporal phases than HR, e.g. 24 vs 5 for a representative exam), so the
# original 4h HR-only budget is doubled here for headroom.

set -euo pipefail

: "${EXAM_ID:?set EXAM_ID}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT}"
: "${VANGUARD_ENV_PREFIX:=/ess/scratch/scratch1/aakrithiram/micromamba/envs/vanguard}"

micromamba run -p "$VANGUARD_ENV_PREFIX" \
  python -m preprocessing.pipeline infer \
  --exam-id "$EXAM_ID" \
  --output-root "$OUTPUT_ROOT" \
  "${@}"
