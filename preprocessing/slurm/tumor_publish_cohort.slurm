#!/usr/bin/env bash
#SBATCH --job-name=uchic-tumor-publish
#SBATCH --account=karczmar-lab
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

export PYTHONNOUSERSITE=1

eval "$(micromamba shell hook -s bash)"
micromamba activate vanguard
set -euo pipefail

: "${REPO_ROOT:?set REPO_ROOT}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT}"
: "${COHORT_MANIFEST:?set COHORT_MANIFEST}"
: "${CENTERLINE_ROOT:?set CENTERLINE_ROOT}"
: "${DIRECT_MANIFEST_1:?set at least DIRECT_MANIFEST_1}"

cd "${REPO_ROOT}"

arguments=(
  --output-root "${OUTPUT_ROOT}"
  --cohort-manifest "${COHORT_MANIFEST}"
  --centerline-root "${CENTERLINE_ROOT}"
  --direct-manifest "${DIRECT_MANIFEST_1}"
)
for variable in DIRECT_MANIFEST_2 DIRECT_MANIFEST_3; do
  value="${!variable:-}"
  if [[ -n "${value}" ]]; then
    arguments+=(--direct-manifest "${value}")
  fi
done

python -m preprocessing.tumor publish-cohort "${arguments[@]}"
