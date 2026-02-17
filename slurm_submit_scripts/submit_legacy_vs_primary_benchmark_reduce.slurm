#!/usr/bin/env bash
#SBATCH --job-name=legacy-primary-reduce
#SBATCH --partition=general
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/legacy-primary-reduce-%j.out
#SBATCH --error=logs/legacy-primary-reduce-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ruochun@duke.edu

set -euo pipefail

if [[ -z "${BENCH_OUTPUT_DIR:-}" ]]; then
  echo "ERROR: BENCH_OUTPUT_DIR is required." >&2
  exit 2
fi

source ~/.bashrc || true
eval "$(micromamba shell hook -s bash)"
micromamba activate vanguard

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p logs "${BENCH_OUTPUT_DIR}"

python -u "${PROJECT_ROOT}/batch_processing/reduce_legacy_vs_primary.py" \
  --benchmark-dir "${BENCH_OUTPUT_DIR}" \
  --output-dir "${BENCH_OUTPUT_DIR}"


echo "========================================="
echo "Benchmark reduction complete"
echo "Output dir: ${BENCH_OUTPUT_DIR}"
echo "Date: $(date)"
echo "========================================="
