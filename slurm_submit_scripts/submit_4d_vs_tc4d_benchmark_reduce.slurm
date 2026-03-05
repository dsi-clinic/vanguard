#!/usr/bin/env bash
#SBATCH --job-name=4d-tc4d-reduce
#SBATCH --partition=general
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
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

python -u "${PROJECT_ROOT}/graph_extraction/reduce_4d_vs_tc4d.py" \
  --benchmark-dir "${BENCH_OUTPUT_DIR}" \
  --output-dir "${BENCH_OUTPUT_DIR}"


echo "========================================="
echo "Benchmark reduction complete"
echo "Output dir: ${BENCH_OUTPUT_DIR}"
echo "Git commit: ${BENCH_GIT_COMMIT:-unknown}"
echo "Git short: ${BENCH_GIT_SHORT:-unknown}"
echo "Git dirty: ${BENCH_GIT_DIRTY:-unknown}"
echo "Date: $(date)"
echo "========================================="
