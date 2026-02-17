#!/usr/bin/env bash
#SBATCH --job-name=legacy-primary-bench
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/legacy-primary-bench-%A-%a.out
#SBATCH --error=logs/legacy-primary-bench-%A-%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ruochun@duke.edu

set -euo pipefail

if [[ -z "${BENCH_MANIFEST:-}" ]]; then
  echo "ERROR: BENCH_MANIFEST is required (newline-delimited study IDs)." >&2
  exit 2
fi

if [[ -z "${BENCH_OUTPUT_DIR:-}" ]]; then
  echo "ERROR: BENCH_OUTPUT_DIR is required." >&2
  exit 2
fi

source ~/.bashrc || true
eval "$(micromamba shell hook -s bash)"
micromamba activate vanguard

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEGMENTATION_DIR="${SEGMENTATION_DIR:-/net/projects2/vanguard/vessel_segmentations}"
COMPARE_SCRIPT="${COMPARE_SCRIPT:-${PROJECT_ROOT}/graph_extraction/run_compare_legacy_pipeline_debug.py}"
AUTO_TUMOR_MASK="${AUTO_TUMOR_MASK:-0}"

mkdir -p logs "${BENCH_OUTPUT_DIR}"

CMD=(
  python -u "${PROJECT_ROOT}/batch_processing/benchmark_legacy_vs_primary.py"
  --manifest-file "${BENCH_MANIFEST}"
  --task-index "${SLURM_ARRAY_TASK_ID}"
  --output-dir "${BENCH_OUTPUT_DIR}"
  --segmentation-dir "${SEGMENTATION_DIR}"
  --compare-script "${COMPARE_SCRIPT}"
  --skip-existing
)

if [[ "${AUTO_TUMOR_MASK}" == "1" ]]; then
  CMD+=(--auto-tumor-mask)
else
  CMD+=(--no-auto-tumor-mask)
fi

echo "========================================="
echo "Legacy-vs-Primary Benchmark Array Task"
echo "========================================="
echo "Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Manifest: ${BENCH_MANIFEST}"
echo "Output dir: ${BENCH_OUTPUT_DIR}"
echo "Segmentation dir: ${SEGMENTATION_DIR}"
echo "Compare script: ${COMPARE_SCRIPT}"
echo "Date: $(date)"
echo "========================================="

"${CMD[@]}"


echo "========================================="
echo "Benchmark array task complete"
echo "Date: $(date)"
echo "========================================="
