#!/usr/bin/env bash
#SBATCH --job-name=4d-tc4d-bench
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%A-%a.out
#SBATCH --error=logs/%x-%A-%a.err
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
COMPARE_SCRIPT="${COMPARE_SCRIPT:-${PROJECT_ROOT}/graph_extraction/debug_compare_4d_vs_tc4d.py}"

mkdir -p logs "${BENCH_OUTPUT_DIR}"

CMD=(
  python -u "${PROJECT_ROOT}/graph_extraction/benchmark_4d_vs_tc4d.py"
  --manifest-file "${BENCH_MANIFEST}"
  --task-index "${SLURM_ARRAY_TASK_ID}"
  --output-dir "${BENCH_OUTPUT_DIR}"
  --segmentation-dir "${SEGMENTATION_DIR}"
  --compare-script "${COMPARE_SCRIPT}"
  --skip-existing
)

echo "========================================="
echo "4d-vs-tc4d Benchmark Array Task"
echo "========================================="
echo "Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Manifest: ${BENCH_MANIFEST}"
echo "Output dir: ${BENCH_OUTPUT_DIR}"
echo "Segmentation dir: ${SEGMENTATION_DIR}"
echo "Compare script: ${COMPARE_SCRIPT}"
echo "Git commit: ${BENCH_GIT_COMMIT:-unknown}"
echo "Git short: ${BENCH_GIT_SHORT:-unknown}"
echo "Git dirty: ${BENCH_GIT_DIRTY:-unknown}"
echo "Date: $(date)"
echo "========================================="

"${CMD[@]}" 2>&1 | awk -v t0="$(date +%s)" '{
  now = systime();
  dt = now - t0;
  hh = int(dt / 3600);
  mm = int((dt % 3600) / 60);
  ss = dt % 60;
  printf("[%s +%02d:%02d:%02d] %s\n", strftime("%Y-%m-%d %H:%M:%S", now), hh, mm, ss, $0);
  fflush();
}'


echo "========================================="
echo "Benchmark array task complete"
echo "Date: $(date)"
echo "========================================="
