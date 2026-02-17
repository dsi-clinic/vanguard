#!/usr/bin/env bash
# Head-node helper to submit legacy-vs-primary benchmark array + reducer.
#
# Usage:
#   slurm_submit_scripts/submit_legacy_vs_primary_benchmark.sh \
#     /net/projects2/vanguard/benchmarks/legacy_vs_primary/run_$(date +%Y%m%d_%H%M%S)
#
# Optional env vars:
#   SEGMENTATION_DIR=/net/projects2/vanguard/vessel_segmentations
#   COMPARE_SCRIPT=/path/to/run_compare_legacy_pipeline_debug.py
#   AUTO_TUMOR_MASK=1

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-${PROJECT_ROOT}/benchmark_runs/legacy_vs_primary_$(date +%Y%m%d_%H%M%S)}"
SEGMENTATION_DIR="${SEGMENTATION_DIR:-/net/projects2/vanguard/vessel_segmentations}"
COMPARE_SCRIPT="${COMPARE_SCRIPT:-${PROJECT_ROOT}/graph_extraction/run_compare_legacy_pipeline_debug.py}"
AUTO_TUMOR_MASK="${AUTO_TUMOR_MASK:-0}"
MANIFEST_PATH="${OUTPUT_DIR}/study_ids.txt"

source ~/.bashrc || true
eval "$(micromamba shell hook -s bash)"
micromamba activate vanguard

mkdir -p "${OUTPUT_DIR}" logs

python -u "${PROJECT_ROOT}/batch_processing/benchmark_legacy_vs_primary.py" \
  --segmentation-dir "${SEGMENTATION_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --manifest-only \
  --write-manifest "${MANIFEST_PATH}"

STUDY_COUNT=$(grep -cv '^\s*$' "${MANIFEST_PATH}" || true)
if [[ "${STUDY_COUNT}" -le 0 ]]; then
  echo "ERROR: Manifest is empty: ${MANIFEST_PATH}" >&2
  exit 3
fi

ARRAY_RANGE="0-$((STUDY_COUNT - 1))"

echo "Submitting benchmark array with ${STUDY_COUNT} studies (${ARRAY_RANGE})"
ARRAY_JOB_ID=$(sbatch \
  --array="${ARRAY_RANGE}" \
  --parsable \
  --export=ALL,BENCH_MANIFEST="${MANIFEST_PATH}",BENCH_OUTPUT_DIR="${OUTPUT_DIR}",SEGMENTATION_DIR="${SEGMENTATION_DIR}",COMPARE_SCRIPT="${COMPARE_SCRIPT}",AUTO_TUMOR_MASK="${AUTO_TUMOR_MASK}" \
  "${PROJECT_ROOT}/slurm_submit_scripts/submit_legacy_vs_primary_benchmark_array.slurm")

REDUCE_JOB_ID=$(sbatch \
  --dependency="afterok:${ARRAY_JOB_ID}" \
  --parsable \
  --export=ALL,BENCH_OUTPUT_DIR="${OUTPUT_DIR}" \
  "${PROJECT_ROOT}/slurm_submit_scripts/submit_legacy_vs_primary_benchmark_reduce.slurm")

cat <<EOF
Submitted legacy-vs-primary benchmark suite:
  array_job_id  : ${ARRAY_JOB_ID}
  reduce_job_id : ${REDUCE_JOB_ID}
  manifest      : ${MANIFEST_PATH}
  output_dir    : ${OUTPUT_DIR}

Monitor:
  squeue -j ${ARRAY_JOB_ID},${REDUCE_JOB_ID}
  tail -f logs/legacy-primary-bench-${ARRAY_JOB_ID}-0.out
EOF
