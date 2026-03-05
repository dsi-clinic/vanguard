#!/usr/bin/env bash
# Head-node helper to submit 4d-vs-tc4d benchmark array + reducer.
#
# Usage:
#   slurm_submit_scripts/submit_4d_vs_tc4d_benchmark.sh \
#     /net/projects2/vanguard/benchmarks/4d_vs_tc4d/run_$(date +%Y%m%d_%H%M%S)
#
# Optional env vars:
#   SEGMENTATION_DIR=/net/projects2/vanguard/vessel_segmentations
#   COMPARE_SCRIPT=/path/to/graph_extraction/debug_compare_4d_vs_tc4d.py

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-${PROJECT_ROOT}/benchmark_runs/4d_vs_tc4d_$(date +%Y%m%d_%H%M%S)}"
SEGMENTATION_DIR="${SEGMENTATION_DIR:-/net/projects2/vanguard/vessel_segmentations}"
COMPARE_SCRIPT="${COMPARE_SCRIPT:-${PROJECT_ROOT}/graph_extraction/debug_compare_4d_vs_tc4d.py}"
MANIFEST_PATH="${OUTPUT_DIR}/study_ids.txt"
BENCH_GIT_COMMIT="$(git -C "${PROJECT_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
BENCH_GIT_SHORT="$(git -C "${PROJECT_ROOT}" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
BENCH_GIT_DIRTY="unknown"
if GIT_STATUS="$(git -C "${PROJECT_ROOT}" status --porcelain 2>/dev/null)"; then
  if [[ -n "${GIT_STATUS}" ]]; then
    BENCH_GIT_DIRTY="true"
  else
    BENCH_GIT_DIRTY="false"
  fi
fi
JOB_TAG="${BENCH_GIT_SHORT}"
if [[ "${BENCH_GIT_DIRTY}" == "true" ]]; then
  JOB_TAG="${JOB_TAG}-dirty"
fi
JOB_TAG="${JOB_TAG//[^A-Za-z0-9_-]/_}"
ARRAY_JOB_NAME="4d-tc4d-bench-${JOB_TAG}"
REDUCE_JOB_NAME="4d-tc4d-reduce-${JOB_TAG}"

source ~/.bashrc || true
eval "$(micromamba shell hook -s bash)"
micromamba activate vanguard

mkdir -p "${OUTPUT_DIR}" logs

python -u "${PROJECT_ROOT}/graph_extraction/benchmark_4d_vs_tc4d.py" \
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
  --job-name="${ARRAY_JOB_NAME}" \
  --array="${ARRAY_RANGE}" \
  --parsable \
  --export=ALL,BENCH_MANIFEST="${MANIFEST_PATH}",BENCH_OUTPUT_DIR="${OUTPUT_DIR}",SEGMENTATION_DIR="${SEGMENTATION_DIR}",COMPARE_SCRIPT="${COMPARE_SCRIPT}",BENCH_GIT_COMMIT="${BENCH_GIT_COMMIT}",BENCH_GIT_SHORT="${BENCH_GIT_SHORT}",BENCH_GIT_DIRTY="${BENCH_GIT_DIRTY}" \
  "${PROJECT_ROOT}/slurm_submit_scripts/submit_4d_vs_tc4d_benchmark_array.slurm")

REDUCE_JOB_ID=$(sbatch \
  --job-name="${REDUCE_JOB_NAME}" \
  --dependency="afterok:${ARRAY_JOB_ID}" \
  --parsable \
  --export=ALL,BENCH_OUTPUT_DIR="${OUTPUT_DIR}",BENCH_GIT_COMMIT="${BENCH_GIT_COMMIT}",BENCH_GIT_SHORT="${BENCH_GIT_SHORT}",BENCH_GIT_DIRTY="${BENCH_GIT_DIRTY}" \
  "${PROJECT_ROOT}/slurm_submit_scripts/submit_4d_vs_tc4d_benchmark_reduce.slurm")

cat <<EOF
Submitted 4d-vs-tc4d benchmark suite:
  array_job_id  : ${ARRAY_JOB_ID}
  reduce_job_id : ${REDUCE_JOB_ID}
  git_commit    : ${BENCH_GIT_COMMIT}
  git_short     : ${BENCH_GIT_SHORT}
  git_dirty     : ${BENCH_GIT_DIRTY}
  array_name    : ${ARRAY_JOB_NAME}
  reduce_name   : ${REDUCE_JOB_NAME}
  manifest      : ${MANIFEST_PATH}
  output_dir    : ${OUTPUT_DIR}

Monitor:
  squeue -j ${ARRAY_JOB_ID},${REDUCE_JOB_ID}
  tail -f logs/${ARRAY_JOB_NAME}-${ARRAY_JOB_ID}-0.out
EOF
