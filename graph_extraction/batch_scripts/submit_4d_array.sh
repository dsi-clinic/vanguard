#!/bin/bash
# Discover study count, submit SLURM arrays in chunks of ~200, wait for each chunk
# to finish before submitting the next (keeps under 250 job queue limit).
#
# Usage: ./submit_4d_array.sh [INPUT_DIR] [OUTPUT_DIR] [--test]
#
#   INPUT_DIR   Base directory for vessel segmentations (default: /net/projects2/vanguard/vessel_segmentations)
#   OUTPUT_DIR  Output directory for morphometry JSONs and manifest (default: /net/projects2/vanguard/report/4d_morphometry)
#   --test      Run only on first 5 studies
#
# Run directly on head node: ./submit_4d_array.sh
# Or via srun (waits for queue to clear between batches):
#   srun -N1 -n1 --mem=512M --partition=general --time=24:00:00 ./submit_4d_array.sh

set -euo pipefail

CHUNK_SIZE=200
STUDIES_PER_TASK=${STUDIES_PER_TASK:-8}
POLL_INTERVAL=30

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TEST_MODE=false
POSITIONAL=()
for arg in "$@"; do
    [[ "$arg" == "--test" ]] && TEST_MODE=true || POSITIONAL+=("$arg")
done
INPUT_DIR="${POSITIONAL[0]:-/net/projects2/vanguard/vessel_segmentations}"
OUTPUT_DIR="${POSITIONAL[1]:-/net/projects2/vanguard/report/4d_morphometry}"

cd "${PROJECT_ROOT}"
mkdir -p logs

COUNT=$(python -c "
from pathlib import Path
import sys
sys.path.insert(0, '.')
from graph_extraction.batch_process_4d import discover_all_study_ids
ids = discover_all_study_ids(Path('${INPUT_DIR}'))
print(len(ids))
")

if [ "${COUNT}" -eq 0 ]; then
    echo "No studies found in ${INPUT_DIR}. Exiting."
    exit 1
fi

if [ "${TEST_MODE}" = true ]; then
    COUNT=$(( COUNT < 5 ? COUNT : 5 ))
    echo "[TEST MODE] Limiting to first ${COUNT} studies"
fi

wait_for_job() {
    local jid="$1"
    echo "Waiting for job ${jid} to complete (polling every ${POLL_INTERVAL}s)..."
    while squeue -h -j "${jid}" 2>/dev/null | grep -q .; do
        sleep "${POLL_INTERVAL}"
    done
    echo "Job ${jid} finished."
}

# Submit chunks one at a time, wait for each to finish before submitting the next
# Each array task processes STUDIES_PER_TASK studies in parallel; array size = ceil(chunk_size / STUDIES_PER_TASK)
ARRAY_JOBS=()
for (( OFFSET=0; OFFSET < COUNT; OFFSET += CHUNK_SIZE )); do
    CHUNK_END=$(( OFFSET + CHUNK_SIZE ))
    if [ "${CHUNK_END}" -gt "${COUNT}" ]; then
        CHUNK_END=${COUNT}
    fi
    CHUNK_SIZE_ACTUAL=$(( CHUNK_END - OFFSET ))
    TASKS_IN_CHUNK=$(( (CHUNK_SIZE_ACTUAL + STUDIES_PER_TASK - 1) / STUDIES_PER_TASK ))
    TASK_LAST=$(( TASKS_IN_CHUNK - 1 ))
    TASK_OFFSET=$(( OFFSET / STUDIES_PER_TASK ))
    echo "Submitting chunk: studies ${OFFSET}-$((CHUNK_END-1)) (${CHUNK_SIZE_ACTUAL} studies, ${TASKS_IN_CHUNK} tasks x ${STUDIES_PER_TASK} workers, array 0-${TASK_LAST})"
    JOB=$(sbatch \
      --array=0-${TASK_LAST} \
      --export=INPUT_DIR="${INPUT_DIR}",OUTPUT_DIR="${OUTPUT_DIR}",STUDY_OFFSET="${OFFSET}",CHUNK_END="${CHUNK_END}",STUDIES_PER_TASK="${STUDIES_PER_TASK}",TASK_OFFSET="${TASK_OFFSET}" \
      "${SCRIPT_DIR}/submit_4d_morphometry.slurm" \
      | awk '{print $4}')
    ARRAY_JOBS+=("${JOB}")
    wait_for_job "${JOB}"
done

DEPS=$(IFS=:; echo "${ARRAY_JOBS[*]}")
echo "All array jobs completed. Submitting merge job (depends on: ${DEPS})..."

sbatch \
  --dependency=afterok:${DEPS} \
  --export=OUTPUT_DIR="${OUTPUT_DIR}" \
  "${SCRIPT_DIR}/submit_4d_merge.slurm"

echo "Done. Check logs/ for output."
