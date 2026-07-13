#!/usr/bin/env bash
# Submit restartable exam shards followed by a semantic merge.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VANGUARD_ROOT="${VANGUARD_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
VANGUARD_PYTHON="${VANGUARD_PYTHON:-$(command -v python)}"
: "${MANIFEST:?set MANIFEST to the SPGR-safe CSV}"
: "${OUTPUT_ROOT:?set OUTPUT_ROOT to a new derived-data directory}"
CONCURRENCY="${CONCURRENCY:-32}"
MAXIMUM_QC_PANELS="${MAXIMUM_QC_PANELS:-24}"

test -x "$VANGUARD_PYTHON"
mkdir -p "$OUTPUT_ROOT/logs"

ROW_COUNT="$($VANGUARD_PYTHON -c 'import csv,sys; print(sum(1 for _ in csv.DictReader(open(sys.argv[1], newline=""))))' "$MANIFEST")"
if [[ "$ROW_COUNT" -le 0 ]]; then
  echo "manifest has no rows: $MANIFEST" >&2
  exit 1
fi
LAST_INDEX="$((ROW_COUNT - 1))"

EXPORTS="ALL,VANGUARD_ROOT=$VANGUARD_ROOT,VANGUARD_PYTHON=$VANGUARD_PYTHON,MANIFEST=$MANIFEST,OUTPUT_ROOT=$OUTPUT_ROOT,MAXIMUM_QC_PANELS=$MAXIMUM_QC_PANELS"
ARRAY_JOB_ID="$(sbatch --parsable \
  --array="0-${LAST_INDEX}%${CONCURRENCY}" \
  --export="$EXPORTS" \
  --output="$OUTPUT_ROOT/logs/motion_%A_%a.out" \
  --error="$OUTPUT_ROOT/logs/motion_%A_%a.err" \
  "$SCRIPT_DIR/motion_correction_array.slurm")"
MERGE_JOB_ID="$(sbatch --parsable \
  --dependency="afterany:$ARRAY_JOB_ID" \
  --kill-on-invalid-dep=yes \
  --export="$EXPORTS" \
  --output="$OUTPUT_ROOT/logs/merge_%j.out" \
  --error="$OUTPUT_ROOT/logs/merge_%j.err" \
  "$SCRIPT_DIR/merge_motion_correction.slurm")"

{
  echo "array_job_id=$ARRAY_JOB_ID"
  echo "merge_job_id=$MERGE_JOB_ID"
  echo "manifest=$MANIFEST"
  echo "output_root=$OUTPUT_ROOT"
  echo "row_count=$ROW_COUNT"
  echo "concurrency=$CONCURRENCY"
} > "$OUTPUT_ROOT/slurm_job_ids.txt"

echo "array_job_id=$ARRAY_JOB_ID"
echo "merge_job_id=$MERGE_JOB_ID"
