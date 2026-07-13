#!/usr/bin/env bash
set -euo pipefail

# Step 4d smoke test: run skeletonization/graph extraction for the same 5
# UChicago patients the segmentation smoke job processes (Step 4d commit),
# via the newly-wired --dataset-name uchicago adapter path.
#
# Deliberately NOT using submit_tc4d_array.sh/.slurm: that script passes
# --study-id, a flag that does not exist on run_skeleton_processing.py's CLI
# (only --case-id does) -- pre-existing drift, unrelated to this work, left
# alone rather than silently "fixed" as a side effect of an unrelated change.
#
# KNOWN, ACCEPTED RISK: see submit_uchicago_segmentation_smoke.slurm --
# UChicagoDataset.preprocess()'s orientation assumption is unverified.
#
# Usage (from repo root, after the segmentation smoke job has completed):
#   SEG_OUTPUT_DIR=/ess/scratch/scratch1/t-9sbose/uchicago_vessel_segmentations_smoke \
#     ./graph_extraction/slurm/submit_uchicago_skeleton_smoke.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UCHICAGO_ROOT="${UCHICAGO_ROOT:-/gpfs/data/karczmar-lab/vanguard/dce2d_internal_ultrafast_manifest}"
SEG_OUTPUT_DIR="${SEG_OUTPUT_DIR:?SEG_OUTPUT_DIR must be set to the segmentation smoke jobs --output-dir}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/ess/scratch/scratch1/t-9sbose/uchicago_centerlines_smoke}"
PATIENT_LIMIT="${PATIENT_LIMIT:-5}"
# `general` does not exist on this cluster; `tier1q` is the confirmed
# replacement (see gnn/slurm/submit_gnn_build.slurm).
PARTITION="${PARTITION:-tier1q}"

mkdir -p "${REPO_ROOT}/logs" "${OUTPUT_ROOT}"
CASE_LIST="$(mktemp "${REPO_ROOT}/logs/uc-skeleton-cases.XXXXXX.txt")"

# Same discovery the segmentation job used (UChicagoDataset.discover_cases()
# is deterministic given a fixed manifest), so both stages agree on the exact
# same 5 patients without hardcoding fragile exam ids. Also records each
# case's sub-source, because build_output_path writes segmentation output
# under SEG_OUTPUT_DIR/<sub_source>/<case_id>/images/ -- the graph stage's
# --input-dir must be sub-source-qualified to find it.
PYTHONPATH="${REPO_ROOT}" python - "${UCHICAGO_ROOT}" "${PATIENT_LIMIT}" <<'PY' > "${CASE_LIST}"
import sys
from pathlib import Path
from cohorts.uchicago import UChicagoDataset

root, limit = Path(sys.argv[1]), int(sys.argv[2])
adapter = UChicagoDataset(root=root)
for case_id in adapter.discover_cases()[:limit]:
    print(f"{case_id},{adapter.case_dataset_name(case_id)}")
PY

if [[ ! -s "${CASE_LIST}" ]]; then
  echo "No cases discovered under ${UCHICAGO_ROOT}" >&2
  exit 1
fi

TASK_COUNT="$(wc -l < "${CASE_LIST}")"
ARRAY_SPEC="0-$((TASK_COUNT - 1))"

JOB_ID="$({
  sbatch --parsable \
    --partition="${PARTITION}" \
    --array="${ARRAY_SPEC}" \
    --export=ALL,REPO_ROOT="${REPO_ROOT}",UCHICAGO_ROOT="${UCHICAGO_ROOT}",SEG_OUTPUT_DIR="${SEG_OUTPUT_DIR}",OUTPUT_ROOT="${OUTPUT_ROOT}",CASE_LIST="${CASE_LIST}" \
    "${SCRIPT_DIR}/submit_uchicago_skeleton_smoke.slurm"
} )"

cat <<MSG
Submitted UChicago skeletonization smoke array job:
  seg_output_dir: ${SEG_OUTPUT_DIR}
  output_root   : ${OUTPUT_ROOT}
  case_list     : ${CASE_LIST}
  task_count    : ${TASK_COUNT}
  job_id        : ${JOB_ID}

Monitor:
  squeue -j ${JOB_ID}
  sacct -j ${JOB_ID} --format=JobIDRaw,State,Elapsed,ExitCode -n -P
MSG
