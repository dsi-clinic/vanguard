#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
INPUT_ROOT="${1:-/ess/scratch/scratch1/t-9sbose/vessel_segmentations}"
OUTPUT_ROOT="${2:-/ess/scratch/scratch1/t-9sbose/centerlines_tc4d/studies}"
PARTITION="${PARTITION:-tier1q}"
MAX_CONCURRENT="${MAX_CONCURRENT:-10}"

mkdir -p "${REPO_ROOT}/logs"
STUDY_LIST="$(mktemp "${REPO_ROOT}/logs/tc4d-missing33.XXXXXX.txt")"

# Define the 33 missing cases
cat > "${STUDY_LIST}" << 'CASES'
DUKE/DUKE_012
DUKE/DUKE_019
DUKE/DUKE_021
DUKE/DUKE_022
DUKE/DUKE_045
DUKE/DUKE_046
DUKE/DUKE_069
DUKE/DUKE_101
DUKE/DUKE_119
DUKE/DUKE_142
DUKE/DUKE_168
DUKE/DUKE_233
DUKE/DUKE_234
DUKE/DUKE_258
DUKE/DUKE_307
DUKE/DUKE_378
DUKE/DUKE_400
DUKE/DUKE_489
DUKE/DUKE_491
ISPY2/ISPY2_239061
ISPY2/ISPY2_255388
ISPY2/ISPY2_255535
ISPY2/ISPY2_275626
ISPY2/ISPY2_277848
ISPY2/ISPY2_277888
ISPY2/ISPY2_287300
ISPY2/ISPY2_287961
ISPY2/ISPY2_299840
ISPY2/ISPY2_311316
ISPY2/ISPY2_311455
ISPY2/ISPY2_313243
ISPY2/ISPY2_317641
ISPY2/ISPY2_318293
CASES

TASK_COUNT="$(wc -l < "${STUDY_LIST}")"
ARRAY_SPEC="0-$((TASK_COUNT - 1))%${MAX_CONCURRENT}"

JOB_ID="$({
  sbatch --parsable \
    --partition="${PARTITION}" \
    --array="${ARRAY_SPEC}" \
    --export=ALL,REPO_ROOT="${REPO_ROOT}",INPUT_ROOT="${INPUT_ROOT}",OUTPUT_ROOT="${OUTPUT_ROOT}",STUDY_LIST="${STUDY_LIST}" \
    "${SCRIPT_DIR}/submit_tc4d_array.slurm"
} )"

cat <<MSG
Submitted tc4d array job for 33 missing centerlines:
  input_root : ${INPUT_ROOT}
  output_root: ${OUTPUT_ROOT}
  study_list : ${STUDY_LIST}
  task_count : ${TASK_COUNT}
  job_id     : ${JOB_ID}

Monitor:
  squeue -j ${JOB_ID}
  sacct -j ${JOB_ID} --format=JobIDRaw,State,Elapsed,ExitCode -n -P
MSG
