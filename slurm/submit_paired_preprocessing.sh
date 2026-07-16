#!/usr/bin/env bash
set -euo pipefail

: "${REPO_ROOT:?Set REPO_ROOT to the Vanguard checkout}"
: "${PAIRED_INVENTORY:?Set PAIRED_INVENTORY to the shared DICOM inventory}"
: "${CASE_MANIFEST:?Set CASE_MANIFEST to the reviewed runnable cases}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a derived-output directory}"

N_CASES="$(($(wc -l < "${CASE_MANIFEST}") - 1))"
if [[ "${N_CASES}" -le 0 ]]; then
  echo "case manifest is empty" >&2
  exit 2
fi
ARRAY="0-$((N_CASES - 1))"
EXPORTS="ALL,REPO_ROOT=${REPO_ROOT},PAIRED_INVENTORY=${PAIRED_INVENTORY},CASE_MANIFEST=${CASE_MANIFEST},OUTPUT_ROOT=${OUTPUT_ROOT}"

prepare_job="$(sbatch --parsable --array="${ARRAY}%8" \
  --export="${EXPORTS},PIPELINE_STAGE=prepare" \
  "${REPO_ROOT}/slurm/run_paired_preprocessing_cpu.slurm")"
infer_job="$(sbatch --parsable --array="${ARRAY}%4" \
  --dependency="aftercorr:${prepare_job}" \
  --export="${EXPORTS}" \
  "${REPO_ROOT}/slurm/run_paired_preprocessing_gpu.slurm")"
post_job="$(sbatch --parsable --array="${ARRAY}%8" \
  --dependency="aftercorr:${infer_job}" \
  --export="${EXPORTS},PIPELINE_STAGE=postprocess" \
  "${REPO_ROOT}/slurm/run_paired_preprocessing_cpu.slurm")"

echo "prepare_job=${prepare_job}"
echo "infer_job=${infer_job}"
echo "postprocess_job=${post_job}"
