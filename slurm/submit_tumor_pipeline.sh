#!/usr/bin/env bash
# Submit the UChicago high-resolution MAMA-MIA tumor-mask pipeline.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUN_ROOT="${RUN_ROOT:?set RUN_ROOT to a fresh output directory}"
INVENTORY="${INVENTORY:-/gpfs/data/karczmar-lab/vanguard/dce2d_internal_ultrafast_manifest/paired_hr_ufast_source_dicom/dicom_file_manifest.parquet}"
CASE_MANIFEST="${CASE_MANIFEST:-/gpfs/data/karczmar-lab/vanguard/dce2d_internal_ultrafast_manifest/paired_hr_ufast_source_dicom/paired_preprocessing_case_manifest.csv}"
COHORT_MANIFEST="${COHORT_MANIFEST:-/gpfs/data/karczmar-lab/vanguard/dce2d_internal_ultrafast_manifest/dce2d_internal_ultrafast_with_high_resolution_manifest.csv}"
MODEL_DIR="${MODEL_DIR:?set MODEL_DIR to the extracted MAMA-MIA model directory}"
NNUNET_ENV="${NNUNET_ENV:?set NNUNET_ENV to an nnU-Net 2.5 environment prefix}"
CENTERLINE_ROOT="${CENTERLINE_ROOT:-}"
PREP_CONCURRENCY="${PREP_CONCURRENCY:-12}"
GPU_CONCURRENCY="${GPU_CONCURRENCY:-8}"

[[ -f "${MODEL_DIR}/fold_0/checkpoint_final.pth" ]] || {
  echo "model is not extracted at ${MODEL_DIR}" >&2
  exit 2
}
[[ -x "${NNUNET_ENV}/bin/nnUNetv2_predict_from_modelfolder" ]] || {
  echo "nnU-Net predictor is missing from ${NNUNET_ENV}" >&2
  exit 2
}

mkdir -p "${REPO_ROOT}/logs/slurm" "${RUN_ROOT}"
N_CASES="$(python - "${CASE_MANIFEST}" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="") as stream:
    print(sum(1 for _ in csv.DictReader(stream)))
PY
)"
LAST_INDEX="$((N_CASES - 1))"
EXPORTS="ALL,REPO_ROOT=${REPO_ROOT},RUN_ROOT=${RUN_ROOT},INVENTORY=${INVENTORY},CASE_MANIFEST=${CASE_MANIFEST},COHORT_MANIFEST=${COHORT_MANIFEST},MODEL_DIR=${MODEL_DIR},NNUNET_ENV=${NNUNET_ENV},CENTERLINE_ROOT=${CENTERLINE_ROOT}"

cd "${REPO_ROOT}"
PREP_JOB="$(sbatch --parsable --array="0-${LAST_INDEX}%${PREP_CONCURRENCY}" \
  --export="${EXPORTS}" preprocessing/slurm/tumor_prepare_case.slurm)"
INDEX_JOB="$(sbatch --parsable --dependency="afterok:${PREP_JOB}" \
  --export="${EXPORTS}" preprocessing/slurm/tumor_index_cohort.slurm)"
INFER_JOB="$(sbatch --parsable --dependency="afterok:${INDEX_JOB}" \
  --array="0-${LAST_INDEX}%${GPU_CONCURRENCY}" --export="${EXPORTS}" \
  preprocessing/slurm/tumor_infer_case.slurm)"
FINAL_JOB="$(sbatch --parsable --dependency="afterok:${INFER_JOB}" \
  --export="${EXPORTS}" preprocessing/slurm/tumor_finalize_cohort.slurm)"

printf 'prepare=%s\nindex=%s\ninfer=%s\nfinalize=%s\n' \
  "${PREP_JOB}" "${INDEX_JOB}" "${INFER_JOB}" "${FINAL_JOB}" \
  | tee "${RUN_ROOT}/tumor_pipeline_jobs.txt"
