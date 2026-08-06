#!/usr/bin/env bash
#SBATCH --job-name=uchic-tumor-infer
#SBATCH --account=karczmar-lab
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x-%A-%a.out
#SBATCH --error=logs/slurm/%x-%A-%a.err

export PYTHONNOUSERSITE=1

eval "$(micromamba shell hook -s bash)"
micromamba activate vanguard
set -euo pipefail

: "${REPO_ROOT:?set REPO_ROOT}"
: "${RUN_ROOT:?set RUN_ROOT}"
: "${MODEL_DIR:?set MODEL_DIR}"
: "${NNUNET_ENV:?set NNUNET_ENV}"

cd "${REPO_ROOT}"
VANGUARD_PYTHON="$(which python)"
EXAM_ID="$(${VANGUARD_PYTHON} - "${RUN_ROOT}/tumor_case_manifest.csv" "${SLURM_ARRAY_TASK_ID}" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="") as stream:
    rows = sorted(csv.DictReader(stream), key=lambda row: row["exam_id"])
print(rows[int(sys.argv[2])]["exam_id"])
PY
)"
INPUT_DIR="${RUN_ROOT}/work/${EXAM_ID}/nnunet_postcontrast_inputs"
OUTPUT_DIR="${RUN_ROOT}/work/${EXAM_ID}/nnunet_postcontrast_predictions"

mkdir -p "${OUTPUT_DIR}"
shopt -s nullglob
INPUTS=("${INPUT_DIR}"/*_0000.nii.gz)
if (( ${#INPUTS[@]} == 0 )); then
  echo "no nnU-Net inputs found for ${EXAM_ID}" >&2
  exit 2
fi
missing_prediction=0
for input_path in "${INPUTS[@]}"; do
  prediction_name="$(basename "${input_path}" _0000.nii.gz).nii.gz"
  [[ -f "${OUTPUT_DIR}/${prediction_name}" ]] || missing_prediction=1
done
if (( missing_prediction == 0 )); then
  echo "reusing ${#INPUTS[@]} complete predictions for ${EXAM_ID}"
  exit 0
fi
export nnUNet_raw="${RUN_ROOT}/nnunet_runtime/raw"
export nnUNet_preprocessed="${RUN_ROOT}/nnunet_runtime/preprocessed"
export nnUNet_results="$(dirname "${MODEL_DIR}")"
mkdir -p "${nnUNet_raw}" "${nnUNet_preprocessed}"
export LD_LIBRARY_PATH="${NNUNET_ENV}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "exam_id=${EXAM_ID}"
echo "python=${NNUNET_ENV}/bin/python"
echo "model=${MODEL_DIR}"
"${NNUNET_ENV}/bin/python" -c \
  'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'

"${NNUNET_ENV}/bin/nnUNetv2_predict_from_modelfolder" \
  -i "${INPUT_DIR}" \
  -o "${OUTPUT_DIR}" \
  -m "${MODEL_DIR}" \
  -f 0 1 2 3 4 \
  -device cuda \
  -npp "${SLURM_CPUS_PER_TASK:-4}" \
  -nps "${SLURM_CPUS_PER_TASK:-4}" \
  --disable_progress_bar
