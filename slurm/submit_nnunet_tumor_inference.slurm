#!/bin/bash
#SBATCH --job-name=nnunet-tumor
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

: "${INPUT_DIR:?set INPUT_DIR to a folder of nnU-Net inputs named CASE_0000.nii.gz}"

REPO_ROOT="${SLURM_SUBMIT_DIR:-${REPO_ROOT:-$PWD}}"
RUN_ID="${SLURM_JOB_ID:-manual}"
OUTPUT_DIR="${OUTPUT_DIR:-results/nnunet_tumor_inference/${RUN_ID}}"
VANGUARD_NNUNET_ENV="${VANGUARD_NNUNET_ENV:-/ess/scratch/scratch1/aakrithiram/micromamba/envs/vanguard-nnunet}"
NNUNET_ROOT="${NNUNET_ROOT:-/ess/scratch/scratch1/aakrithiram/nnunet_vanguard}"
MODEL_DIR="${MODEL_DIR:-${NNUNET_ROOT}/results/full_image_dce_mri_tumor_segmentation}"
DEVICE="${DEVICE:-cuda}"
FOLDS="${FOLDS:-0 1 2 3 4}"
NPP="${NPP:-${SLURM_CPUS_PER_TASK:-4}}"
NPS="${NPS:-${SLURM_CPUS_PER_TASK:-4}}"

cd "${REPO_ROOT}"
mkdir -p logs/slurm "${OUTPUT_DIR}"

set +u
source ~/.bashrc || true
if command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook -s bash)"
  micromamba activate "${VANGUARD_NNUNET_ENV}"
fi
set -u

export nnUNet_raw="${NNUNET_ROOT}/raw"
export nnUNet_preprocessed="${NNUNET_ROOT}/preprocessed"
export nnUNet_results="${NNUNET_ROOT}/results"
export LD_LIBRARY_PATH="${VANGUARD_NNUNET_ENV}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

echo "=== nnU-Net tumor inference ==="
echo "Job: ${SLURM_JOB_ID:-manual}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Repo: ${REPO_ROOT}"
echo "Env: ${VANGUARD_NNUNET_ENV}"
echo "Input: ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Model: ${MODEL_DIR}"
echo "Device: ${DEVICE}"
echo "Folds: ${FOLDS}"
which python
python - <<'PY'
import sys
import torch
print(sys.executable)
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
PY

read -r -a FOLD_ARGS <<< "${FOLDS}"
cmd=(
  nnUNetv2_predict_from_modelfolder
  -i "${INPUT_DIR}"
  -o "${OUTPUT_DIR}"
  -m "${MODEL_DIR}"
  -f "${FOLD_ARGS[@]}"
  -device "${DEVICE}"
  -npp "${NPP}"
  -nps "${NPS}"
  --disable_progress_bar
)

echo "Command: ${cmd[*]}"
"${cmd[@]}"
echo "=== Finished: $(date) ==="
