#!/bin/bash
#SBATCH --job-name=issue118-baseline
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

# Under Slurm, $0 may point at a copy under /var/spool/slurmd — use submit cwd.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
  REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fi
cd "${REPO_ROOT}"

# Non-interactive Slurm jobs often lack micromamba on PATH; match submit_feature_ablation.slurm.
source "${HOME}/.bashrc" || true
if command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook -s bash)"
  micromamba activate vanguard
fi

LR_CONFIG="${LR_CONFIG:-configs/issue118_baseline_arms_lr.yaml}"
XGB_CONFIG="${XGB_CONFIG:-configs/issue118_baseline_arms_xgb.yaml}"
LR_OUT="${LR_OUT:-experiments/issue118_baseline_arms_lr}"
XGB_OUT="${XGB_OUT:-experiments/issue118_baseline_arms_xgb}"
MERGED_DIR="${MERGED_DIR:-experiments/issue118_baseline_arms_merged}"

echo "REPO_ROOT=${REPO_ROOT}"
echo "LR_CONFIG=${LR_CONFIG} LR_OUT=${LR_OUT}"
echo "XGB_CONFIG=${XGB_CONFIG} XGB_OUT=${XGB_OUT}"

python run_ablation_matrix.py \
  --config "${LR_CONFIG}" \
  --outdir "${LR_OUT}"

python run_ablation_matrix.py \
  --config "${XGB_CONFIG}" \
  --outdir "${XGB_OUT}"

mkdir -p "${MERGED_DIR}"
python merge_issue118_summaries.py \
  --lr-summary "${LR_OUT}/ablation_summary.csv" \
  --xgb-summary "${XGB_OUT}/ablation_summary.csv" \
  --out "${MERGED_DIR}/issue118_baseline_arms_combined.csv"

echo "Merged summary: ${MERGED_DIR}/issue118_baseline_arms_combined.csv"
