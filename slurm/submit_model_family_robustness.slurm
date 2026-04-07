#!/bin/bash
#SBATCH --job-name=modelfam-robust
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out

set -euo pipefail

CONFIG="${CONFIG:-../configs/model_family_robustness.yaml}"
OUT_ROOT="${OUT_ROOT:-../experiments/model_family_robustness_ispy2}"

echo "CONFIG=${CONFIG}"
echo "OUT_ROOT=${OUT_ROOT}"

micromamba run -n vanguard python ../run_model_family_robustness.py \
  --config "${CONFIG}" \
  --outdir "${OUT_ROOT}"
