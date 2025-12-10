#!/bin/bash
#SBATCH --job-name=vtp-to-json
#SBATCH --time=02:00:00
#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/vtp-to-json-%j.out
#SBATCH --error=logs/vtp-to-json-%j.err

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source ~/.bashrc
eval "$(micromamba shell hook -s bash)"
micromamba activate vanguard-gpu

cd "${PROJECT_ROOT}"

python3 batch_processing/batch_convert_vtp_to_json.py



