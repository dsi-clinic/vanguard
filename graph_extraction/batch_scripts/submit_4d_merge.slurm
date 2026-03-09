#!/usr/bin/env bash
#SBATCH --job-name=4d-morphometry-merge
#SBATCH --partition=general
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0:30:00
#SBATCH --output=logs/4d-morphometry-merge-%j.out
#SBATCH --error=logs/4d-morphometry-merge-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lunad@uchicago.edu

# Merge manifest_task_*.json into manifest.json after array job completes.
# Submit via submit_4d_array.sh (after array chunks finish); or manually:
#   sbatch --dependency=afterok:ARRAY_JOB_ID --export=OUTPUT_DIR=... submit_4d_merge.slurm

set -eo pipefail

export HOME="${HOME:-/home/$(whoami)}"
source ~/.bashrc || true
eval "$(micromamba shell hook -s bash)"
micromamba activate vanguard

export xml_catalog_files_libxml2=""

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OUTPUT_DIR="${OUTPUT_DIR:-/net/projects2/vanguard/report/4d_morphometry}"

mkdir -p logs

echo "Merging task manifests in ${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"
python -u graph_extraction/batch_process_4d.py \
  --output-dir "${OUTPUT_DIR}" \
  --merge-manifests

echo "Merge completed at $(date)"
