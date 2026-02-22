#!/usr/bin/env bash
#SBATCH --job-name=4d-morphometry
#SBATCH --partition=general
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --array=0-9999
#SBATCH --output=logs/4d-morphometry-%A-%a.out
#SBATCH --error=logs/4d-morphometry-%A-%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lunad@uchicago.edu

# Run batch 4D morphometry extraction on the cluster.
# Each array task processes one study via --study-index.
# Submit via submit_4d_array.sh (discovers study count and chunks); or manually:
#   sbatch --array=0-99 --export=INPUT_DIR=...,OUTPUT_DIR=...,STUDY_OFFSET=0 submit_4d_morphometry.slurm

set -eo pipefail

export HOME="${HOME:-/home/$(whoami)}"
source ~/.bashrc || true
eval "$(micromamba shell hook -s bash)"
micromamba activate vanguard

export xml_catalog_files_libxml2=""

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
INPUT_DIR="${INPUT_DIR:-/net/projects2/vanguard/vessel_segmentations}"
OUTPUT_DIR="${OUTPUT_DIR:-/net/projects2/vanguard/report/4d_morphometry}"

mkdir -p logs
mkdir -p "${OUTPUT_DIR}"
export TMPDIR="/net/projects2/vanguard/tmp/4d_morphometry_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${TMPDIR}"

cleanup() {
    rm -rf "${TMPDIR}" || true
}
trap cleanup EXIT

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1

TASK_ID=${SLURM_ARRAY_TASK_ID}
STUDY_INDEX=$((${STUDY_OFFSET:-0} + TASK_ID))

echo "========================================="
echo "4D Morphometry Array Task"
echo "========================================="
echo "Array Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array Task ID: ${TASK_ID}"
echo "Node: $(hostname)"
echo "Input: ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Date: $(date)"
echo "========================================="

cd "${PROJECT_ROOT}"
python -u graph_extraction/batch_process_4d.py \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --study-index ${STUDY_INDEX} \
  --skip-existing \
  --quiet

echo ""
echo "Task ${TASK_ID} completed at $(date)"
