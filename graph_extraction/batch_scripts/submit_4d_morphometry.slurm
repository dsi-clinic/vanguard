#!/usr/bin/env bash
#SBATCH --job-name=4d-morphometry
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --array=0-9999
#SBATCH --output=logs/4d-morphometry-%A-%a.out
#SBATCH --error=logs/4d-morphometry-%A-%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lunad@uchicago.edu

# Run batch 4D morphometry extraction on the cluster.
# Each array task processes STUDIES_PER_TASK studies in parallel via --study-range.
# Submit via submit_4d_array.sh (discovers study count and chunks); or manually:
#   sbatch --array=0-49 --export=INPUT_DIR=...,OUTPUT_DIR=...,STUDY_OFFSET=0,STUDIES_PER_TASK=4 submit_4d_morphometry.slurm

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
STUDIES_PER_TASK=${STUDIES_PER_TASK:-4}
STUDY_OFFSET=${STUDY_OFFSET:-0}
CHUNK_END=${CHUNK_END:-$(( STUDY_OFFSET + 1 ))}
TASK_OFFSET=${TASK_OFFSET:-0}

STUDY_START=$(( STUDY_OFFSET + TASK_ID * STUDIES_PER_TASK ))
STUDY_END=$(( STUDY_START + STUDIES_PER_TASK ))
if [ "${STUDY_END}" -gt "${CHUNK_END}" ]; then
  STUDY_END=${CHUNK_END}
fi
CHUNK_ID=$(( TASK_OFFSET + TASK_ID ))

echo "========================================="
echo "4D Morphometry Array Task"
echo "========================================="
echo "Array Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array Task ID: ${TASK_ID}"
echo "Studies: ${STUDY_START}-${STUDY_END} (${STUDIES_PER_TASK} workers)"
echo "Node: $(hostname)"
echo "Input: ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Date: $(date)"
echo "========================================="

cd "${PROJECT_ROOT}"
python -u graph_extraction/batch_process_4d.py \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --study-range ${STUDY_START} ${STUDY_END} \
  --studies-per-task ${STUDIES_PER_TASK} \
  --chunk-id ${CHUNK_ID} \
  --skip-existing \
  --quiet

echo ""
echo "Task ${TASK_ID} completed at $(date)"
