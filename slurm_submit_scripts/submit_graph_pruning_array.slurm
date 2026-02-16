#!/usr/bin/env bash
#SBATCH --job-name=graph-prune-array
#SBATCH --partition=general
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/graph-prune-array-%A-%a.out
#SBATCH --error=logs/graph-prune-array-%A-%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ruochun@duke.edu

set -eo pipefail

# --- Activate your environment ---
source ~/.bashrc || true
eval "$(micromamba shell hook -s bash)"
micromamba activate vanguard

# Prevent libxml2 deactivate hook from failing under `set -u`
export xml_catalog_files_libxml2=""

# --- Make logs/temp unique per job ---
mkdir -p logs
export TMPDIR="/net/projects2/vanguard/tmp/graph_prune_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${TMPDIR}"

cleanup() {
    echo "Cleaning up temporary directory: ${TMPDIR}"
    rm -rf "${TMPDIR}" || true
}
trap cleanup EXIT

# --- Optimize thread/process settings ---
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1

# --- Configuration ---
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
INPUT_DIR="${INPUT_DIR:-/net/projects2/vanguard/vessel_segmentations}"
OUTPUT_DIR="${OUTPUT_DIR:-/net/projects2/vanguard/graph_pruning_outdir}"
PATTERN="${PATTERN:-*_vessel_segmentation.npy}"
THRESHOLD="${THRESHOLD:-0.5}"
RECURSIVE="${RECURSIVE:-0}"

cd "${PROJECT_ROOT}"

TASK_ID=${SLURM_ARRAY_TASK_ID}
FILES_PER_TASK=${FILES_PER_TASK:-40}
FILE_START=$((TASK_ID * FILES_PER_TASK))
FILE_END=$((FILE_START + FILES_PER_TASK - 1))

RECURSIVE_FLAG=""
if [[ "${RECURSIVE}" == "1" ]]; then
  RECURSIVE_FLAG="--recursive"
fi

echo "========================================="
echo "Graph Pruning Array Job Started"
echo "========================================="
echo "Array Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "Input dir: ${INPUT_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Pattern: ${PATTERN}"
echo "Threshold: ${THRESHOLD}"
echo "File range: ${FILE_START}-${FILE_END}"
echo "Date: $(date)"
echo "========================================="

python -u "${PROJECT_ROOT}/graph_pruning_centerline_extraction/batch_graph_pruning_centerlines.py" \
  --input-dir  "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --pattern    "${PATTERN}" \
  ${RECURSIVE_FLAG} \
  --threshold  "${THRESHOLD}" \
  --file-start ${FILE_START} \
  --file-end   ${FILE_END} \
  --resume

echo ""
echo "========================================="
echo "Array Task ${SLURM_ARRAY_TASK_ID} Completed"
echo "Date: $(date)"
echo "========================================="
