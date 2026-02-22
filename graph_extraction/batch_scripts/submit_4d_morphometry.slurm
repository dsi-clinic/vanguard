#!/usr/bin/env bash
#SBATCH --job-name=4d-morphometry
#SBATCH --partition=general
#SBATCH --cpus-per-task=8             # CPU-only (no GPU): skeleton/morphometry
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --array=0-9999                # Override when submitting: sbatch --array=0-N submit_4d_morphometry.slurm
#SBATCH --output=logs/4d-morphometry-%A-%a.out
#SBATCH --error=logs/4d-morphometry-%A-%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lunad@uchicago.edu

# Run batch 4D morphometry extraction on the cluster.
# Each array task processes one study. To submit with correct array size:
#   ./submit_4d_array.sh   # discovers study count and submits
# Or manually:
#   sbatch --array=0-99 graph_extraction/batch_scripts/submit_4d_morphometry.slurm

set -eo pipefail

# --- Activate environment ---
# Ensure HOME is set (some CPU nodes have minimal env; GPU nodes tend to be fine)
export HOME="${HOME:-/home/$(whoami)}"
source ~/.bashrc || true
eval "$(micromamba shell hook -s bash)"
micromamba activate vanguard

# Prevent libxml2 deactivate hook from failing
export xml_catalog_files_libxml2=""

# --- Project paths ---
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
INPUT_DIR="${INPUT_DIR:-/net/projects2/vanguard/vessel_segmentations}"
OUTPUT_DIR="${OUTPUT_DIR:-/net/projects2/vanguard/report/4d_morphometry}"

# --- Temp and logs ---
mkdir -p logs
mkdir -p "${OUTPUT_DIR}"
export TMPDIR="/net/projects2/vanguard/tmp/4d_morphometry_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${TMPDIR}"

cleanup() {
    rm -rf "${TMPDIR}" || true
}
trap cleanup EXIT

# --- Thread settings ---
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
