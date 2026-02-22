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
export TMPDIR="/tmp/graph_prune_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
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
OUTPUT_DIR="${OUTPUT_DIR:-/net/projects2/vanguard/centerlines_4d}"
PATTERN="${PATTERN:-*_vessel_segmentation.npz}"
THRESHOLD_LOW="${THRESHOLD_LOW:-0.5}"
THRESHOLD_HIGH="${THRESHOLD_HIGH:-0.85}"
NPY_CHANNEL="${NPY_CHANNEL:-1}"

cd "${PROJECT_ROOT}"

TASK_ID=${SLURM_ARRAY_TASK_ID}
STUDIES_PER_TASK=${STUDIES_PER_TASK:-1}
STUDY_START=$((TASK_ID * STUDIES_PER_TASK))
STUDY_END=$((STUDY_START + STUDIES_PER_TASK - 1))

echo "========================================="
echo "4D Skeleton Processing Array Job Started"
echo "========================================="
echo "Array Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "Input dir: ${INPUT_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Pattern: ${PATTERN}"
echo "Threshold low: ${THRESHOLD_LOW}"
echo "Threshold high: ${THRESHOLD_HIGH}"
echo "NPY channel: ${NPY_CHANNEL}"
echo "Study range: ${STUDY_START}-${STUDY_END}"
echo "Date: $(date)"
echo "========================================="

# Get list of study IDs and process the ones for this task
python - <<PY
import sys
from pathlib import Path

input_dir = Path("${INPUT_DIR}")
pattern = "${PATTERN}"

# Extract unique study IDs
files = list(input_dir.rglob(pattern))
study_ids = set()
for f in files:
    parts = f.stem.split("_")
    if len(parts) >= 3:
        study_id = "_".join(parts[:2])
        study_ids.add(study_id)

# Sort for consistent ordering
study_ids = sorted(study_ids)

# Get studies for this task
start = ${STUDY_START}
end = ${STUDY_END}
for i, study_id in enumerate(study_ids):
    if start <= i <= end:
        print(study_id)
PY

# Read study IDs and process each one
readarray -t STUDY_IDS < <(python - <<PY
import sys
from pathlib import Path

input_dir = Path("${INPUT_DIR}")
pattern = "${PATTERN}"

files = list(input_dir.rglob(pattern))
study_ids = set()
for f in files:
    parts = f.stem.split("_")
    if len(parts) >= 3:
        study_id = "_".join(parts[:2])
        study_ids.add(study_id)

study_ids = sorted(study_ids)
start = ${STUDY_START}
end = ${STUDY_END}
for i, study_id in enumerate(study_ids):
    if start <= i <= end:
        print(study_id)
PY
)

echo "Processing ${#STUDY_IDS[@]} study IDs for this task"

for STUDY_ID in "${STUDY_IDS[@]}"; do
    echo ""
    echo "--- Processing study: ${STUDY_ID} ---"
    
    python -u "${PROJECT_ROOT}/graph_extraction/run_skeleton_processing.py" 4d \
        --input-dir "${INPUT_DIR}" \
        --study-id "${STUDY_ID}" \
        --output-dir "${OUTPUT_DIR}/${STUDY_ID}" \
        --npy-channel ${NPY_CHANNEL} \
        --threshold-low ${THRESHOLD_LOW} \
        --threshold-high ${THRESHOLD_HIGH}
    
    if [[ $? -eq 0 ]]; then
        echo "✓ Successfully processed ${STUDY_ID}"
    else
        echo "✗ Failed to process ${STUDY_ID}"
    fi
done

echo ""
echo "========================================="
echo "Array Task ${SLURM_ARRAY_TASK_ID} Completed"
echo "Date: $(date)"
echo "========================================="
