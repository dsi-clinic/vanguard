#!/usr/bin/env bash
#SBATCH --job-name=duke001-skel
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/skel-%A-%a.out
#SBATCH --error=logs/skel-%A-%a.err
set -eo pipefail

# Build the 4D vessel skeleton + centerline manifold + coverage MIP for one
# version (one array task per version), from that version's 5-timepoint vessel
# masks.  This is the existing graph-extraction pipeline; it runs on CPU.

source ~/.bashrc || true
eval "$(micromamba shell hook -s bash)"
micromamba activate vanguard
export xml_catalog_files_libxml2=""
export PYTHONUNBUFFERED=1

mkdir -p logs

VERSIONS=(version1_z2.0mm version2_z1.4mm version3_z1.0mm version4_original)
VERSION="${VERSIONS[$SLURM_ARRAY_TASK_ID]}"

PROJECT_ROOT="$HOME/vanguard"
EXP_ROOT="$HOME/vanguard_qc_pngs/resampling_experiment"
# Segmentation layout is <out>/<VERSION>/DUKE/DUKE_001/images/*.npz, and the
# pipeline expects <input-dir>/<case-id>/images/*.npz, so input-dir is .../DUKE.
INPUT_DIR="${EXP_ROOT}/segmentations/${VERSION}/DUKE"
# Encode the version in the output dir name so the mp4 title/filename (which is
# derived from the parent directory name) is unique per version.
OUTPUT_DIR="${EXP_ROOT}/skeletons/DUKE_001__${VERSION}"
mkdir -p "${OUTPUT_DIR}"

cd "${PROJECT_ROOT}"

echo "========================================="
echo "Task ${SLURM_ARRAY_TASK_ID}  version: ${VERSION}"
echo "Node: $(hostname)   Date: $(date)"
echo "Input dir : ${INPUT_DIR}"
echo "Output dir: ${OUTPUT_DIR}"
echo "========================================="

python -u graph_extraction/run_skeleton_processing.py \
    --case-id  DUKE_001 \
    --input-dir  "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}"

echo "Skeleton task ${SLURM_ARRAY_TASK_ID} (${VERSION}) complete: $(date)"
