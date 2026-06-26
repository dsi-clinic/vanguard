#!/usr/bin/env bash
#SBATCH --job-name=duke001-skel-mp4
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --array=0-3
#SBATCH --output=logs/skel-mp4-%A-%a.out
#SBATCH --error=logs/skel-mp4-%A-%a.err
set -eo pipefail

# Render a rotating 3D mp4 of each version's vessel skeleton.  One array task per
# version.  Depends (afterok) on the skeleton-extraction job.

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
SKELETON="${EXP_ROOT}/skeletons/DUKE_001__${VERSION}/DUKE_001_skeleton_4d_exam_mask.npy"

cd "${PROJECT_ROOT}"

echo "Task ${SLURM_ARRAY_TASK_ID}  version: ${VERSION}"
echo "Skeleton: ${SKELETON}"

if [[ ! -f "${SKELETON}" ]]; then
  echo "ERROR: skeleton not found: ${SKELETON}" >&2
  exit 1
fi

python -u graph_extraction/render_skeleton_mp4.py "${SKELETON}"

echo "MP4 task ${SLURM_ARRAY_TASK_ID} (${VERSION}) complete: $(date)"
