#!/usr/bin/env bash
#SBATCH --job-name=skeleton-mp4
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/skeleton-mp4-%A-%a.out
#SBATCH --error=logs/skeleton-mp4-%A-%a.err

set -euo pipefail

cd "${REPO_ROOT}"
set +u
source ~/.bashrc || true
if command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook -s bash)"
  micromamba activate vanguard
fi
set -u

LINE_NO=$((SLURM_ARRAY_TASK_ID + 1))
SKELETON_PATH="$(sed -n "${LINE_NO}p" "${SKELETON_LIST}")"

if [[ -z "${SKELETON_PATH}" ]]; then
  echo "ERROR: no path at line ${LINE_NO} of ${SKELETON_LIST}"
  exit 1
fi

echo "Task ${SLURM_ARRAY_TASK_ID}: ${SKELETON_PATH}"
python graph_extraction/render_skeleton_mp4.py "${SKELETON_PATH}"
