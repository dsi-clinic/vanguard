#!/usr/bin/env bash
#SBATCH --job-name=uc_skel_html
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/uc-skel-html-%j.out
#SBATCH --error=logs/uc-skel-html-%j.err
set -eo pipefail

# Render the UChicago interactive 3D vessel-centerline QC scenes (one self-
# contained HTML per case). Skeletonization is CPU-bound and takes a few minutes
# per case, hence tier1q rather than the head node.
#
# Requires the segmentation runs to have finished first:
#   external (real breast masks) -> --external-dir
#   baseline (model breast masks) -> --baseline-dir
#
# Submit from the repo root; all args after the script are passed to the CLI:
#   PROJECT_ROOT="$(pwd)" sbatch data_viz/run_uchicago_skeleton_interactive.sbatch \
#     --sub-source simbiosys uch_nac

source ~/.bashrc || true
eval "$(micromamba shell hook -s bash)"
micromamba activate vanguard
export xml_catalog_files_libxml2=""

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT must be set, e.g. PROJECT_ROOT=\$(pwd) sbatch ... from the repo root}"
cd "${PROJECT_ROOT}"
mkdir -p logs

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4

echo "=== UChicago interactive skeleton HTML: $(date) ==="
echo "Node: $(hostname)  args: $*"

python -u "${PROJECT_ROOT}/data_viz/uchicago_skeleton_interactive.py" "$@"

echo "=== Finished: $(date) ==="
