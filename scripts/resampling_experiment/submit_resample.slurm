#!/usr/bin/env bash
#SBATCH --job-name=duke001-resample
#SBATCH --partition=express
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/resample-%j.out
#SBATCH --error=logs/resample-%j.err
set -eo pipefail

# STEP 1: build the four z-spacing versions of DUKE_001 (CPU-only, lightweight).
# We run this through Slurm rather than on the head node because resampling 20
# volumes (4 versions x 5 timepoints) can exceed the head-node time/RAM rules.

source ~/.bashrc || true
eval "$(micromamba shell hook -s bash)"
micromamba activate vanguard
export xml_catalog_files_libxml2=""   # avoid a known libxml2 deactivate-hook crash

mkdir -p logs

# Slurm copies the batch script to a spool dir on the compute node, so we cannot
# derive the script location from BASH_SOURCE. Use a fixed absolute path.
SCRIPT_DIR="$HOME/vanguard/scripts/resampling_experiment"
OUTPUT_ROOT="${OUTPUT_ROOT:-$HOME/vanguard_qc_pngs/resampling_experiment/volumes}"

echo "Node: $(hostname)   Date: $(date)"
echo "Output root: ${OUTPUT_ROOT}"

python -u "${SCRIPT_DIR}/resample_duke001.py" \
    --output-root "${OUTPUT_ROOT}"

echo "Resampling job done: $(date)"
