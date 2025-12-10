#!/usr/bin/env bash
#SBATCH --job-name=centerline-extract
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/centerline-extract-%j.out
#SBATCH --error=logs/centerline-extract-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ruochun@duke.edu

set -euo pipefail

# Activate environment
source ~/.bashrc || true
eval "$(micromamba shell hook -s bash)"
micromamba activate vmtk

# Set up
mkdir -p logs
export PYTHONUNBUFFERED=1

echo "========================================="
echo "Centerline Extraction Job Started"
echo "========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "Date: $(date)"
echo "========================================="

# Run batch centerline extraction
# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
python batch_processing/batch_extract_centerlines.py

echo ""
echo "========================================="
echo "Job Completed Successfully"
echo "Date: $(date)"
echo "========================================="

