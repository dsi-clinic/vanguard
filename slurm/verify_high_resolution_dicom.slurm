#!/usr/bin/env bash
#SBATCH --job-name=verify-hr-dicom
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/verify-hr-dicom-%j.out
#SBATCH --error=logs/slurm/verify-hr-dicom-%j.err

set -euo pipefail

: "${DESTINATION_ROOT:?Set DESTINATION_ROOT to the HR staging directory}"

cd "${DESTINATION_ROOT}"
sha256sum --strict --check SHA256SUMS
