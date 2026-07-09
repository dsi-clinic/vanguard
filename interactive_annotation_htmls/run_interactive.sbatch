#!/bin/bash
#SBATCH --job-name=duke_qc_html
#SBATCH --partition=tier2q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Resolve the repo root whether this is submitted via sbatch or run directly.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p "${REPO_ROOT}/logs"
cd "${REPO_ROOT}"

# Activate the project environment (see AGENTS.md).
source "${HOME}/.bashrc" || true
if command -v micromamba >/dev/null 2>&1; then
  set +u
  eval "$(micromamba shell hook -s bash)"
  micromamba activate vanguard
  set -u
fi

# Load local path configuration if present (see duke_qc.env.example).
ENV_FILE="${REPO_ROOT}/interactive_annotation_htmls/duke_qc.env"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

# A single case id (e.g. DUKE_693) or ALL to render every annotated case.
CASE="${1:-DUKE_693}"

echo "host=$(hostname)  start=$(date)  case=${CASE}  repo=${REPO_ROOT}"
python interactive_annotation_htmls/duke_qc_interactive.py "${CASE}"
echo "end=$(date)"
