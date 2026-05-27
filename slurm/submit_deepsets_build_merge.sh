#!/usr/bin/env bash
# Submit Deep Sets dataset build without training: either one long serial job or
# a sharded Slurm array plus a manifest merge dependency.
#
# Typical use (notebook expects manifest under results/deepsets/):
#
#   cd /path/to/vanguard
#   ./slurm/submit_deepsets_build_merge.sh
#
# Or override paths and resources:
#
#   CONFIG=configs/deepsets_ispy2.yaml OUT_ROOT=results/deepsets \
#     BUILD_SHARDS=8 BUILD_CPUS=8 BUILD_MEM=64G BUILD_TIME=24:00:00 \
#     ./slurm/submit_deepsets_build_merge.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG="${CONFIG:-${REPO_ROOT}/configs/deepsets_ispy2.yaml}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/results/deepsets}"
PARTITION="${PARTITION:-general}"

_resolve_out_root() {
  local out="$1"
  if [[ "${out}" != /* ]]; then
    (cd "${SCRIPT_DIR}" && mkdir -p "${out}" && cd "${out}" && pwd)
  else
    mkdir -p "${out}"
    (cd "${out}" && pwd)
  fi
}
OUT_ROOT="$(_resolve_out_root "${OUT_ROOT}")"

BUILD_CPUS="${BUILD_CPUS:-8}"
BUILD_MEM="${BUILD_MEM:-32G}"
BUILD_TIME="${BUILD_TIME:-4:00:00}"
BUILD_SHARDS="${BUILD_SHARDS:-8}"

MERGE_CPUS="${MERGE_CPUS:-2}"
MERGE_MEM="${MERGE_MEM:-8G}"
MERGE_TIME="${MERGE_TIME:-01:00:00}"

mkdir -p "${REPO_ROOT}/logs"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found: ${CONFIG}" >&2
  exit 2
fi

if [[ "${BUILD_SHARDS}" -lt 1 ]]; then
  echo "BUILD_SHARDS must be >= 1" >&2
  exit 2
fi

if [[ "${BUILD_SHARDS}" -eq 1 ]]; then
  BUILD_JOB_ID="$(sbatch --parsable \
    --partition="${PARTITION}" \
    --cpus-per-task="${BUILD_CPUS}" \
    --mem="${BUILD_MEM}" \
    --time="${BUILD_TIME}" \
    --job-name="deepsets-build-serial" \
    --output="${REPO_ROOT}/logs/deepsets-build-serial-%j.out" \
    --error="${REPO_ROOT}/logs/deepsets-build-serial-%j.err" \
    --export=ALL,MODE=build-single,REPO_ROOT="${REPO_ROOT}",CONFIG="${CONFIG}",OUT_ROOT="${OUT_ROOT}" \
    "${SCRIPT_DIR}/deepsets_job.slurm")"

  cat <<MSG
Submitted Deep Sets dataset build (serial, BUILD_SHARDS=1):
  config       : ${CONFIG}
  out_root     : ${OUT_ROOT}
  build_job_id : ${BUILD_JOB_ID}

When finished, manifest path:
  ${OUT_ROOT}/deepsets_manifest.csv

Monitor logs:
  tail -f "${REPO_ROOT}/logs/deepsets-build-serial-${BUILD_JOB_ID}.out"

Other commands:
  squeue -u "\$USER" -j ${BUILD_JOB_ID}
  sacct -j ${BUILD_JOB_ID} --format=JobIDRaw,State,Elapsed,ExitCode -n -P
MSG
  exit 0
fi

BUILD_ARRAY_JOB_ID="$(sbatch --parsable \
  --partition="${PARTITION}" \
  --array="0-$((BUILD_SHARDS - 1))" \
  --cpus-per-task="${BUILD_CPUS}" \
  --mem="${BUILD_MEM}" \
  --time="${BUILD_TIME}" \
  --job-name="deepsets-build-only" \
  --output="${REPO_ROOT}/logs/deepsets-build-only-%A-%a.out" \
  --error="${REPO_ROOT}/logs/deepsets-build-only-%A-%a.err" \
  --export=ALL,MODE=build,REPO_ROOT="${REPO_ROOT}",CONFIG="${CONFIG}",OUT_ROOT="${OUT_ROOT}",NUM_SHARDS="${BUILD_SHARDS}" \
  "${SCRIPT_DIR}/deepsets_job.slurm")"

MERGE_JOB_ID="$(sbatch --parsable \
  --partition="${PARTITION}" \
  --dependency="afterok:${BUILD_ARRAY_JOB_ID}" \
  --cpus-per-task="${MERGE_CPUS}" \
  --mem="${MERGE_MEM}" \
  --time="${MERGE_TIME}" \
  --job-name="deepsets-merge-only" \
  --output="${REPO_ROOT}/logs/deepsets-merge-only-%j.out" \
  --error="${REPO_ROOT}/logs/deepsets-merge-only-%j.err" \
  --export=ALL,MODE=merge,REPO_ROOT="${REPO_ROOT}",OUT_ROOT="${OUT_ROOT}" \
  "${SCRIPT_DIR}/deepsets_job.slurm")"

cat <<MSG
Submitted Deep Sets dataset build (sharded array + manifest merge):
  config             : ${CONFIG}
  out_root           : ${OUT_ROOT}
  build_array_job_id : ${BUILD_ARRAY_JOB_ID}
  merge_job_id       : ${MERGE_JOB_ID}

When merge finishes:
  ${OUT_ROOT}/deepsets_manifest.csv

Monitor:
  squeue -j ${BUILD_ARRAY_JOB_ID},${MERGE_JOB_ID}
  sacct -j ${BUILD_ARRAY_JOB_ID},${MERGE_JOB_ID} --format=JobIDRaw,State,Elapsed,ExitCode -n -P

Build logs pattern:
  ${REPO_ROOT}/logs/deepsets-build-only-<array_job_id>_*.out
Merge log:
  ${REPO_ROOT}/logs/deepsets-merge-only-<merge_job_id>.out
MSG
