#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG="${CONFIG:-${REPO_ROOT}/configs/deepsets_ispy2.yaml}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/experiments/deepsets_ispy2}"
PARTITION="${PARTITION:-general}"

# Resolve OUT_ROOT to an absolute path. Relative paths are interpreted from
# slurm/ (SCRIPT_DIR) so examples like OUT_ROOT=../experiments/foo match the
# directory used when the submit script runs "cd slurm" — compute jobs use
# cwd REPO_ROOT, so passing a bare ../experiments/... CONFIG breaks there.
_resolve_out_root() {
  local out="$1"
  if [[ "${out}" != /* ]]; then
    ( cd "${SCRIPT_DIR}" && mkdir -p "${out}" && cd "${out}" && pwd )
  else
    mkdir -p "${out}"
    ( cd "${out}" && pwd )
  fi
}
OUT_ROOT="$(_resolve_out_root "${OUT_ROOT}")"

BUILD_CPUS="${BUILD_CPUS:-8}"
BUILD_MEM="${BUILD_MEM:-64G}"
BUILD_TIME="${BUILD_TIME:-04:00:00}"
BUILD_SHARDS="${BUILD_SHARDS:-8}"

TRAIN_CPUS="${TRAIN_CPUS:-8}"
TRAIN_MEM="${TRAIN_MEM:-32G}"
TRAIN_TIME="${TRAIN_TIME:-08:00:00}"
TRAIN_GPUS="${TRAIN_GPUS:-0}"

MERGE_CPUS="${MERGE_CPUS:-2}"
MERGE_MEM="${MERGE_MEM:-8G}"
MERGE_TIME="${MERGE_TIME:-00:30:00}"

FORCE_BUILD="${FORCE_BUILD:-0}"
FORCE_MERGE="${FORCE_MERGE:-0}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
DEEPSETS_CACHE_ROOT="${DEEPSETS_CACHE_ROOT:-${OUT_ROOT}/../.deepsets_build_cache}"
DEEPSETS_BUILD_WORKERS="${DEEPSETS_BUILD_WORKERS:-${BUILD_CPUS}}"

mkdir -p "${REPO_ROOT}/logs"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found: ${CONFIG}" >&2
  exit 2
fi

PROFILE_JSON="$(python "${REPO_ROOT}/deepsets/deepsets_pipeline_profile.py" --config "${CONFIG}")"
if [[ "${BUILD_TIME}" == "04:00:00" ]]; then
  BUILD_TIME="$(python - <<PY
import json
print(json.loads(${PROFILE_JSON@Q})["build_time"])
PY
)"
fi
if [[ "${BUILD_SHARDS}" == "8" ]]; then
  BUILD_SHARDS="$(python - <<PY
import json
print(json.loads(${PROFILE_JSON@Q})["build_shards"])
PY
)"
fi
if [[ "${TRAIN_TIME}" == "08:00:00" ]]; then
  TRAIN_TIME="$(python - <<PY
import json
print(json.loads(${PROFILE_JSON@Q})["train_time"])
PY
)"
fi
if [[ "${TRAIN_GPUS}" == "0" ]]; then
  TRAIN_GPUS="$(python - <<PY
import json
print(json.loads(${PROFILE_JSON@Q})["train_gpus"])
PY
)"
fi

RUNTIME_CONFIG="${OUT_ROOT}/deepsets_runtime_config.yaml"
MANIFEST_PATH="${OUT_ROOT}/deepsets_manifest.csv"
TRAIN_OUTDIR="${OUT_ROOT}/train"

python - <<PY
from pathlib import Path
import yaml
base_path = Path(${CONFIG@Q})
out_path = Path(${RUNTIME_CONFIG@Q})
manifest_path = str(Path(${MANIFEST_PATH@Q}).resolve())
base = yaml.safe_load(base_path.read_text()) or {}
base.setdefault("data_paths", {})
base["data_paths"]["deepsets_manifest_csv"] = manifest_path
out_path.write_text(yaml.safe_dump(base, sort_keys=False))
print(out_path)
PY

_build_is_complete() {
  local manifest="${OUT_ROOT}/deepsets_manifest.csv"
  [[ -f "${manifest}" ]] || return 1
  python - <<PY
import sys
from pathlib import Path
import pandas as pd

manifest = Path(${manifest@Q})
df = pd.read_csv(manifest)
missing = [
    str(path)
    for path in df["set_path"].astype(str)
    if not Path(path).is_file()
]
sys.exit(0 if not missing else 1)
PY
}

_train_is_complete() {
  python - <<PY
import sys
from pathlib import Path

train_root = Path(${TRAIN_OUTDIR@Q})
metrics = list(train_root.rglob("metrics.json"))
sys.exit(0 if metrics else 1)
PY
}

BUILD_ARRAY_JOB_ID="skipped"
MERGE_JOB_ID="skipped"
TRAIN_JOB_ID="skipped"
BUILD_DEPENDENCY=""
MERGE_DEPENDENCY=""

if [[ "${FORCE_BUILD}" == "1" ]] || ! _build_is_complete; then
  BUILD_ARRAY_JOB_ID="$({
    sbatch --parsable \
      --partition="${PARTITION}" \
      --array="0-$((BUILD_SHARDS - 1))" \
      --cpus-per-task="${BUILD_CPUS}" \
      --mem="${BUILD_MEM}" \
      --time="${BUILD_TIME}" \
      --job-name="deepsets-build" \
      --output="${REPO_ROOT}/logs/deepsets-build-%A-%a.out" \
      --error="${REPO_ROOT}/logs/deepsets-build-%A-%a.err" \
      --export=ALL,MODE=build,REPO_ROOT="${REPO_ROOT}",CONFIG="${RUNTIME_CONFIG}",OUT_ROOT="${OUT_ROOT}",NUM_SHARDS="${BUILD_SHARDS}",DEEPSETS_CACHE_ROOT="${DEEPSETS_CACHE_ROOT}",DEEPSETS_BUILD_WORKERS="${DEEPSETS_BUILD_WORKERS}" \
      "${SCRIPT_DIR}/deepsets_job.slurm"
  })"
  BUILD_DEPENDENCY="afterok:${BUILD_ARRAY_JOB_ID}"
else
  echo "Skipping build: existing manifest and case sets found under ${OUT_ROOT}"
fi

if [[ "${FORCE_MERGE}" == "1" ]] || [[ -n "${BUILD_DEPENDENCY}" ]]; then
  MERGE_DEPENDENCY="${BUILD_DEPENDENCY}"
  MERGE_SBATCH_ARGS=(
    --parsable
    --partition="${PARTITION}"
    --cpus-per-task="${MERGE_CPUS}"
    --mem="${MERGE_MEM}"
    --time="${MERGE_TIME}"
    --job-name="deepsets-merge"
    --output="${REPO_ROOT}/logs/deepsets-merge-%j.out"
    --error="${REPO_ROOT}/logs/deepsets-merge-%j.err"
    --export=ALL,MODE=merge,REPO_ROOT="${REPO_ROOT}",OUT_ROOT="${OUT_ROOT}"
  )
  if [[ -n "${MERGE_DEPENDENCY}" ]]; then
    MERGE_SBATCH_ARGS+=(--dependency="${MERGE_DEPENDENCY}")
  fi
  MERGE_JOB_ID="$(sbatch "${MERGE_SBATCH_ARGS[@]}" "${SCRIPT_DIR}/deepsets_job.slurm")"
elif ! _build_is_complete; then
  echo "Build outputs incomplete; merge not submitted." >&2
  exit 2
else
  echo "Skipping merge: using existing ${OUT_ROOT}/deepsets_manifest.csv"
  MERGE_JOB_ID="existing"
fi

if [[ "${FORCE_RETRAIN}" == "1" ]] || ! _train_is_complete; then
  if [[ "${MERGE_JOB_ID}" == "existing" ]]; then
    TRAIN_DEPENDENCY=""
  else
    TRAIN_DEPENDENCY="afterok:${MERGE_JOB_ID}"
  fi
  TRAIN_SBATCH_ARGS=(
    --parsable
    --partition="${PARTITION}"
    --cpus-per-task="${TRAIN_CPUS}"
    --mem="${TRAIN_MEM}"
    --time="${TRAIN_TIME}"
    --job-name="deepsets-train"
    --output="${REPO_ROOT}/logs/deepsets-train-%j.out"
    --error="${REPO_ROOT}/logs/deepsets-train-%j.err"
    --export=ALL,MODE=train,REPO_ROOT="${REPO_ROOT}",CONFIG="${RUNTIME_CONFIG}",OUTDIR="${TRAIN_OUTDIR}"
  )
  if [[ -n "${TRAIN_DEPENDENCY}" ]]; then
    TRAIN_SBATCH_ARGS+=(--dependency="${TRAIN_DEPENDENCY}")
  fi
  if [[ "${TRAIN_GPUS}" -gt 0 ]]; then
    TRAIN_SBATCH_ARGS+=(--gres="gpu:${TRAIN_GPUS}")
  fi
  TRAIN_JOB_ID="$(sbatch "${TRAIN_SBATCH_ARGS[@]}" "${SCRIPT_DIR}/deepsets_job.slurm")"
else
  echo "Skipping train: metrics.json already present under ${TRAIN_OUTDIR}"
fi

cat <<MSG
Submitted Deep Sets pipeline:
  config             : ${CONFIG}
  runtime_config     : ${RUNTIME_CONFIG}
  out_root           : ${OUT_ROOT}
  build_array_job_id : ${BUILD_ARRAY_JOB_ID}
  merge_job_id       : ${MERGE_JOB_ID}
  train_job_id       : ${TRAIN_JOB_ID}
  build_time         : ${BUILD_TIME}
  build_shards       : ${BUILD_SHARDS}
  train_gpus         : ${TRAIN_GPUS}
  cache_root         : ${DEEPSETS_CACHE_ROOT}

Monitor:
  squeue -j ${BUILD_ARRAY_JOB_ID},${MERGE_JOB_ID},${TRAIN_JOB_ID}
  sacct -j ${BUILD_ARRAY_JOB_ID},${MERGE_JOB_ID},${TRAIN_JOB_ID} --format=JobIDRaw,State,Elapsed,ExitCode -n -P
  bash scripts/deepsets_sacct_summary.sh ${BUILD_ARRAY_JOB_ID},${MERGE_JOB_ID},${TRAIN_JOB_ID}

Outputs:
  ${OUT_ROOT}/deepsets_manifest.csv
  ${TRAIN_OUTDIR}
MSG
