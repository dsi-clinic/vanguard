#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG="${CONFIG:-${REPO_ROOT}/configs/deepsets_ispy2.yaml}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/experiments/deepsets_ispy2}"
PARTITION="${PARTITION:-general}"

BUILD_CPUS="${BUILD_CPUS:-8}"
BUILD_MEM="${BUILD_MEM:-64G}"
BUILD_TIME="${BUILD_TIME:-02:00:00}"
BUILD_SHARDS="${BUILD_SHARDS:-8}"

TRAIN_CPUS="${TRAIN_CPUS:-8}"
TRAIN_MEM="${TRAIN_MEM:-32G}"
TRAIN_TIME="${TRAIN_TIME:-08:00:00}"

MERGE_CPUS="${MERGE_CPUS:-2}"
MERGE_MEM="${MERGE_MEM:-8G}"
MERGE_TIME="${MERGE_TIME:-00:30:00}"

mkdir -p "${REPO_ROOT}/logs" "${OUT_ROOT}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found: ${CONFIG}" >&2
  exit 2
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

BUILD_ARRAY_JOB_ID="$({
  sbatch --parsable \
    --partition="${PARTITION}" \
    --array="0-$((BUILD_SHARDS - 1))" \
    --cpus-per-task="${BUILD_CPUS}" \
    --mem="${BUILD_MEM}" \
    --time="${BUILD_TIME}" \
    --export=ALL,REPO_ROOT="${REPO_ROOT}",CONFIG="${RUNTIME_CONFIG}",OUT_ROOT="${OUT_ROOT}",NUM_SHARDS="${BUILD_SHARDS}" \
    "${SCRIPT_DIR}/submit_build_deepsets_dataset_array.slurm"
})"

MERGE_JOB_ID="$({
  sbatch --parsable \
    --partition="${PARTITION}" \
    --dependency="afterok:${BUILD_ARRAY_JOB_ID}" \
    --cpus-per-task="${MERGE_CPUS}" \
    --mem="${MERGE_MEM}" \
    --time="${MERGE_TIME}" \
    --export=ALL,REPO_ROOT="${REPO_ROOT}",OUT_ROOT="${OUT_ROOT}" \
    "${SCRIPT_DIR}/submit_merge_deepsets_manifest.slurm"
})"

TRAIN_JOB_ID="$({
  sbatch --parsable \
    --partition="${PARTITION}" \
    --dependency="afterok:${MERGE_JOB_ID}" \
    --cpus-per-task="${TRAIN_CPUS}" \
    --mem="${TRAIN_MEM}" \
    --time="${TRAIN_TIME}" \
    --export=ALL,REPO_ROOT="${REPO_ROOT}",CONFIG="${RUNTIME_CONFIG}",OUTDIR="${TRAIN_OUTDIR}" \
    "${SCRIPT_DIR}/submit_train_deepsets.slurm"
})"

cat <<MSG
Submitted Deep Sets pipeline:
  config             : ${CONFIG}
  runtime_config     : ${RUNTIME_CONFIG}
  out_root           : ${OUT_ROOT}
  build_array_job_id : ${BUILD_ARRAY_JOB_ID}
  merge_job_id       : ${MERGE_JOB_ID}
  train_job_id       : ${TRAIN_JOB_ID}

Monitor:
  squeue -j ${BUILD_ARRAY_JOB_ID},${MERGE_JOB_ID},${TRAIN_JOB_ID}
  sacct -j ${BUILD_ARRAY_JOB_ID},${MERGE_JOB_ID},${TRAIN_JOB_ID} --format=JobIDRaw,State,Elapsed,ExitCode -n -P

Outputs:
  ${OUT_ROOT}/deepsets_manifest.csv
  ${TRAIN_OUTDIR}
MSG
