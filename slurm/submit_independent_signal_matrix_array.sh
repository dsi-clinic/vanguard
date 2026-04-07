#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG="${CONFIG:-${REPO_ROOT}/configs/independent_signal.yaml}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/experiments/independent_signal_q3_array}"
PARTITION="${PARTITION:-general}"

CACHE_CPUS="${CACHE_CPUS:-8}"
CACHE_MEM="${CACHE_MEM:-64G}"
CACHE_TIME="${CACHE_TIME:-02:00:00}"

FOLD_CPUS="${FOLD_CPUS:-8}"
FOLD_MEM="${FOLD_MEM:-32G}"
FOLD_TIME="${FOLD_TIME:-08:00:00}"

MERGE_CPUS="${MERGE_CPUS:-2}"
MERGE_MEM="${MERGE_MEM:-16G}"
MERGE_TIME="${MERGE_TIME:-01:00:00}"

mkdir -p "${REPO_ROOT}/logs" "${OUT_ROOT}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found: ${CONFIG}" >&2
  exit 2
fi

N_SPLITS="$(
  python - <<PY
import yaml
from pathlib import Path
conf = yaml.safe_load(Path("${CONFIG}").read_text())
print(int(conf["model_params"].get("n_splits", 5)))
PY
)"

N_ARMS="$(
  python - <<PY
import yaml
from pathlib import Path
conf = yaml.safe_load(Path("${CONFIG}").read_text())
print(len(conf.get("ablation_arms") or []))
PY
)"

if [[ "${N_ARMS}" -le 0 ]]; then
  echo "No ablation_arms defined in ${CONFIG}" >&2
  exit 2
fi

N_TASKS=$(( N_ARMS * N_SPLITS ))
ARRAY_SPEC="0-$((N_TASKS - 1))"
FEATURES_CSV="${OUT_ROOT}/features_full_labeled.csv"

CACHE_WRAP="bash -lc 'source ~/.bashrc || true; if command -v micromamba >/dev/null 2>&1; then eval \"\$(micromamba shell hook -s bash)\"; micromamba activate vanguard; fi; cd \"${REPO_ROOT}\"; python -m modeling.build_cached_table --config \"${CONFIG}\" --outdir \"${OUT_ROOT}\"'"
MERGE_WRAP="bash -lc 'source ~/.bashrc || true; if command -v micromamba >/dev/null 2>&1; then eval \"\$(micromamba shell hook -s bash)\"; micromamba activate vanguard; fi; cd \"${REPO_ROOT}\"; python -m modeling.merge_results --config \"${CONFIG}\" --features-csv \"${FEATURES_CSV}\" --out-root \"${OUT_ROOT}\"'"

CACHE_JOB_ID="$(
  sbatch --parsable \
    --partition="${PARTITION}" \
    --cpus-per-task="${CACHE_CPUS}" \
    --mem="${CACHE_MEM}" \
    --time="${CACHE_TIME}" \
    --output="${REPO_ROOT}/logs/ml-cache-build-%j.out" \
    --error="${REPO_ROOT}/logs/ml-cache-build-%j.err" \
    --wrap="${CACHE_WRAP}"
)"

ARRAY_JOB_ID="$(
  sbatch --parsable \
    --partition="${PARTITION}" \
    --dependency="afterok:${CACHE_JOB_ID}" \
    --array="${ARRAY_SPEC}" \
    --cpus-per-task="${FOLD_CPUS}" \
    --mem="${FOLD_MEM}" \
    --time="${FOLD_TIME}" \
    --export=ALL,REPO_ROOT="${REPO_ROOT}",CONFIG="${CONFIG}",FEATURES_CSV="${FEATURES_CSV}",OUT_ROOT="${OUT_ROOT}",N_SPLITS="${N_SPLITS}" \
    "${SCRIPT_DIR}/submit_ablation_arm_fold_array.slurm"
)"

MERGE_JOB_ID="$(
  sbatch --parsable \
    --partition="${PARTITION}" \
    --dependency="afterok:${ARRAY_JOB_ID}" \
    --cpus-per-task="${MERGE_CPUS}" \
    --mem="${MERGE_MEM}" \
    --time="${MERGE_TIME}" \
    --output="${REPO_ROOT}/logs/ml-ablation-merge-%j.out" \
    --error="${REPO_ROOT}/logs/ml-ablation-merge-%j.err" \
    --wrap="${MERGE_WRAP}"
)"

cat <<EOF
Submitted independent-signal matrix array pipeline:
  config             : ${CONFIG}
  out_root           : ${OUT_ROOT}
  n_arms             : ${N_ARMS}
  n_splits           : ${N_SPLITS}
  n_tasks            : ${N_TASKS}
  array_spec         : ${ARRAY_SPEC}
  cache_job_id       : ${CACHE_JOB_ID}
  array_job_id       : ${ARRAY_JOB_ID}
  merge_job_id       : ${MERGE_JOB_ID}

Monitor:
  squeue -j ${CACHE_JOB_ID},${ARRAY_JOB_ID},${MERGE_JOB_ID}
  sacct -j ${CACHE_JOB_ID},${ARRAY_JOB_ID},${MERGE_JOB_ID} --format=JobIDRaw,State,Elapsed,ExitCode -n -P

Final outputs:
  ${OUT_ROOT}/ablation_summary.csv
  ${OUT_ROOT}/ablation_fold_auc.csv
EOF
