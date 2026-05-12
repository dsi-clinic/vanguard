#!/usr/bin/env bash
# Phase 2a repeated CV: top-2 attention variants from the seed=42 sweep
# x 2 additional seeds (7, 123) so we can do a paired-fold comparison
# against the Phase 3 winners (cos_T80 / h128_d02_lfocal / h256_d02_lfocal,
# already run on seeds 7 and 123 in experiments/deepsets_phase3_repeated_cv/).
#
# Seed=42 reuses the existing predictions.csv from
# experiments/deepsets_sweep_her2_attention_full/<vid>/train/*/*/predictions.csv;
# seeds 7 and 123 are fresh sbatch jobs cloned from the original
# runtime_config.yaml (overriding only model_params.random_state).
#
# Usage:
#   ./scripts/run_phase2a_repeated_cv.sh
#
# Optional env vars:
#   PARTITION   default: general
#   TRAIN_CPUS  default: 8
#   TRAIN_MEM   default: 32G
#   TRAIN_TIME  default: 04:00:00
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PARTITION="${PARTITION:-general}"
TRAIN_CPUS="${TRAIN_CPUS:-8}"
TRAIN_MEM="${TRAIN_MEM:-32G}"
TRAIN_TIME="${TRAIN_TIME:-04:00:00}"

PHASE2A_ROOT="${REPO_ROOT}/experiments/deepsets_phase2a_repeated_cv"
SWEEP_ROOT="${REPO_ROOT}/experiments/deepsets_sweep_her2_attention_full"
mkdir -p "${PHASE2A_ROOT}" "${REPO_ROOT}/logs"

# variant_id -> source runtime config from the Phase 2a seed=42 sweep
declare -A SOURCE_CONFIGS=(
  [attn_logn_h16]="${SWEEP_ROOT}/attn_logn_h16/runtime_config.yaml"
  [attn_h64]="${SWEEP_ROOT}/attn_h64/runtime_config.yaml"
)

NEW_SEEDS=(7 123)

submit_one() {
  local vid="$1"
  local seed="$2"
  local src="${SOURCE_CONFIGS[$vid]}"
  if [[ ! -f "${src}" ]]; then
    echo "MISSING source runtime config: ${src}" >&2
    return 1
  fi
  local outdir="${PHASE2A_ROOT}/${vid}/seed${seed}"
  mkdir -p "${outdir}"
  local rt_cfg="${outdir}/runtime_config.yaml"
  python "${REPO_ROOT}/scripts/clone_runtime_config.py" \
    --base "${src}" \
    --override "model_params.random_state=${seed}" \
    --out "${rt_cfg}" >/dev/null

  local train_out="${outdir}/train"
  local jid
  jid="$(sbatch --parsable \
    --partition="${PARTITION}" \
    --cpus-per-task="${TRAIN_CPUS}" \
    --mem="${TRAIN_MEM}" \
    --time="${TRAIN_TIME}" \
    --job-name="deepsets-p2a-${vid}-s${seed}" \
    --output="${REPO_ROOT}/logs/deepsets-p2a-${vid}-s${seed}-%j.out" \
    --error="${REPO_ROOT}/logs/deepsets-p2a-${vid}-s${seed}-%j.err" \
    --export=ALL,MODE=train,REPO_ROOT="${REPO_ROOT}",CONFIG="${rt_cfg}",OUTDIR="${train_out}" \
    "${REPO_ROOT}/slurm/deepsets_job.slurm")"
  echo "  ${vid} seed=${seed} -> job ${jid}"
}

echo "Submitting Phase 2a repeated-CV jobs (2 variants × 2 new seeds = 4 jobs)..."
for vid in attn_logn_h16 attn_h64; do
  for seed in "${NEW_SEEDS[@]}"; do
    submit_one "${vid}" "${seed}"
  done
done

echo
echo "Seed=42 will reuse existing predictions.csv from the prior sweep runs:"
echo "  attn_logn_h16 -> experiments/deepsets_sweep_her2_attention_full/attn_logn_h16/train/.../predictions.csv"
echo "  attn_h64      -> experiments/deepsets_sweep_her2_attention_full/attn_h64/train/.../predictions.csv"
