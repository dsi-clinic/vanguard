#!/usr/bin/env bash
set -euo pipefail

MICROMAMBA_BIN="${MICROMAMBA_BIN:-/home/annawoodard/bin/micromamba}"
CODER_ENV="${CODER_ENV:-codex}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-/net/projects/annawoodard/micromamba}"
VANGUARD_DIR="/home/annawoodard/gt/vanguard"
AMY_DIR="/home/annawoodard/gt/vanguard/crew/amy"
RAD_DIR="${AMY_DIR}/radiomics_baseline"
NOTEBOOK="${RAD_DIR}/lab_notebook.md"
STATE_DIR="${AMY_DIR}/logs"
LAST_NUDGE_FILE="${STATE_DIR}/.amy_radiomics_last_nudge_epoch"
FLOOR_BREACH_FILE="${STATE_DIR}/.amy_radiomics_floor_breach_state"
NUDGE_INTERVAL_MIN="${NUDGE_INTERVAL_MIN:-30}"
GENERAL_FLOOR="${GENERAL_FLOOR:-8}"
BURST_GPU_CAP="${BURST_GPU_CAP:-8}"
FORCE_NUDGE="${FORCE_NUDGE:-0}"
STAMP="$(date '+%Y-%m-%d %H:%M:%S %Z')"
USER_NAME="${USER:-$(id -un)}"

mkdir -p "${STATE_DIR}"

status_text="$(${MICROMAMBA_BIN} run -n ${CODER_ENV} bash -lc "cd '${VANGUARD_DIR}' && gt crew status amy" 2>&1 || true)"
did_start=0
if ! grep -q "running vanguard/amy" <<<"${status_text}"; then
  ${MICROMAMBA_BIN} run -n ${CODER_ENV} bash -lc "cd '${VANGUARD_DIR}' && gt crew start amy" >/dev/null 2>&1 || true
  did_start=1
fi

running_count="$(squeue -u "$USER_NAME" -h -o '%t|%j|%Z' | awk -F'|' '$3 ~ /\/vanguard\/crew\/amy\/radiomics_baseline/ && $1=="R" {c++} END {print c+0}')"
pending_count="$(squeue -u "$USER_NAME" -h -o '%t|%j|%Z' | awk -F'|' '$3 ~ /\/vanguard\/crew\/amy\/radiomics_baseline/ && $1=="PD" {c++} END {print c+0}')"
recent_jobs="$(squeue -u "$USER_NAME" -h -o '%i|%t|%j|%Z' | awk -F'|' '$4 ~ /\/vanguard\/crew\/amy\/radiomics_baseline/ {print $0}' | head -n 8 | sed 's/"/\\"/g')"

# Standard slot floor: count amy radiomics jobs on partition=general with qos!=burst.
general_nonburst_active="$(squeue -u "$USER_NAME" -h -o '%t|%P|%q|%Z' | awk -F'|' '$4 ~ /\/vanguard\/crew\/amy\/radiomics_baseline/ && $2=="general" && $3!="burst" && ($1=="R" || $1=="PD") {c++} END {print c+0}')"

# Burst usage estimate: sum requested GPUs for amy radiomics jobs with qos=burst (R/PD).
read -r burst_job_count burst_gpu_total < <(
  squeue -u "$USER_NAME" -h -o '%t|%q|%b|%Z' | awk -F'|' '
    $4 ~ /\/vanguard\/crew\/amy\/radiomics_baseline/ && $2=="burst" && ($1=="R" || $1=="PD") {
      jobs++
      g=0
      if ($3 ~ /gpu/) {
        n=split($3, a, ":")
        if (a[n] ~ /^[0-9]+$/) g=a[n]+0
        else g=1
      }
      sum += g
    }
    END { printf "%d %d\n", jobs+0, sum+0 }
  '
)

floor_breach=0
if [[ "$general_nonburst_active" =~ ^[0-9]+$ ]] && (( general_nonburst_active < GENERAL_FLOOR )); then
  floor_breach=1
fi

# Nudge once when entering a floor-breach episode; avoid repeated near-duplicate nudges.
prev_floor_breach=0
if [[ -f "${FLOOR_BREACH_FILE}" ]]; then
  read -r prev_floor_breach < "${FLOOR_BREACH_FILE}" || true
fi
if ! [[ "${prev_floor_breach}" =~ ^[0-9]+$ ]]; then
  prev_floor_breach=0
fi
floor_breach_transition=0
if (( floor_breach == 1 )) && (( prev_floor_breach == 0 )); then
  floor_breach_transition=1
fi
if (( floor_breach == 0 )) && (( prev_floor_breach == 1 )); then
  printf '0\n' > "${FLOOR_BREACH_FILE}"
fi

now_epoch="$(date +%s)"
last_nudge_epoch=0
if [[ -f "${LAST_NUDGE_FILE}" ]]; then
  read -r last_nudge_epoch < "${LAST_NUDGE_FILE}" || true
fi
if ! [[ "${last_nudge_epoch}" =~ ^[0-9]+$ ]]; then
  last_nudge_epoch=0
fi

interval_sec=$((NUDGE_INTERVAL_MIN * 60))
elapsed_sec=$((now_epoch - last_nudge_epoch))
should_nudge=0
if [[ "${FORCE_NUDGE}" == "1" ]] || (( did_start == 1 )) || (( elapsed_sec >= interval_sec )) || (( floor_breach_transition == 1 )); then
  should_nudge=1
fi

if (( should_nudge == 1 )); then
  message="$(cat <<MSG
[radiomics night-watch tick] ${STAMP}
Scope: /home/annawoodard/gt/vanguard/crew/amy/radiomics_baseline
Priority: improve radiomics_baseline performance with ISPY2-first focus.

Current queue snapshot (amy radiomics): running=${running_count}, pending=${pending_count}

Slot policy (required):
- Do not leave standard capacity idle: keep at least ${GENERAL_FLOOR} active amy-radiomics jobs on partition=general with qos!=burst (running+pending).
- Current standard active count: ${general_nonburst_active}/${GENERAL_FLOOR}.
- If below floor, submit highest-value experiments now until floor is restored.
- Burst is allowed for fast probes: up to ${BURST_GPU_CAP} total GPUs in qos=burst for amy radiomics.
- Current burst usage estimate: ${burst_gpu_total}/${BURST_GPU_CAP} GPUs across ${burst_job_count} burst jobs.
- Any burst submission must set walltime <= 04:00:00.

Cycle instructions:
- Read latest evidence first: experiment_results.csv, experiment_averages.csv, recent Slurm logs, and sacct/squeue state.
- Think deeply before acting: extract 2-3 concrete insights from the latest runs (effect size + uncertainty/variance), then state what those imply causally.
- Update ${NOTEBOOK} every cycle with: Results, Insights learned, Hypothesis update, Next experiments, and Actions taken.
- Do not propose new experiments until insights + hypothesis are written in the notebook for this cycle.
- Choose next experiments to directly test the top hypothesis and maximize expected performance gain per slot.
- Prefer high-decision-value ISPY2 experiments first; defer non-ISPY2 unless needed for diagnosis.
- If a job crashes/fails, root-cause quickly, implement minimal fix, and resubmit.
- If you change code/config, commit locally with concise message and validation evidence.
- NEVER push; local commits only. Do not run gt done or gt mq submit.
- Never run compute on head node; all heavy runs via Slurm.
- If no safe useful next action exists, append a PASS note in the notebook with reason.
MSG
)"

  if printf '%s\n' "${message}" | ${MICROMAMBA_BIN} run -n ${CODER_ENV} bash -lc \
    "cd '${VANGUARD_DIR}' && gt nudge vanguard/amy --mode immediate --priority urgent --stdin" >/dev/null; then
    printf '%s\n' "${now_epoch}" > "${LAST_NUDGE_FILE}"
    printf '%s\n' "${floor_breach}" > "${FLOOR_BREACH_FILE}"
    echo "[${STAMP}] amy heartbeat ok | nudged | running=${running_count} pending=${pending_count} general_nonburst=${general_nonburst_active}/${GENERAL_FLOOR} burst_gpu=${burst_gpu_total}/${BURST_GPU_CAP}"
  else
    echo "[${STAMP}] amy heartbeat ok | nudge_failed | running=${running_count} pending=${pending_count} general_nonburst=${general_nonburst_active}/${GENERAL_FLOOR} burst_gpu=${burst_gpu_total}/${BURST_GPU_CAP}"
  fi
else
  echo "[${STAMP}] amy heartbeat ok | nudge skipped (${elapsed_sec}s < ${interval_sec}s) | running=${running_count} pending=${pending_count} general_nonburst=${general_nonburst_active}/${GENERAL_FLOOR} burst_gpu=${burst_gpu_total}/${BURST_GPU_CAP}"
fi
