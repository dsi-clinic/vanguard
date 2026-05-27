#!/usr/bin/env bash
# Summarize Slurm elapsed time for Deep Sets build, merge, and train jobs.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <build_array_job_id>[,<merge_job_id>,<train_job_id> ...]" >&2
  echo "Example: $0 843002,843010,843011" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_CSV="${OUT_CSV:-${REPO_ROOT}/analysis/figures/deepsets_issue120/deepsets_slurm_elapsed.csv}"
mkdir -p "$(dirname "${OUT_CSV}")"

{
  echo "job_id_raw,job_name,state,elapsed_sec,exit_code"
  for job_id in "$@"; do
    IFS=',' read -r -a parts <<< "${job_id}"
    for part in "${parts[@]}"; do
      [[ -n "${part}" ]] || continue
      sacct -j "${part}" --format=JobIDRaw,JobName,State,Elapsed,ExitCode -n -P | while IFS='|' read -r raw name state elapsed exit_code; do
        [[ "${raw}" == *_batch ]] && continue
        [[ "${raw}" == *_extern ]] && continue
        elapsed_sec="$(python3 - <<PY
parts = "${elapsed}".split(":")
if len(parts) == 2:
    minutes, seconds = parts
    print(int(minutes) * 60 + float(seconds))
elif len(parts) == 3:
    hours, minutes, seconds = parts
    print(int(hours) * 3600 + int(minutes) * 60 + float(seconds))
else:
    print(0.0)
PY
)"
        echo "${raw},${name},${state},${elapsed_sec},${exit_code}"
      done
    done
  done
} > "${OUT_CSV}"

echo "Wrote ${OUT_CSV}"
