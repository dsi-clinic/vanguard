#!/bin/bash
# Run the ComBat harmonization matrix (12 runs total: 3 model settings x 4 modes).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(dirname "${SCRIPT_DIR}")"

cd "${PROJ_DIR}"

python scripts/run_ablations.py \
  configs/sweep_combat_matrix_all_mrmr20.yaml \
  --generated-dir configs/generated/combat_all_mrmr20

python scripts/run_ablations.py \
  configs/sweep_combat_matrix_all_kbest40.yaml \
  --generated-dir configs/generated/combat_all_kbest40

python scripts/run_ablations.py \
  configs/sweep_combat_matrix_kinsub_kbest50.yaml \
  --generated-dir configs/generated/combat_kinsub_kbest50
