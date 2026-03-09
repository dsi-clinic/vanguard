#!/bin/bash
#SBATCH --job-name=subtype_models
#SBATCH --output=logs/subtype_models_%j.out
#SBATCH --error=logs/subtype_models_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=general
# ---------------------------------------------------------------------------
# Train per-subtype models using the best overall config (mRMR-20, kinetic-
# only, bin100).  All reuse existing shared extraction — training only.
#
# Usage:
#   sbatch scripts/slurm_rerun_subtype_models.sh
# ---------------------------------------------------------------------------

set -euo pipefail

REPO_DIR="$HOME/vanguard/radiomics_baseline"
SCRIPTS_DIR="${REPO_DIR}/scripts"
CONFIGS_DIR="${REPO_DIR}/configs/rerun"

mkdir -p "${REPO_DIR}/logs" "${REPO_DIR}/outputs"

eval "$(micromamba shell hook --shell bash)"
micromamba activate vanguard

cd "${REPO_DIR}"

echo "============================================================"
echo "[$(date)] Starting per-subtype model training"
echo "============================================================"

for subtype in her2_enriched luminal_a luminal_b triple_negative; do
    echo ""
    echo "[$(date)] === ${subtype} ==="
    python "${SCRIPTS_DIR}/run_experiment.py" \
        "${CONFIGS_DIR}/rerun_bin100_kinsubonly_mrmr20_${subtype}.yaml"
done

# --- Summary ---

echo ""
echo "============================================================"
echo "[$(date)] All subtype models complete. Results summary:"
echo "============================================================"

for subtype in her2_enriched luminal_a luminal_b triple_negative; do
    metrics="${REPO_DIR}/outputs/rerun_bin100_kinsubonly_mrmr20_${subtype}/training/metrics.json"
    if [ -f "${metrics}" ]; then
        echo ""
        echo "--- ${subtype} ---"
        python -c "
import json, pathlib
m = json.loads(pathlib.Path('${metrics}').read_text())
cv = m.get('auc_train_cv', 'N/A')
cv_std = m.get('auc_train_cv_std', 'N/A')
test = m.get('auc_test', 'N/A')
nf = m.get('n_features_used', 'N/A')
ns = m.get('n_samples', 'N/A')
print(f'  CV AUC: {cv:.4f} ± {cv_std:.4f}' if isinstance(cv, float) else f'  CV AUC: {cv}')
print(f'  Test AUC: {test:.4f}' if isinstance(test, float) else f'  Test AUC: {test}')
print(f'  Features used: {nf}')
print(f'  Test samples: {ns}')
"
    else
        echo ""
        echo "--- ${subtype} ---"
        echo "  [WARN] metrics.json not found"
    fi
done

echo ""
echo "[$(date)] Done."
