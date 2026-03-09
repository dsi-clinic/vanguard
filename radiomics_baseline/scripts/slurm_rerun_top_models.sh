#!/bin/bash
#SBATCH --job-name=rerun_top6
#SBATCH --output=/home/summe/vanguard/radiomics_baseline/logs/rerun_top6_%j.out
#SBATCH --error=/home/summe/vanguard/radiomics_baseline/logs/rerun_top6_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --partition=general
# ---------------------------------------------------------------------------
# Re-run top 6 ISPY2 phase-blind models to verify teammate's results.
#
# Ordered to maximize extraction sharing:
#   Group A (bin100 + kinsubonly): models 1, 4, 6  — 1 extraction, 3 trainings
#   Group B (bin100 + all):        models 2, 3     — 1 extraction, 2 trainings
#   Group C (bin8 + kinsubonly):   model 5         — 1 extraction, 1 training
#
# Usage:
#   sbatch scripts/slurm_rerun_top_models.sh
# ---------------------------------------------------------------------------

set -euo pipefail

REPO_DIR="/home/summe/vanguard/radiomics_baseline"
SCRIPTS_DIR="${REPO_DIR}/scripts"
CONFIGS_DIR="${REPO_DIR}/configs/rerun"

mkdir -p "${REPO_DIR}/logs" "${REPO_DIR}/outputs"

eval "$(micromamba shell hook --shell bash)"
micromamba activate vanguard

cd "${REPO_DIR}"

echo "============================================================"
echo "[$(date)] Starting re-run of top 6 models"
echo "============================================================"

# --- Group A: bin100 + kinsubonly (3 models, 1 shared extraction) ---

echo ""
echo "[$(date)] === Group A: bin100 + kinsubonly ==="

echo "[$(date)] Model 1: kbest40 corr080 (triggers extraction)"
python "${SCRIPTS_DIR}/run_experiment.py" "${CONFIGS_DIR}/rerun_bin100_kinsubonly_kbest40_corr080.yaml"

echo "[$(date)] Model 4: kbest60 corr095 (reuses extraction)"
python "${SCRIPTS_DIR}/run_experiment.py" "${CONFIGS_DIR}/rerun_bin100_kinsubonly_kbest60_corr095.yaml"

echo "[$(date)] Model 6: mrmr20 (reuses extraction)"
python "${SCRIPTS_DIR}/run_experiment.py" "${CONFIGS_DIR}/rerun_bin100_kinsubonly_mrmr20.yaml"

# --- Group B: bin100 + all (2 models, 1 shared extraction) ---

echo ""
echo "[$(date)] === Group B: bin100 + all ==="

echo "[$(date)] Model 2: mrmr20 corr080 (triggers extraction)"
python "${SCRIPTS_DIR}/run_experiment.py" "${CONFIGS_DIR}/rerun_bin100_all_mrmr20_corr080.yaml"

echo "[$(date)] Model 3: mrmr20 corr070 (reuses extraction)"
python "${SCRIPTS_DIR}/run_experiment.py" "${CONFIGS_DIR}/rerun_bin100_all_mrmr20_corr070.yaml"

# --- Group C: bin8 + kinsubonly (1 model, own extraction) ---

echo ""
echo "[$(date)] === Group C: bin8 + kinsubonly ==="

echo "[$(date)] Model 5: kbest50 corr095 (triggers extraction)"
python "${SCRIPTS_DIR}/run_experiment.py" "${CONFIGS_DIR}/rerun_bin8_kinsubonly_kbest50_corr095.yaml"

# --- Summary ---

echo ""
echo "============================================================"
echo "[$(date)] All 6 models complete. Results summary:"
echo "============================================================"

for d in \
    rerun_bin100_kinsubonly_kbest40_corr080 \
    rerun_bin100_all_mrmr20_corr080 \
    rerun_bin100_all_mrmr20_corr070 \
    rerun_bin100_kinsubonly_kbest60_corr095 \
    rerun_bin8_kinsubonly_kbest50_corr095 \
    rerun_bin100_kinsubonly_mrmr20; do
    metrics="${REPO_DIR}/outputs/${d}/training/metrics.json"
    if [ -f "${metrics}" ]; then
        echo ""
        echo "--- ${d} ---"
        python -c "
import json, pathlib
m = json.loads(pathlib.Path('${metrics}').read_text())
cv = m.get('auc_train_cv', 'N/A')
cv_std = m.get('auc_train_cv_std', 'N/A')
test = m.get('auc_test', 'N/A')
nf = m.get('n_features_used', 'N/A')
print(f'  CV AUC: {cv:.4f} ± {cv_std:.4f}' if isinstance(cv, float) else f'  CV AUC: {cv}')
print(f'  Test AUC: {test:.4f}' if isinstance(test, float) else f'  Test AUC: {test}')
print(f'  Features used: {nf}')
"
    else
        echo ""
        echo "--- ${d} ---"
        echo "  [WARN] metrics.json not found"
    fi
done

echo ""
echo "[$(date)] Done."
