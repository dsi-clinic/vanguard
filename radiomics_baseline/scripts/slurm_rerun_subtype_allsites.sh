#!/bin/bash
#SBATCH --job-name=subtype_allsites
#SBATCH --output=/home/summe/vanguard/radiomics_baseline/logs/subtype_allsites_%j.out
#SBATCH --error=/home/summe/vanguard/radiomics_baseline/logs/subtype_allsites_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=general
# ---------------------------------------------------------------------------
# Train HER2-enriched and triple-negative models using ALL sites (no site
# filter), with and without ComBat harmonization.
#
# Usage:
#   sbatch scripts/slurm_rerun_subtype_allsites.sh
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
echo "[$(date)] Starting multi-site subtype models"
echo "============================================================"

for cfg in \
    rerun_bin100_kinsubonly_mrmr20_her2_enriched_allsites \
    rerun_bin100_kinsubonly_mrmr20_her2_enriched_allsites_combat \
    rerun_bin100_kinsubonly_mrmr20_triple_negative_allsites \
    rerun_bin100_kinsubonly_mrmr20_triple_negative_allsites_combat; do
    echo ""
    echo "[$(date)] === ${cfg} ==="
    python "${SCRIPTS_DIR}/run_experiment.py" "${CONFIGS_DIR}/${cfg}.yaml"
done

# --- Summary ---

echo ""
echo "============================================================"
echo "[$(date)] All models complete. Results summary:"
echo "============================================================"

for cfg in \
    rerun_bin100_kinsubonly_mrmr20_her2_enriched_allsites \
    rerun_bin100_kinsubonly_mrmr20_her2_enriched_allsites_combat \
    rerun_bin100_kinsubonly_mrmr20_triple_negative_allsites \
    rerun_bin100_kinsubonly_mrmr20_triple_negative_allsites_combat; do
    metrics="${REPO_DIR}/outputs/${cfg}/training/metrics.json"
    if [ -f "${metrics}" ]; then
        echo ""
        echo "--- ${cfg} ---"
        python -c "
import json, pathlib
m = json.loads(pathlib.Path('${metrics}').read_text())
cv = m.get('auc_train_cv', 'N/A')
cv_std = m.get('auc_train_cv_std', 'N/A')
test = m.get('auc_test', 'N/A')
nf = m.get('n_features_used', 'N/A')
ns = m.get('n_samples', 'N/A')
harm = m.get('harmonization_mode', 'N/A')
print(f'  CV AUC: {cv:.4f} ± {cv_std:.4f}' if isinstance(cv, float) else f'  CV AUC: {cv}')
print(f'  Test AUC: {test:.4f}' if isinstance(test, float) else f'  Test AUC: {test}')
print(f'  Features used: {nf}')
print(f'  Test samples: {ns}')
print(f'  Harmonization: {harm}')
"
    else
        echo ""
        echo "--- ${cfg} ---"
        echo "  [WARN] metrics.json not found"
    fi
done

echo ""
echo "[$(date)] Done."
