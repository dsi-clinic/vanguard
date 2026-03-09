#!/bin/bash
# ---------------------------------------------------------------------------
# submit_2d_vs_3d.sh — Submit both peritumor sweeps in parallel
#
# Usage:
#   bash submit_2d_vs_3d.sh          # submit both sweeps
#   bash submit_2d_vs_3d.sh --dry    # dry-run (just print commands)
# ---------------------------------------------------------------------------

set -euo pipefail

PROJ_DIR="$HOME/vanguard/radiomics_baseline"
SCRIPTS_DIR="${PROJ_DIR}/scripts"
DRY_RUN="${1:-}"

mkdir -p logs

echo "============================================"
echo "  Radiomics 2D vs 3D Ablation Sweep"
echo "============================================"
echo ""
echo "  Sweep A (2D peri): 12 configs, 4 extractions"
echo "  Sweep B (3D peri): 12 configs, 4 extractions"
echo "  Total:             24 configs, 8 extractions"
echo ""

if [[ "$DRY_RUN" == "--dry" ]]; then
    echo "[DRY RUN] Would submit:"
    echo "  sbatch slurm_sweep_peri2d.sh"
    echo "  sbatch slurm_sweep_peri3d.sh"
    exit 0
fi

# Submit both sweeps — they run in parallel on separate nodes
JOB_2D=$(sbatch --parsable "${SCRIPTS_DIR}/slurm_sweep_peri2d.sh")
echo "Submitted 2D sweep: job ${JOB_2D}"

JOB_3D=$(sbatch --parsable "${SCRIPTS_DIR}/slurm_sweep_peri3d.sh")
echo "Submitted 3D sweep: job ${JOB_3D}"

echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/sweep_peri2d_${JOB_2D}.out"
echo "  tail -f logs/sweep_peri3d_${JOB_3D}.out"
echo ""
echo "When both finish, merge summaries with:"
echo "  head -1 configs/ablation_summary.csv > configs/ablation_summary_merged.csv"
echo "  tail -n +2 -q configs/*ablation_summary*.csv >> configs/ablation_summary_merged.csv"