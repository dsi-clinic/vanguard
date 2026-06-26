#!/usr/bin/env bash
# Launcher job: runs after segmentation job 12108664 completes.
# Submits skeletonization for the 33 newly segmented cases,
# then chains mp4 rendering onto that.
#SBATCH --job-name=followup-33-launcher
#SBATCH --partition=tier1q
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/followup-33-launcher-%j.out
#SBATCH --error=logs/followup-33-launcher-%j.err

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INPUT_ROOT="/ess/scratch/scratch1/t-9sbose/vessel_segmentations"
OUTPUT_ROOT="/ess/scratch/scratch1/t-9sbose/centerlines_tc4d/studies"

cd "${REPO_ROOT}"

# ── 1. Build study list from only the 33 newly segmented cases ──────────────
STUDY_LIST="${REPO_ROOT}/logs/followup-33-study-list.txt"

python3 - <<'PY' > "${STUDY_LIST}"
from pathlib import Path

missing = {
    'DUKE_012','DUKE_019','DUKE_021','DUKE_022','DUKE_045','DUKE_046','DUKE_069',
    'DUKE_101','DUKE_119','DUKE_142','DUKE_168','DUKE_233','DUKE_234','DUKE_258',
    'DUKE_307','DUKE_378','DUKE_400','DUKE_489','DUKE_491',
    'ISPY2_239061','ISPY2_255388','ISPY2_255535','ISPY2_275626','ISPY2_277848',
    'ISPY2_277888','ISPY2_287300','ISPY2_287961','ISPY2_299840','ISPY2_311316',
    'ISPY2_311455','ISPY2_313243','ISPY2_317641','ISPY2_318293',
}

input_root = Path('/ess/scratch/scratch1/t-9sbose/vessel_segmentations')
rows = []
for site_dir in sorted(input_root.iterdir()):
    if not site_dir.is_dir():
        continue
    for study_dir in sorted(site_dir.iterdir()):
        if study_dir.name not in missing:
            continue
        images_dir = study_dir / 'images'
        if not images_dir.is_dir() or not any(
            f.name.endswith('_vessel_segmentation.npz') for f in images_dir.iterdir()
        ):
            print(f'WARNING: {study_dir.name} still has no segmentation — skipping', flush=True)
            continue
        rows.append(f'{site_dir.name}/{study_dir.name}')
print('\n'.join(rows))
PY

TASK_COUNT="$(wc -l < "${STUDY_LIST}")"
echo "Cases ready for skeletonization: ${TASK_COUNT}"

if [[ "${TASK_COUNT}" -eq 0 ]]; then
    echo "ERROR: no cases found — segmentation may not have completed cleanly." >&2
    exit 1
fi

# ── 2. Submit skeletonization array ─────────────────────────────────────────
SKEL_JOB=$(sbatch --parsable \
    --partition=tier1q \
    --array="0-$((TASK_COUNT - 1))%16" \
    --export=ALL,REPO_ROOT="${REPO_ROOT}",INPUT_ROOT="${INPUT_ROOT}",OUTPUT_ROOT="${OUTPUT_ROOT}",STUDY_LIST="${STUDY_LIST}" \
    "${REPO_ROOT}/graph_extraction/slurm/submit_tc4d_array.slurm")

echo "Submitted skeletonization: job ${SKEL_JOB}"

# ── 3. Build mp4 list and submit rendering chained on skeletonization ────────
MP4_LIST="${REPO_ROOT}/logs/followup-33-mp4-list.txt"

# Write the skeleton paths for the 33 cases (they will exist once SKEL_JOB completes)
python3 - <<'PY' > "${MP4_LIST}"
missing = {
    'DUKE_012','DUKE_019','DUKE_021','DUKE_022','DUKE_045','DUKE_046','DUKE_069',
    'DUKE_101','DUKE_119','DUKE_142','DUKE_168','DUKE_233','DUKE_234','DUKE_258',
    'DUKE_307','DUKE_378','DUKE_400','DUKE_489','DUKE_491',
    'ISPY2_239061','ISPY2_255388','ISPY2_255535','ISPY2_275626','ISPY2_277848',
    'ISPY2_277888','ISPY2_287300','ISPY2_287961','ISPY2_299840','ISPY2_311316',
    'ISPY2_311455','ISPY2_313243','ISPY2_317641','ISPY2_318293',
}
from pathlib import Path
root = Path('/ess/scratch/scratch1/t-9sbose/centerlines_tc4d/studies')
for site_dir in sorted(root.iterdir()):
    for study_dir in sorted(site_dir.iterdir()):
        if study_dir.name not in missing:
            continue
        sk = study_dir / f'{study_dir.name}_skeleton_4d_exam_mask.npy'
        print(sk)
PY

MP4_COUNT="$(wc -l < "${MP4_LIST}")"
echo "Cases queued for mp4 rendering: ${MP4_COUNT}"

MP4_JOB=$(REPO_ROOT="${REPO_ROOT}" SKELETON_LIST="${MP4_LIST}" \
    sbatch --parsable \
    --dependency=afterok:${SKEL_JOB} \
    --partition=tier1q \
    --array="0-$((MP4_COUNT - 1))%16" \
    --export=ALL,REPO_ROOT="${REPO_ROOT}",SKELETON_LIST="${MP4_LIST}" \
    "${REPO_ROOT}/graph_extraction/slurm/submit_skeleton_mp4_array.slurm")

echo "Submitted mp4 rendering: job ${MP4_JOB} (depends on ${SKEL_JOB})"
echo ""
echo "Pipeline:"
echo "  12108664 (segmentation) → ${SKEL_JOB} (skeletonization) → ${MP4_JOB} (mp4)"
