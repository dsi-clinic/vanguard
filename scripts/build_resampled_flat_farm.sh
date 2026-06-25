#!/usr/bin/env bash
# =============================================================================
# Build a FLAT symlink farm over the resampled dataset.
#
# WHY THIS EXISTS
# ---------------
# The resampled data is laid out nested by cohort:
#     <RESAMPLED_ROOT>/<COHORT>/<CASE>/<CASE>_NNNN.nii.gz
# but the segmentation pipeline's `find_nii_files` expects a FLAT layout where
# every immediate subdirectory of the images dir is a case:
#     <IMAGES_DIR>/<CASE>/<CASE>_NNNN.nii.gz
# A "symlink farm" bridges the two without copying any data: we create one
# lightweight symlink per case (e.g. DUKE_002 -> .../DUKE/DUKE_002) inside a flat
# directory, then point the pipeline at that flat directory. The pipeline still
# derives the output cohort from the case-name prefix, so outputs stay organized.
#
# This script is idempotent: `ln -sfn` refreshes existing links, so re-running it
# is safe and always reflects the current contents of RESAMPLED_ROOT.
#
# Usage:
#   bash scripts/build_resampled_flat_farm.sh
#   RESAMPLED_ROOT=... FLAT_FARM=... bash scripts/build_resampled_flat_farm.sh
# =============================================================================
set -euo pipefail

RESAMPLED_ROOT="${RESAMPLED_ROOT:-/ess/scratch/scratch1/t-9sbose/resampled_segmentation}"
FLAT_FARM="${FLAT_FARM:-/ess/scratch/scratch1/t-9sbose/resampled_segmentation_flat}"

if [[ ! -d "${RESAMPLED_ROOT}" ]]; then
  echo "ERROR: RESAMPLED_ROOT does not exist: ${RESAMPLED_ROOT}" >&2
  exit 1
fi

mkdir -p "${FLAT_FARM}"

n=0
for cohort_dir in "${RESAMPLED_ROOT}"/*/; do
  [[ -d "${cohort_dir}" ]] || continue
  for case_dir in "${cohort_dir}"*/; do
    [[ -d "${case_dir}" ]] || continue
    case_name="$(basename "${case_dir}")"
    # -s symbolic, -f force/replace, -n don't follow an existing link when replacing
    ln -sfn "${case_dir%/}" "${FLAT_FARM}/${case_name}"
    n=$((n + 1))
  done
done

echo "Flat farm: ${FLAT_FARM}"
echo "Case symlinks created/refreshed: ${n}"
echo -n "Reachable .nii.gz via flat layout (*/*.nii.gz): "
ls "${FLAT_FARM}"/*/*.nii.gz 2>/dev/null | wc -l
