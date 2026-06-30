#!/usr/bin/env bash
#
# Stage 0 of the "resampled segmentations -> pCR" comparison.
#
# WHY THIS EXISTS
# ---------------
# When the resampled vessel segmentations were turned into centerlines, the run
# produced the skeletons and `*_morphometry.json` (the `morph` feature block) but
# NOT `*_tumor_graph_features.json` (the `graph` + `kinematic` feature blocks).
# The previous group's independent-signal analysis needs all of those blocks, so
# before we can re-run it on the resampled data we must recompute the missing
# tumor-graph JSONs.
#
# WHAT IT DOES
# ------------
# Runs `run_skeleton_processing.py --features-only --force-features` over every
# resampled centerline study dir for one cohort (default ISPY2). "features-only"
# reuses the existing skeleton `.npy` files (no expensive re-skeletonization);
# "--force-features" is required because `morphometry.json` already exists and the
# pipeline would otherwise skip recomputation. The kinematic block is rebuilt from
# the resampled vessel-segmentation time series (INPUT_ROOT), and the tumor-graph
# block needs the expert tumor masks (TUMOR_MASK_DIR).
#
# This is a thin wrapper around the general `submit_tc4d_array.sh`; all the array
# logic lives there. It is idempotent (safe to re-run; re-forces the JSONs) and
# parameterized via the env vars below.
#
# USAGE
#   bash graph_extraction/slurm/submit_resampled_tumor_graph_features.sh
#   COHORT=NACT bash graph_extraction/slurm/submit_resampled_tumor_graph_features.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Parameters (override via environment) ----------------------------------
WORKSPACE="${WORKSPACE:-/gpfs/data/karczmar-lab/workspaces/saritbose}"
# Cohort to process. The previous group's baseline is ISPY2-only, so that is the
# default; set COHORT=DUKE|ISPY1|NACT to do another cohort.
COHORT="${COHORT:-ISPY2}"
# Resampled vessel segmentations: <root>/<cohort>/<case>/images/*_vessel_segmentation.npz
INPUT_ROOT="${INPUT_ROOT:-${WORKSPACE}/resampled-vessel-segmentations}"
# Resampled centerlines (where the skeleton .npy live and the JSONs are written):
# <root>/<cohort>/<case>/
OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKSPACE}/resamples_centerlines_tc4d}"
# Tumor masks RESAMPLED onto the resampled image grid (so they align with the
# resampled skeletons). These are produced by scripts/resample_tumor_masks.py
# (submit: scripts/slurm/submit_resample_tumor_masks.slurm). The native expert
# masks must NOT be used here: they are on the native grid and cause a
# "mask shape mismatch" that silently drops the graph + kinematic feature blocks.
TUMOR_MASK_DIR="${TUMOR_MASK_DIR:-/gpfs/data/karczmar-lab/workspaces/saritbose/resampled-tumor-masks}"

# --- Hand off to the general array launcher ---------------------------------
# These env vars are read by submit_tc4d_array.sh / .slurm (forwarded via
# sbatch --export=ALL).
export FEATURES_ONLY=1     # reuse existing skeletons; recompute features only
export FORCE_FEATURES=1    # required: morphometry.json already exists
export SEG_ONLY=1          # only enumerate cases that actually have seg .npz
export NO_RENDER_MIP=1     # we don't need the coverage MIP for this pass
export SITE_FILTER="${COHORT}"
export TUMOR_MASK_DIR

echo "Stage 0: recompute tumor-graph features for resampled ${COHORT}"
echo "  input_root  : ${INPUT_ROOT}"
echo "  output_root : ${OUTPUT_ROOT}"
echo "  tumor masks : ${TUMOR_MASK_DIR}"
echo

exec bash "${SCRIPT_DIR}/submit_tc4d_array.sh" "${INPUT_ROOT}" "${OUTPUT_ROOT}"
