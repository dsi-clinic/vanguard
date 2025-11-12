#!/bin/bash
#SBATCH -J rad_expts
#SBATCH -p general
#SBATCH -N 1
#SBATCH -c 8                # 8 CPUs
#SBATCH -t 04:00:00         # 4 hours, adjust as needed
#SBATCH --mem=32G
#SBATCH -o slurm-rad-%j.out

set -euo pipefail

# 1 activate your conda/micromamba env in a non-interactive shell
# (this is the part your previous sbatch was missing)
eval "$(micromamba shell hook --shell bash)"
micromamba activate vanguard

# 2 base paths (adjust if your home is different)
ROOT=/home/summe/vanguard
SCRIPTS=$ROOT/radiomics_baseline

IMAGES=/net/projects2/vanguard/MAMA-MIA-syn60868042/images
MASKS=/net/projects2/vanguard/MAMA-MIA-syn60868042/segmentations/expert
LABELS=$SCRIPTS/labels.csv
SPLIT=$SCRIPTS/splits_train_test_ready.csv
PARAMS=$SCRIPTS/pyradiomics_params.yaml

# We'll make all outputs inside this dir
OUTROOT=$SCRIPTS/experiments
mkdir -p "$OUTROOT"

echo "[INFO] Starting radiomics experiments at $(date)"

############################
# EXTRACTION RUNS  #
############################
# We’ll build 4 feature sets:
#   E1: peri=5,  multi-phase (0001,0002)
#   E2: peri=5,  single-phase (0001)
#   E3: peri=0,  multi-phase (0001,0002)
#   E4: peri=0,  single-phase (0001)

# 1 peri 5mm, multi-phase
python $SCRIPTS/radiomics_extract.py \
  --images "$IMAGES" \
  --masks  "$MASKS" \
  --labels "$LABELS" \
  --split  "$SPLIT" \
  --output "$OUTROOT/extract_peri5_multiphase" \
  --params "$PARAMS" \
  --image-pattern "{pid}/{pid}_0001.nii.gz,{pid}/{pid}_0002.nii.gz" \
  --mask-pattern  "{pid}.nii.gz" \
  --peri-radius-mm 5 \
  --n-proc 8

# 2 peri 5mm, single-phase
python $SCRIPTS/radiomics_extract.py \
  --images "$IMAGES" \
  --masks  "$MASKS" \
  --labels "$LABELS" \
  --split  "$SPLIT" \
  --output "$OUTROOT/extract_peri5_p0001" \
  --params "$PARAMS" \
  --image-pattern "{pid}/{pid}_0001.nii.gz" \
  --mask-pattern  "{pid}.nii.gz" \
  --peri-radius-mm 5 \
  --n-proc 8

# 3 peri 0mm, multi-phase
python $SCRIPTS/radiomics_extract.py \
  --images "$IMAGES" \
  --masks  "$MASKS" \
  --labels "$LABELS" \
  --split  "$SPLIT" \
  --output "$OUTROOT/extract_peri0_multiphase" \
  --params "$PARAMS" \
  --image-pattern "{pid}/{pid}_0001.nii.gz,{pid}/{pid}_0002.nii.gz" \
  --mask-pattern  "{pid}.nii.gz" \
  --peri-radius-mm 0 \
  --n-proc 8

# 4 peri 0mm, single-phase
python $SCRIPTS/radiomics_extract.py \
  --images "$IMAGES" \
  --masks  "$MASKS" \
  --labels "$LABELS" \
  --split  "$SPLIT" \
  --output "$OUTROOT/extract_peri0_p0001" \
  --params "$PARAMS" \
  --image-pattern "{pid}/{pid}_0001.nii.gz" \
  --mask-pattern  "{pid}.nii.gz" \
  --peri-radius-mm 0 \
  --n-proc 8

echo "[INFO] Extraction runs done at $(date)"