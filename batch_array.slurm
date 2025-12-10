#!/bin/bash
#SBATCH --job-name=vessels
#SBATCH --partition=general
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-999
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err

INPUT_DIR="/net/projects2/vanguard/vessel_segmentations"
OUTPUT_DIR="/net/projects2/vanguard/vessel_segmentations/processed_3D"

FILES=($(ls $INPUT_DIR/*.npy))
TOTAL=${#FILES[@]}

TASK_ID=$SLURM_ARRAY_TASK_ID

if [ $TASK_ID -ge $TOTAL ]; then
    echo "TASK_ID $TASK_ID >= TOTAL $TOTAL. Skipping."
    exit 0
fi

INPUT_FILE=${FILES[$TASK_ID]}

echo "Processing TASK_ID $TASK_ID / $TOTAL: $INPUT_FILE"

# 👇 IMPORTANT: use the python from your vanguard env here
/home/jmcarias/micromamba/envs/vanguard/bin/python \
    /home/jmcarias/vanguard/graph_pruning_centerline_extraction/skeleton3d_utils/run_pipeline.py \
    "$INPUT_FILE" "$OUTPUT_DIR" 0.5
