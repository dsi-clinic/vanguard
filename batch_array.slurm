#!/bin/bash
#SBATCH --job-name=vessels
#SBATCH --partition=general
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --array=0-113
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err


INPUT_DIR="/net/projects2/vanguard/vessel_segmentations"
OUTPUT_DIR="/home/jmcarias/vanguard/output/skeleton_to_graph_output"

# Build file array
FILES=($(ls $INPUT_DIR/*.npy))

TOTAL=${#FILES[@]}

if [ $SLURM_ARRAY_TASK_ID -ge $TOTAL ]; then
    echo "Task ID $SLURM_ARRAY_TASK_ID out of range (only $TOTAL files). Exiting."
    exit 0
fi

INPUT_FILE=${FILES[$SLURM_ARRAY_TASK_ID]}

echo "Processing file $SLURM_ARRAY_TASK_ID / $TOTAL:"
echo "  $INPUT_FILE"

python /home/jmcarias/vanguard/notebooks/skeleton3d_utils/run_pipeline.py "$INPUT_FILE" "$OUTPUT_DIR" 0.5
