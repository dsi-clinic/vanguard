#!/bin/bash
# Quick script to submit the vessel segmentation job

echo "Submitting vessel segmentation job to SLURM..."
echo "This will process all 7926 images through the complete pipeline:"
echo "  - STEP-1: Preprocessing"
echo "  - STEP-2: Breast mask segmentation"
echo "  - STEP-3: Vessel segmentation"
echo ""
echo "Estimated time: 12-24 hours"
echo "Output location: /net/projects2/vanguard/vessel_segmentations/"
echo ""

# Check if job already exists
JOB_ID=$(sbatch submit_vessel_segmentation.slurm 2>&1 | grep -oP '\d+')

if [ -n "$JOB_ID" ]; then
    echo "✓ Job submitted successfully!"
    echo "  Job ID: $JOB_ID"
    echo ""
    echo "Monitor progress with:"
    echo "  squeue -u $USER"
    echo ""
    echo "Check logs with:"
    echo "  tail -f logs/vessel-seg-${JOB_ID}.out"
    echo ""
    echo "Cancel job with:"
    echo "  scancel $JOB_ID"
else
    echo "✗ Failed to submit job"
fi

