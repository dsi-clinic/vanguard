# SLURM Submit Scripts

This directory contains SLURM batch job submission scripts for the Vanguard project pipeline.

## Overview

The scripts automate the submission of batch processing jobs on the cluster, handling environment setup, resource allocation, and job configuration.

## Available Scripts

### 1. `submit_vessel_segmentation.slurm` ⭐ **Main Script**

**Purpose**: Run vessel segmentation on all images in the dataset.

**Resources**:
- GPU: 1
- CPUs: 16
- Memory: 128GB
- Time: 12 hours
- Partition: `general`

**Usage**:
```bash
cd /path/to/vanguard
sbatch slurm_submit_scripts/submit_vessel_segmentation.slurm
```

**Features**:
- Processes all images from `/net/projects2/vanguard/MAMA-MIA-syn60868042/images`
- Uses `--resume` flag to skip already processed files
- Automatically cleans up temporary files
- Outputs to `/net/projects2/vanguard/vessel_segmentations/`
- Logs saved to `logs/vessel-seg-<JOB_ID>.out`

**Monitor**:
```bash
squeue -u $USER
tail -f logs/vessel-seg-<JOB_ID>.out
```

---

### 2. `submit_vessel_segmentation_optimized.slurm`

**Purpose**: Optimized version with more resources for faster processing.

**Resources**:
- GPU: 1
- CPUs: 32 (increased from 16)
- Memory: 256GB (increased from 128GB)
- Time: 12 hours
- Workers: 16 (increased from 8)

**Usage**:
```bash
sbatch slurm_submit_scripts/submit_vessel_segmentation_optimized.slurm
```

**When to use**: When you need faster processing and have access to high-memory nodes.

---

### 3. `submit_vessel_segmentation_array.slurm`

**Purpose**: Array job version that runs multiple tasks in parallel.

**Resources**:
- GPU: 1 per array task
- CPUs: 16 per task
- Memory: 128GB per task
- Time: 12 hours per task
- Array: 0-3 (4 parallel jobs)

**Usage**:
```bash
sbatch slurm_submit_scripts/submit_vessel_segmentation_array.slurm
```

**When to use**: When you have multiple GPUs available and want to process files in parallel across multiple nodes.

**Note**: Each array task processes all files but uses `--resume` to avoid duplicate work.

---

### 4. `submit_centerline_extraction.slurm`

**Purpose**: Extract centerlines from vessel segmentation files.

**Resources**:
- CPUs: 8
- Memory: 64GB
- Time: 12 hours
- Partition: `general`
- Environment: `vmtk`

**Usage**:
```bash
sbatch slurm_submit_scripts/submit_centerline_extraction.slurm
```

**What it does**:
- Runs `batch_processing/batch_extract_centerlines.py`
- Processes vessel segmentation files to extract centerlines
- Uses VMTK for centerline extraction

---

### 5. `submit_vtp_to_json_conversion.slurm`

**Purpose**: Convert VTP centerline files to JSON format.

**Resources**:
- CPUs: 4
- Memory: 16GB
- Time: 2 hours
- Partition: `general`

**Usage**:
```bash
sbatch slurm_submit_scripts/submit_vtp_to_json_conversion.slurm
```

**What it does**:
- Runs `batch_processing/batch_convert_vtp_to_json.py`
- Converts VTP centerline files to JSON format for downstream processing

---

### 6. `submit_job.sh`

**Purpose**: Helper script to submit vessel segmentation jobs.

**Usage**:
```bash
cd /path/to/vanguard
./submit_job.sh
```

**What it does**:
- Convenience wrapper for submitting the main vessel segmentation job
- Ensures you're in the correct directory

---

## Common Workflow

### Complete Pipeline

1. **Vessel Segmentation**:
   ```bash
   sbatch slurm_submit_scripts/submit_vessel_segmentation.slurm
   ```

2. **Centerline Extraction** (after segmentation completes):
   ```bash
   sbatch slurm_submit_scripts/submit_centerline_extraction.slurm
   ```

3. **Convert to JSON** (after centerline extraction):
   ```bash
   sbatch slurm_submit_scripts/submit_vtp_to_json_conversion.slurm
   ```

---

## Monitoring Jobs

### Check Job Status
```bash
squeue -u $USER
```

### View Output Logs
```bash
# Vessel segmentation
tail -f logs/vessel-seg-<JOB_ID>.out

# Centerline extraction
tail -f logs/centerline-extract-<JOB_ID>.out

# VTP to JSON
tail -f logs/vtp-to-json-<JOB_ID>.out
```

### View Error Logs
```bash
tail -f logs/vessel-seg-<JOB_ID>.err
```

### Cancel a Job
```bash
scancel <JOB_ID>
```

---

## Configuration

### Email Notifications

All scripts are configured to send email notifications on:
- Job start (`BEGIN`)
- Job completion (`END`)
- Job failure (`FAIL`)

Email address: `ruochun@duke.edu`

To disable email notifications, remove or comment out the `#SBATCH --mail-*` lines.

### Log Directory

All logs are saved to `logs/` directory (relative to project root) with the format:
- Output: `logs/<job-name>-<JOB_ID>.out`
- Errors: `logs/<job-name>-<JOB_ID>.err`

### Temporary Files

All scripts use temporary directories on **local `/tmp`** on each compute node (e.g. `/tmp/batch_seg_<JOB_ID>_<TASK_ID>`). This avoids filling shared storage when jobs crash or are killed before cleanup runs. Temp files are automatically cleaned up on normal job completion.

---

## Environment Requirements

### Vessel Segmentation Scripts
- Environment: `vanguard-gpu`
- Required packages: PyTorch, SimpleITK, NumPy, etc.

### Centerline Extraction Scripts
- Environment: `vmtk`
- Required packages: VTK, PyVista, VMTK

### JSON Conversion Scripts
- Environment: `vanguard-gpu`
- Required packages: Standard Python libraries

---

## Troubleshooting

### Job Fails Immediately
- Check if the environment exists: `micromamba env list`
- Verify paths in the script are correct
- Check error log: `tail logs/<job-name>-<JOB_ID>.err`

### Out of Memory Errors
- Reduce `--max-workers` in the Python command
- Increase `--mem` in the SLURM script
- Use the optimized version with more memory

### Disk Space Issues
- Temporary files use local `/tmp` on compute nodes (no longer on shared storage)
- Check available space: `df -h /net/projects2/vanguard`
- **Reclaiming storage:** If `/net/projects2/vanguard/tmp/` has old orphaned dirs (e.g. `batch_vessel_seg_*` from crashed jobs), remove them to free ~100–200+ GB: `rm -rf /net/projects2/vanguard/tmp/*` (requires write access; coordinate with ruochun if dirs are owned by them)

### Job Times Out
- Increase `--time` limit in the SLURM script
- Use `--resume` flag to continue from where it stopped
- Resubmit the same script (it will skip already processed files)

---

## Notes

- All scripts use `--resume` functionality where applicable to allow safe resubmission
- Temporary directories are automatically cleaned up
- Jobs are configured for the `general` partition which has GPU access
- Maximum time limit for `general` partition is 12 hours

