# 4D Morphometry Batch Scripts

SLURM scripts to run batch 4D skeleton extraction and morphometry on a cluster, parallelizing across studies while staying within job submit limits.

## Overview

| Script | Purpose |
|--------|---------|
| `submit_4d_array.sh` | Main entry: discovers studies, submits array jobs in chunks, waits for each chunk to finish, then runs merge |
| `submit_4d_morphometry.slurm` | Array job: one task per study (invoked by the wrapper) |
| `submit_4d_merge.slurm` | Merges `manifest_task_*.json` into `manifest.json` after all array tasks complete |

## Prerequisites

- Micromamba with `vanguard` environment
- Run from the vanguard project root
- Update `#SBATCH --mail-user` in the `.slurm` files to your email

## Usage

### Quick Start

```bash
# From project root
cd /path/to/vanguard

# Test mode: run only on first 5 studies
./graph_extraction/batch_scripts/submit_4d_array.sh --test

# Full run on head node (submit all chunks at once; may hit job limits)
./graph_extraction/batch_scripts/submit_4d_array.sh

# Full run via srun (waits for each chunk to finish before submitting next; stays under 250 jobs)
srun -N1 -n1 --mem=512M --partition=general --time=24:00:00 \
  ./graph_extraction/batch_scripts/submit_4d_array.sh
```

### Arguments

```
./submit_4d_array.sh [INPUT_DIR] [OUTPUT_DIR] [--test]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `INPUT_DIR` | `/net/projects2/vanguard/vessel_segmentations` | Base directory for vessel segmentations (`SITE/STUDY_ID/images/`) |
| `OUTPUT_DIR` | `/net/projects2/vanguard/report/4d_morphometry` | Output directory for morphometry JSONs and manifest |
| `--test` | — | Limit to first 5 studies (can appear anywhere in args) |

### Examples

```bash
# Default input/output
./submit_4d_array.sh

# Custom paths
./submit_4d_array.sh /path/to/vessel_segmentations /path/to/output

# Test with custom output
./submit_4d_array.sh /path/to/input /path/to/output --test

# Via srun (recommended when job submit limit is low)
srun -N1 -n1 --mem=512M --partition=general --time=24:00:00 \
  ./graph_extraction/batch_scripts/submit_4d_array.sh
```

## How It Works

1. **Discovery**: Script finds all studies in `INPUT_DIR` (layout: `SITE/STUDY_ID/images/*vessel_segmentation*`).

2. **Chunking**: Submits array jobs in chunks of 250 (configurable via `CHUNK_SIZE` in the script).

3. **Waiting**: When run via `srun`, the script polls `squeue` every 2 minutes and waits for each chunk to finish before submitting the next. This keeps your queue under the job limit (e.g., 250).

4. **Merge**: After all array chunks complete, submits `submit_4d_merge.slurm` with `--dependency=afterok` to combine `manifest_task_*.json` into `manifest.json`.

## Output

- **Per study**: `{OUTPUT_DIR}/{study_id}_morphometry.json`
- **Task manifests**: `{OUTPUT_DIR}/manifest_task_0.json`, `manifest_task_1.json`, … (deleted after merge)
- **Final manifest**: `{OUTPUT_DIR}/manifest.json`
- **Logs**: `logs/4d-morphometry-*.out` and `logs/4d-morphometry-*.err`

## Manual Submission

To submit a single chunk manually (e.g., studies 0–99):

```bash
sbatch --array=0-99 \
  --export=INPUT_DIR=/path/to/input,OUTPUT_DIR=/path/to/output,STUDY_OFFSET=0 \
  graph_extraction/batch_scripts/submit_4d_morphometry.slurm
```

To merge manifests after arrays complete:

```bash
sbatch --dependency=afterok:JOB_ID1:JOB_ID2 \
  --export=OUTPUT_DIR=/path/to/output \
  graph_extraction/batch_scripts/submit_4d_merge.slurm
```

## Configuration

Edit `submit_4d_array.sh` to change:

- `CHUNK_SIZE=250` — Max jobs per chunk (set to your job limit)
- `POLL_INTERVAL=120` — Seconds between queue checks when waiting

Edit the `.slurm` files to adjust resources (CPUs, memory, time, partition).
