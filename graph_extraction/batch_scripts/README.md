# Batch 4D Morphometry – SLURM Scripts

SLURM array scripts to run `batch_process_4d.py` on a compute cluster. Workload is split into single-study array tasks to stay within job queue limits (250 max; ~200 concurrent).

The output is equivalent to running:

```bash
python graph_extraction/batch_process_4d.py \
  --input-dir /net/projects2/vanguard/vessel_segmentations \
  --output-dir report/4d_morphometry \
  --skip-existing \
  --quiet
```

---

## Quick Start

### Full run

```bash
cd /path/to/vanguard
./graph_extraction/batch_scripts/submit_4d_array.sh
```

### Test run (first 5 studies)

```bash
./graph_extraction/batch_scripts/submit_4d_array.sh --test
```

### Custom input/output

```bash
./graph_extraction/batch_scripts/submit_4d_array.sh \
  /path/to/vessel_segmentations \
  /path/to/output
```

### Test mode with custom paths

```bash
./graph_extraction/batch_scripts/submit_4d_array.sh \
  /path/to/vessel_segmentations \
  /path/to/output \
  --test
```

---

## Scripts Overview

### 1. `submit_4d_array.sh`

**Role:** Main orchestrator. Discovers studies, submits morphometry array jobs in chunks, waits for each chunk to finish, then submits the merge job.

**Parameters:**

| Position/Flag | Description | Default |
|---------------|-------------|---------|
| `[1]` | `INPUT_DIR` – base directory for vessel segmentations (layout: SITE/STUDY_ID/images/) | `/net/projects2/vanguard/vessel_segmentations` |
| `[2]` | `OUTPUT_DIR` – output directory for morphometry JSONs and manifest | `/net/projects2/vanguard/report/4d_morphometry` |
| `--test` | Limit to first 5 studies | off |

**Behavior:**
- Runs `discover_all_study_ids()` from `batch_process_4d.py` to get study count
- Submits chunks of 200 array tasks (keeps queue around 200; max 250)
- Waits for each chunk to finish before submitting the next
- Submits the merge job with `--dependency=afterok:` on all array job IDs

**Usage examples:**
```bash
./submit_4d_array.sh
./submit_4d_array.sh /net/projects2/vanguard/vessel_segmentations report/4d_morphometry
./submit_4d_array.sh --test
./submit_4d_array.sh /custom/input /custom/output --test
```

---

### 2. `submit_4d_morphometry.slurm`

**Role:** Single array task. Runs one study via `--study-index`, writes `manifest_task_<index>.json` and morphometry files.

**Exported variables (from submit_4d_array.sh):**

| Variable | Description | Default |
|----------|-------------|---------|
| `INPUT_DIR` | Base directory for vessel segmentations | `/net/projects2/vanguard/vessel_segmentations` |
| `OUTPUT_DIR` | Output directory for morphometry JSONs | `/net/projects2/vanguard/report/4d_morphometry` |
| `STUDY_OFFSET` | Index offset for this chunk (e.g. 0, 200, 400) | `0` |

**SBATCH directives:**
- `--partition=general`
- `--cpus-per-task=1`
- `--mem=16G`
- `--time=8:00:00`
- Logs: `logs/4d-morphometry-%A-%a.out`, `logs/4d-morphometry-%A-%a.err`

**Manual submit (for one chunk):**
```bash
sbatch --array=0-99 \
  --export=INPUT_DIR=/path/to/input,OUTPUT_DIR=/path/to/output,STUDY_OFFSET=0 \
  graph_extraction/batch_scripts/submit_4d_morphometry.slurm
```

---

### 3. `submit_4d_merge.slurm`

**Role:** Merge all `manifest_task_*.json` into `manifest.json` and remove task manifests. Should run after all array jobs finish.

**Exported variables:**

| Variable | Description | Default |
|----------|-------------|---------|
| `OUTPUT_DIR` | Directory containing manifest_task_*.json | `/net/projects2/vanguard/report/4d_morphometry` |

**SBATCH directives:**
- `--partition=general`
- `--cpus-per-task=1`
- `--mem=4G`
- `--time=0:30:00`
- Typically submitted with `--dependency=afterok:JOBID` by `submit_4d_array.sh`

**Manual submit (after array job completes):**
```bash
sbatch --dependency=afterok:12345678 \
  --export=OUTPUT_DIR=/path/to/output \
  graph_extraction/batch_scripts/submit_4d_merge.slurm
```

---

## Output

Final outputs in `OUTPUT_DIR`:

- `{STUDY_ID}_morphometry.json` – per-study morphometry
- `manifest.json` – merged manifest with `completed`, `failed`, `skipped`, and counts

Same layout and format as when running `batch_process_4d.py` directly.

---

## Running via srun

To avoid holding a login session, run the orchestrator under SLURM:

```bash
srun -N1 -n1 --mem=512M --partition=general --time=24:00:00 \
  ./graph_extraction/batch_scripts/submit_4d_array.sh
```

---

## Prerequisites

- `micromamba` with `vanguard` environment
- `logs/` directory (created by `submit_4d_array.sh`)
- Write access to `OUTPUT_DIR` and `/net/projects2/vanguard/tmp/` for per-job temp dirs
