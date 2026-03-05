# SLURM Submit Scripts

Tracked Slurm helpers in this branch are:

- `submit_vessel_segmentation.slurm`
- `submit_vessel_segmentation_array.slurm`
- `submit_4d_vs_tc4d_benchmark.sh`
- `submit_4d_vs_tc4d_benchmark_array.slurm`
- `submit_4d_vs_tc4d_benchmark_reduce.slurm`

## Rules

- Run heavy workloads on Slurm compute nodes, not headnode.
- Use headnode only for edits, job submission, and monitoring.
- Worker jobs must activate the `vanguard` environment.

## Script Roles

### Vessel segmentation

- `submit_vessel_segmentation.slurm`
  - Single-job GPU segmentation run.
- `submit_vessel_segmentation_array.slurm`
  - Array version for one-file-per-task segmentation.

### 4d-vs-tc4d benchmark

- `submit_4d_vs_tc4d_benchmark.sh`
  - Headnode orchestrator: builds study manifest, submits array, then submits reducer with dependency.
- `submit_4d_vs_tc4d_benchmark_array.slurm`
  - Worker script: runs one compare task per study ID.
- `submit_4d_vs_tc4d_benchmark_reduce.slurm`
  - Reducer script: aggregates per-study records to summary CSV/JSON.

## Recommended Usage

### Benchmark suite (preferred)

```bash
OUT_DIR="/net/projects2/vanguard/benchmarks/4d_vs_tc4d/run_$(date +%Y%m%d_%H%M%S)"
slurm_submit_scripts/submit_4d_vs_tc4d_benchmark.sh "${OUT_DIR}"
```

Optional environment overrides for the orchestrator:

- `SEGMENTATION_DIR=/net/projects2/vanguard/vessel_segmentations`
- `COMPARE_SCRIPT=/path/to/graph_extraction/debug_compare_4d_vs_tc4d.py`

### Vessel segmentation

```bash
# Single job
sbatch slurm_submit_scripts/submit_vessel_segmentation.slurm

# Array job
sbatch slurm_submit_scripts/submit_vessel_segmentation_array.slurm
```

To run a higher-resource variant, prefer `sbatch` overrides instead of a separate script:

```bash
sbatch --cpus-per-task=32 --mem=256G slurm_submit_scripts/submit_vessel_segmentation.slurm
```

## Monitoring

```bash
squeue -u "$USER"
squeue -j <JOB_ID_OR_ARRAY_ID>
tail -f logs/<job-name>-<job-id>.out
sacct -j <JOB_ID_OR_ARRAY_ID> --format=JobID,State,ExitCode,Elapsed
```
