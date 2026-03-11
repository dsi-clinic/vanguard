# SLURM Submit Scripts

Tracked Slurm helpers in this branch are:

- `submit_vessel_segmentation.slurm`
- `submit_vessel_segmentation_array.slurm`
- `submit_batch_segmentation_array.sh`
- `submit_batch_segmentation_array.slurm`
- `submit_graph_pruning_array.sh`
- `submit_graph_pruning_array.slurm`

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

### Batch segmentation and graph pruning

- `submit_batch_segmentation_array.sh`
  - Headnode helper for array submission of vessel segmentation jobs.
- `submit_batch_segmentation_array.slurm`
  - Array worker for segmentation processing.
- `submit_graph_pruning_array.sh`
  - Headnode helper for array submission of graph-pruning jobs.
- `submit_graph_pruning_array.slurm`
  - Array worker for graph-pruning / centerline extraction.

## Recommended Usage

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
